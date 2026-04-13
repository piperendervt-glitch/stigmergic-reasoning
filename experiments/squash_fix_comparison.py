"""
squash_fix_comparison.py
========================
一次方程式の Over-squashing を5手法で比較する。

手法:
  0. Baseline（単一GNN、元のグラフ）
  1. Global pooling + broadcast
  2. Graph rewiring（ショートカットエッジ追加）
  3. Virtual node（全結合仮想ノード追加）
  4. Gated broadcast（ノードごとにゲート制御）
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import random

from solvers.math_problem_gnn import (
    MathProblemGraph,
    MathSolverGNN,
    build_linear_equation,
    build_quadratic_equation,
    build_simultaneous_equations,
    generate_training_data,
    answer_regression_loss,
    path_contrastive_loss,
    NODE_DIM,
    SCALE,
)
from core.root_layer_gnn import GNNDiffusionLayer
from solvers.multi_agent_solver import (
    MathAgent,
    PheromoneEnvironment,
    GatedGlobalBroadcast,
    MultiAgentMathSolver,
)


# ─────────────────────────────────────────────
#  共通: Jacobian 測定
# ─────────────────────────────────────────────

def measure_leaf_jacobian(forward_fn, problem):
    """answer ノード出力の root (Node 0) に対する Jacobian ノルムを計測する。"""
    x = problem.x.clone().requires_grad_(True)
    ans_idx = problem.answer_node

    h_answer = forward_fn(x, problem)
    D = h_answer.size(0)

    grads_root = []
    for d in range(D):
        if x.grad is not None:
            x.grad.zero_()
        h_answer[d].backward(retain_graph=True)
        grads_root.append(x.grad[0].clone())
    root_norm = torch.stack(grads_root).norm().item()

    grads_self = []
    for d in range(D):
        x.grad.zero_()
        h_answer[d].backward(retain_graph=True)
        grads_self.append(x.grad[ans_idx].clone())
    self_norm = torch.stack(grads_self).norm().item()

    normalized = root_norm / self_norm if self_norm > 1e-12 else 0.0
    return root_norm, normalized


# ─────────────────────────────────────────────
#  共通: Over-smoothing 測定
# ─────────────────────────────────────────────

def measure_os_start_gnn(problem, max_layers=8, seed=42):
    """単一GNN の OS開始層を測定する。"""
    for n_layers in range(1, max_layers + 1):
        torch.manual_seed(seed)
        model = MathSolverGNN(node_dim=NODE_DIM, hidden_dim=64, num_layers=n_layers)
        model.eval()
        with torch.no_grad():
            model.gnn(problem.x, problem.adj, record_history=True)
        final_h = model.gnn.signal_history[-1]
        N = final_h.size(0)
        norms = final_h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normed = final_h / norms
        sim_matrix = normed @ normed.T
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        score = sim_matrix[mask].mean().item()
        if score > 0.9:
            return n_layers
    return max_layers + 1


def measure_os_start_multiagent(solver_cls, problem, max_steps=8, seed=42, **kwargs):
    """マルチエージェント系の OS開始層を測定する。"""
    for n_steps in range(1, max_steps + 1):
        torch.manual_seed(seed)
        solver = solver_cls(n_steps=n_steps, **kwargs)
        solver.eval()
        with torch.no_grad():
            pheromone = solver._forward_pheromone(problem.x, problem)
        N = pheromone.size(0)
        norms = pheromone.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normed = pheromone / norms
        sim_matrix = normed @ normed.T
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        score = sim_matrix[mask].mean().item()
        if score > 0.9:
            return n_steps
    return max_steps + 1


# ─────────────────────────────────────────────
#  ベースライン（単一GNN）
# ─────────────────────────────────────────────

def _train_single_gnn(train_problems, n_epochs, lr, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    solver = MathSolverGNN(node_dim=NODE_DIM, hidden_dim=64, num_layers=4)
    optimizer = torch.optim.Adam(solver.parameters(), lr=lr)
    solver.train()
    for epoch in range(1, n_epochs + 1):
        for prob in train_problems:
            optimizer.zero_grad()
            out = solver.gnn(prob.x, prob.adj)
            pred = solver.decoder(out[prob.answer_node])
            true_val = torch.tensor(prob.answer_value / SCALE)
            loss = F.mse_loss(pred, true_val)
            loss.backward()
            optimizer.step()
        if epoch % 50 == 0:
            print(f"    epoch {epoch}/{n_epochs}")
    solver.eval()
    return solver


# ─────────────────────────────────────────────
#  手法 1: Global pooling（旧: λ=0.3 均一加算）
# ─────────────────────────────────────────────

class _GlobalEnv(PheromoneEnvironment):
    """Global pooling 旧版（λ均一加算、ゲートなし）。比較用に残す。"""
    def __init__(self, pheromone_dim=16, global_lambda=0.3, **kw):
        # GatedGlobalBroadcast 付きの親を初期化
        super().__init__(pheromone_dim=pheromone_dim, **kw)
        self.global_lambda = global_lambda

    def step(self, pheromone, deposits, adj):
        # 堆積 + 拡散
        pheromone = pheromone + deposits
        for layer in self.diffusion_layers:
            pheromone = layer(pheromone, adj)
        # ゲートなし均一加算（旧方式）
        g = pheromone.mean(dim=0, keepdim=True)
        pheromone = pheromone + self.global_lambda * g.expand_as(pheromone)
        # 蒸発
        pheromone = pheromone * (1.0 - self.evaporation)
        return pheromone


class GlobalMultiAgentSolver(nn.Module):
    """Global pooling 付きマルチエージェントソルバー（旧版）。"""
    def __init__(self, node_dim=NODE_DIM, pheromone_dim=16, n_steps=4):
        super().__init__()
        self.node_dim = node_dim
        self.pheromone_dim = pheromone_dim
        self.n_steps = n_steps
        self.encoder = nn.Sequential(
            nn.Linear(node_dim + pheromone_dim * 2, pheromone_dim), nn.ReLU(),
        )
        self.env = _GlobalEnv(pheromone_dim=pheromone_dim)
        self.decoder = nn.Sequential(
            nn.Linear(pheromone_dim, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def _forward_pheromone(self, x, problem):
        N = x.size(0)
        pheromone = torch.zeros(N, self.pheromone_dim)
        for step in range(self.n_steps):
            global_signal = pheromone.mean(dim=0, keepdim=True).expand(N, -1)
            combined = torch.cat([x, pheromone, global_signal], dim=-1)
            deposits = self.encoder(combined)
            pheromone = self.env.step(pheromone, deposits, problem.adj)
        return pheromone

    def forward_for_jacobian(self, x, problem):
        return self._forward_pheromone(x, problem)[problem.answer_node]

    def forward_train(self, x, problem):
        pheromone = self._forward_pheromone(x, problem)
        pred = self.decoder(pheromone[problem.answer_node]).squeeze(-1)
        return pred, pheromone

    def forward(self, problem):
        self.eval()
        with torch.no_grad():
            pheromone = self._forward_pheromone(problem.x, problem)
            pred = self.decoder(pheromone[problem.answer_node]).squeeze(-1)
        return pred.item() * SCALE, problem.answer_value


# ─────────────────────────────────────────────
#  手法 4: Gated broadcast（新方式）
# ─────────────────────────────────────────────

class GatedMultiAgentSolver(nn.Module):
    """Gated global broadcast 付きマルチエージェントソルバー。"""
    def __init__(self, node_dim=NODE_DIM, pheromone_dim=16, n_steps=4):
        super().__init__()
        self.node_dim = node_dim
        self.pheromone_dim = pheromone_dim
        self.n_steps = n_steps
        # エージェント: concat([node_feat, local_pheromone]) — global なし
        self.encoder = nn.Sequential(
            nn.Linear(node_dim + pheromone_dim, pheromone_dim), nn.ReLU(),
        )
        self.env = PheromoneEnvironment(pheromone_dim=pheromone_dim)
        self.decoder = nn.Sequential(
            nn.Linear(pheromone_dim, 32), nn.ReLU(), nn.Linear(32, 1),
        )

    def _forward_pheromone(self, x, problem):
        N = x.size(0)
        pheromone = torch.zeros(N, self.pheromone_dim)
        for step in range(self.n_steps):
            combined = torch.cat([x, pheromone], dim=-1)
            deposits = self.encoder(combined)
            pheromone = self.env.step(pheromone, deposits, problem.adj)
        return pheromone

    def forward_for_jacobian(self, x, problem):
        return self._forward_pheromone(x, problem)[problem.answer_node]

    def forward_train(self, x, problem):
        pheromone = self._forward_pheromone(x, problem)
        pred = self.decoder(pheromone[problem.answer_node]).squeeze(-1)
        return pred, pheromone

    def forward(self, problem):
        self.eval()
        with torch.no_grad():
            pheromone = self._forward_pheromone(problem.x, problem)
            pred = self.decoder(pheromone[problem.answer_node]).squeeze(-1)
        return pred.item() * SCALE, problem.answer_value


# ─────────────────────────────────────────────
#  手法 2: Graph rewiring
# ─────────────────────────────────────────────

def build_linear_equation_rewired(a, b, c):
    prob = build_linear_equation(a, b, c)
    adj = prob.adj.clone()
    adj[0][6] = adj[6][0] = 1.0
    adj[0][4] = adj[4][0] = 1.0
    return MathProblemGraph(
        name=prob.name + " (rewired)", problem_str=prob.problem_str,
        x=prob.x, adj=adj, answer_node=prob.answer_node,
        answer_value=prob.answer_value, node_labels=prob.node_labels,
    )


def _generate_rewired_data(n_samples, seed):
    rng = random.Random(seed)
    problems = []
    for _ in range(n_samples):
        a = rng.randint(1, 9)
        b = rng.randint(-9, 9)
        c = rng.randint(-20, 20)
        problems.append(build_linear_equation_rewired(float(a), float(b), float(c)))
    rng.shuffle(problems)
    return problems


# ─────────────────────────────────────────────
#  手法 3: Virtual node
# ─────────────────────────────────────────────

def add_virtual_node(problem):
    N = problem.x.size(0)
    virt_feat = torch.zeros(1, NODE_DIM)
    new_x = torch.cat([problem.x, virt_feat], dim=0)
    new_adj = torch.zeros(N + 1, N + 1)
    new_adj[:N, :N] = problem.adj
    for i in range(N):
        new_adj[i][N] = 1.0
        new_adj[N][i] = 1.0
    new_labels = list(problem.node_labels) + ["Virtual"]
    return MathProblemGraph(
        name=problem.name + " (+vnode)", problem_str=problem.problem_str,
        x=new_x, adj=new_adj, answer_node=problem.answer_node,
        answer_value=problem.answer_value, node_labels=new_labels,
    )


def _generate_vnode_data(n_samples, seed):
    rng = random.Random(seed)
    problems = []
    for _ in range(n_samples):
        a = rng.randint(1, 9)
        b = rng.randint(-9, 9)
        c = rng.randint(-20, 20)
        problems.append(add_virtual_node(build_linear_equation(float(a), float(b), float(c))))
    rng.shuffle(problems)
    return problems


# ─────────────────────────────────────────────
#  訓練ヘルパー
# ─────────────────────────────────────────────

def _train_multiagent(solver, train_problems, n_epochs, lr):
    optimizer = torch.optim.Adam(solver.parameters(), lr=lr)
    solver.train()
    for epoch in range(1, n_epochs + 1):
        for prob in train_problems:
            optimizer.zero_grad()
            pred, pheromone = solver.forward_train(prob.x, prob)
            true_val = torch.tensor(prob.answer_value / SCALE)
            reg = F.mse_loss(pred, true_val)
            con = path_contrastive_loss(pheromone, prob)
            (reg + 0.5 * con).backward()
            optimizer.step()
        if epoch % 50 == 0:
            print(f"    epoch {epoch}/{n_epochs}")
    solver.eval()
    return solver


# ─────────────────────────────────────────────
#  比較実験
# ─────────────────────────────────────────────

def run_squash_fix_comparison(
    n_samples: int = 1000,
    n_epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 42,
):
    test_a, test_b, test_c = 2.0, 3.0, 7.0
    test_prob = build_linear_equation(test_a, test_b, test_c)

    rng = random.Random(seed)
    base_problems = []
    for _ in range(n_samples):
        a = rng.randint(1, 9)
        b = rng.randint(-9, 9)
        c = rng.randint(-20, 20)
        base_problems.append(build_linear_equation(float(a), float(b), float(c)))
    rng.shuffle(base_problems)
    split = int(len(base_problems) * 0.8)
    train_base = base_problems[:split]

    results = []  # (name, leaf_norm, error, os_start)

    # ── 0. Baseline ──
    print("  [1/5] Baseline 訓練中...")
    solver_bl = _train_single_gnn(train_base, n_epochs, lr, seed)
    def fwd_bl(x, prob): return solver_bl.gnn(x, prob.adj)[prob.answer_node]
    bl_raw, _ = measure_leaf_jacobian(fwd_bl, test_prob)
    bl_err = solver_bl(test_prob)['error']
    bl_os = measure_os_start_gnn(test_prob, seed=seed)
    results.append(("Baseline", bl_raw, bl_err, bl_os))

    # ── 1. Global pooling ──
    print("  [2/5] Global pooling 訓練中...")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    solver_gp = GlobalMultiAgentSolver()
    solver_gp = _train_multiagent(solver_gp, train_base, n_epochs, lr)
    def fwd_gp(x, prob): return solver_gp.forward_for_jacobian(x, prob)
    gp_raw, _ = measure_leaf_jacobian(fwd_gp, test_prob)
    gp_pred, gp_true = solver_gp(test_prob)
    gp_err = abs(gp_pred - gp_true)
    gp_os = measure_os_start_multiagent(GlobalMultiAgentSolver, test_prob, seed=seed)
    results.append(("Global pooling", gp_raw, gp_err, gp_os))

    # ── 2. Graph rewiring ──
    print("  [3/5] Graph rewiring 訓練中...")
    rewired_data = _generate_rewired_data(n_samples, seed)
    train_rw = rewired_data[:split]
    solver_rw = _train_single_gnn(train_rw, n_epochs, lr, seed)
    test_rw = build_linear_equation_rewired(test_a, test_b, test_c)
    def fwd_rw(x, prob): return solver_rw.gnn(x, prob.adj)[prob.answer_node]
    rw_raw, _ = measure_leaf_jacobian(fwd_rw, test_rw)
    rw_err = solver_rw(test_rw)['error']
    rw_os = measure_os_start_gnn(test_rw, seed=seed)
    results.append(("Graph rewiring", rw_raw, rw_err, rw_os))

    # ── 3. Virtual node ──
    print("  [4/5] Virtual node 訓練中...")
    vnode_data = _generate_vnode_data(n_samples, seed)
    train_vn = vnode_data[:split]
    solver_vn = _train_single_gnn(train_vn, n_epochs, lr, seed)
    test_vn = add_virtual_node(build_linear_equation(test_a, test_b, test_c))
    def fwd_vn(x, prob): return solver_vn.gnn(x, prob.adj)[prob.answer_node]
    vn_raw, _ = measure_leaf_jacobian(fwd_vn, test_vn)
    vn_err = solver_vn(test_vn)['error']
    vn_os = measure_os_start_gnn(test_vn, seed=seed)
    results.append(("Virtual node", vn_raw, vn_err, vn_os))

    # ── 4. Gated broadcast ──
    print("  [5/5] Gated broadcast 訓練中...")
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    solver_gb = GatedMultiAgentSolver()
    solver_gb = _train_multiagent(solver_gb, train_base, n_epochs, lr)
    def fwd_gb(x, prob): return solver_gb.forward_for_jacobian(x, prob)
    gb_raw, _ = measure_leaf_jacobian(fwd_gb, test_prob)
    gb_pred, gb_true = solver_gb(test_prob)
    gb_err = abs(gb_pred - gb_true)
    gb_os = measure_os_start_multiagent(GatedMultiAgentSolver, test_prob, seed=seed)
    results.append(("Gated broadcast", gb_raw, gb_err, gb_os))

    # ── 結果表 ──
    print()
    print("=" * 68)
    print("  Over-squashing Fix Comparison (updated): 2x+3=7")
    print("=" * 68)
    print()
    print(f"  {'Method':<22} {'leaf_norm':>12} {'error':>10} {'OS_start':>10} {'pass?':>7}")
    print("  " + "-" * 65)
    for name, raw, err, os_s in results:
        os_str = str(os_s) if os_s <= 8 else ">8"
        leaf_ok = raw > 0.01
        os_ok = os_s >= 4
        if leaf_ok and os_ok:
            verdict = "PASS"
        elif leaf_ok:
            verdict = "OS_bad"
        elif os_ok:
            verdict = "SQ_bad"
        else:
            verdict = "FAIL"
        print(f"  {name:<22} {raw:>12.2e} {err:>10.4f} {os_str:>10} {verdict:>7}")
    print("  " + "-" * 65)
    print("  PASS: leaf_norm > 0.01 AND OS_start >= 4")
    print()

    return results


# ─────────────────────────────────────────────
#  ボトルネック再測定
# ─────────────────────────────────────────────

def run_bottleneck_comparison():
    """Gated broadcast の全タスクでのボトルネック測定。"""
    problems = {
        'linear': build_linear_equation(2, 3, 7),
        'quadratic': build_quadratic_equation(1, -5, 6),
        'simultaneous': build_simultaneous_equations(1, 1, 5, 1, -1, 1),
    }

    # Global pooling 参照値
    gp_ref = {
        'linear':       {'os': 1, 'leaf': 0.62},
        'quadratic':    {'os': 1, 'leaf': 0.64},
        'simultaneous': {'os': 1, 'leaf': 0.59},
    }

    print("=" * 72)
    print("  Bottleneck Re-measurement: Gated broadcast")
    print("=" * 72)
    print()

    print(f"  {'':16} {'-- Global pooling --':>24}  {'-- Gated broadcast --':>24}  {'verdict'}")
    print(f"  {'Task':<16} {'OS':>6} {'leaf_norm':>12}    {'OS':>6} {'leaf_norm':>12}  ")
    print("  " + "-" * 68)

    for task_name, prob in problems.items():
        gp = gp_ref[task_name]

        # Gated broadcast: OS
        gb_os = measure_os_start_multiagent(
            GatedMultiAgentSolver, prob, max_steps=8, seed=42,
        )

        # Gated broadcast: leaf_norm
        torch.manual_seed(42)
        solver = GatedMultiAgentSolver(n_steps=4)
        solver.eval()
        def fwd(x, p, _s=solver): return _s.forward_for_jacobian(x, p)
        gb_leaf, _ = measure_leaf_jacobian(fwd, prob)

        gb_os_str = str(gb_os) if gb_os <= 8 else ">8"
        gp_os_str = str(gp['os'])

        leaf_ok = gb_leaf > 0.01
        os_ok = gb_os >= 4
        if leaf_ok and os_ok:
            verdict = "PASS"
        elif leaf_ok:
            verdict = "OS_bad"
        elif os_ok:
            verdict = "SQ_bad"
        else:
            verdict = "FAIL"

        print(
            f"  {task_name:<16} {gp_os_str:>6} {gp['leaf']:>12.2e}    "
            f"{gb_os_str:>6} {gb_leaf:>12.2e}  {verdict}"
        )

    print("  " + "-" * 68)
    print("  PASS: OS_start >= 4 AND leaf_norm > 0.01")
    print()


if __name__ == '__main__':
    run_squash_fix_comparison()
    run_bottleneck_comparison()
