"""
hgnn_solver.py
==============
Hypergraph Neural Network (HGNN) ベースの数学問題ソルバー。

超エッジの設計でグラフ直径を圧縮し、
Over-squashing と Over-smoothing を同時解消する。
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import random
import math

from solvers.math_problem_gnn import (
    MathProblemGraph,
    MathSolverGNN,
    AnswerDecoder,
    build_linear_equation,
    build_quadratic_equation,
    build_simultaneous_equations,
    generate_training_data,
    answer_regression_loss,
    path_contrastive_loss,
    dependency_loss,
    NODE_DIM,
    SCALE,
)
from experiments.squash_fix_comparison import measure_leaf_jacobian


# ─────────────────────────────────────────────
#  HyperedgeLayer
# ─────────────────────────────────────────────

class HyperedgeLayer(nn.Module):
    """
    HGNN の1層。ノード→超エッジ→ノードの2段階伝達。

    Step 1: ノード → 超エッジ (集約)
    Step 2: 超エッジ → ノード (配布 + 残差 + LayerNorm + ReLU)
    """

    def __init__(self, node_dim: int, hyperedge_dim: int):
        super().__init__()
        self.W_e = nn.Linear(node_dim, hyperedge_dim, bias=False)
        self.W_n = nn.Linear(hyperedge_dim, node_dim, bias=False)
        self.norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        h: torch.Tensor,
        hyperedges: list[list[int]],
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h          : (N, node_dim)
        hyperedges : list of list of int (各超エッジに属するノードID)

        Returns
        -------
        h_new : (N, node_dim)
        """
        N, D = h.size()
        E = len(hyperedges)

        # Step 1: ノード → 超エッジ
        e_feats = []
        for edge_nodes in hyperedges:
            e_feat = h[edge_nodes].mean(dim=0)
            e_feats.append(e_feat)
        e_feats = torch.stack(e_feats)           # (E, node_dim)
        e_feats = self.W_e(e_feats)              # (E, hyperedge_dim)

        # Step 2: 超エッジ → ノード
        # 各ノードが属する超エッジを集約
        node_agg = torch.zeros(N, e_feats.size(-1), device=h.device)
        node_count = torch.zeros(N, 1, device=h.device)
        for e_idx, edge_nodes in enumerate(hyperedges):
            for v in edge_nodes:
                node_agg[v] += e_feats[e_idx]
                node_count[v] += 1.0

        node_count = node_count.clamp(min=1.0)
        node_agg = node_agg / node_count          # (N, hyperedge_dim)

        h_new = self.W_n(node_agg)                # (N, node_dim)
        h_new = self.norm(h_new + h)              # 残差接続
        return F.relu(h_new)


# ─────────────────────────────────────────────
#  HypergraphBuilder
# ─────────────────────────────────────────────

class HypergraphBuilder:
    """問題タイプに応じた超エッジを定義する。"""

    @staticmethod
    def linear_equation(n_nodes: int = 7) -> list[list[int]]:
        return [
            [0, 6],           # root ↔ answer（直径解消）
            [0, 1, 2, 3],    # 問題分解ステップ
            [3, 4, 5, 6],    # 変形・解導出ステップ
            [0, 3],          # root ↔ 中間ノード
        ]

    @staticmethod
    def quadratic_equation(n_nodes: int = 5) -> list[list[int]]:
        return [
            [0, 2, 3, 4],    # root ↔ 解 ↔ answer
            [0, 1],          # root ↔ 判別式
            [1, 2, 3],       # 判別式 ↔ 各解
        ]

    @staticmethod
    def simultaneous_equations(n_nodes: int = 7) -> list[list[int]]:
        return [
            [0, 3, 6],       # root ↔ 消去 ↔ answer
            [1, 2, 3],       # 2式 ↔ 消去ノード
            [3, 4, 5, 6],    # 消去後の解導出
            [0, 6],          # root ↔ answer 直接補強
            [0, 4, 5, 6],    # root ↔ 解ノードx,y ↔ answer 経路強化
        ]


def get_hyperedges(problem: MathProblemGraph) -> list[list[int]]:
    """問題タイプに応じた超エッジを返す。"""
    N = problem.x.size(0)
    s = problem.problem_str
    if N == 5:
        return HypergraphBuilder.quadratic_equation(N)
    elif '/' in s or ('y' in s.lower() and N == 7):
        return HypergraphBuilder.simultaneous_equations(N)
    else:
        return HypergraphBuilder.linear_equation(N)


# ─────────────────────────────────────────────
#  HGNNMathSolver
# ─────────────────────────────────────────────

class HGNNStigmergyLayer(nn.Module):
    """HGNN 版の Stigmergy Layer。超エッジで情報伝達する。"""

    def __init__(self, node_dim: int = 32, hidden_dim: int = 64, num_layers: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.input_proj = nn.Linear(node_dim, hidden_dim)
        self.layers = nn.ModuleList([
            HyperedgeLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, node_dim)
        self.signal_history: list[torch.Tensor] = []

    def forward(
        self,
        x: torch.Tensor,
        hyperedges: list[list[int]],
        record_history: bool = False,
    ) -> torch.Tensor:
        h = F.relu(self.input_proj(x))
        if record_history:
            self.signal_history = [h.detach().clone()]
        for layer in self.layers:
            h = layer(h, hyperedges)
            if record_history:
                self.signal_history.append(h.detach().clone())
        return self.output_proj(h)


class HGNNMathSolver(nn.Module):
    """HGNN ベースの数学問題ソルバー。"""

    def __init__(self, node_dim: int = NODE_DIM, hidden_dim: int = 64, num_layers: int = 4):
        super().__init__()
        self.hgnn = HGNNStigmergyLayer(node_dim, hidden_dim, num_layers)
        self.decoder = AnswerDecoder(node_dim, hidden_dim)

    def forward(self, problem: MathProblemGraph) -> dict:
        self.eval()
        hyperedges = get_hyperedges(problem)
        with torch.no_grad():
            out = self.hgnn(problem.x, hyperedges, record_history=True)
        answer_feat = out[problem.answer_node]
        predicted = float(self.decoder(answer_feat).item()) * SCALE
        return {
            'gnn_output': out,
            'answer_feat': answer_feat,
            'predicted_value': predicted,
            'true_answer': problem.answer_value,
            'error': abs(predicted - problem.answer_value),
        }

    def forward_for_jacobian(self, x, problem):
        hyperedges = get_hyperedges(problem)
        out = self.hgnn(x, hyperedges)
        return out[problem.answer_node]


# ─────────────────────────────────────────────
#  訓練
# ─────────────────────────────────────────────

def train_hgnn(train_problems, n_epochs=200, lr=1e-3, seed=42, silent=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    solver = HGNNMathSolver()
    optimizer = torch.optim.Adam(solver.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5,
    )

    solver.train()
    for epoch in range(1, n_epochs + 1):
        for prob in train_problems:
            optimizer.zero_grad()
            hyperedges = get_hyperedges(prob)
            out = solver.hgnn(prob.x, hyperedges)
            pred = solver.decoder(out[prob.answer_node])
            true_val = torch.tensor(prob.answer_value / SCALE)
            reg = F.mse_loss(pred, true_val)
            con = path_contrastive_loss(out, prob)
            dep = dependency_loss(out, prob)
            (reg + 0.5 * con + 1.0 * dep).backward()
            optimizer.step()
        scheduler.step()
        if not silent and epoch % 50 == 0:
            print(f"    epoch {epoch}/{n_epochs}")

    solver.eval()
    return solver


# ─────────────────────────────────────────────
#  測定ヘルパー
# ─────────────────────────────────────────────

def measure_os_start_hgnn(problem, max_layers=8, seed=42):
    """HGNN の OS開始層を測定する。"""
    hyperedges = get_hyperedges(problem)
    for n_layers in range(1, max_layers + 1):
        torch.manual_seed(seed)
        model = HGNNMathSolver(num_layers=n_layers)
        model.eval()
        with torch.no_grad():
            model.hgnn(problem.x, hyperedges, record_history=True)
        final_h = model.hgnn.signal_history[-1]
        N = final_h.size(0)
        norms = final_h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normed = final_h / norms
        sim_matrix = normed @ normed.T
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        score = sim_matrix[mask].mean().item()
        if score > 0.9:
            return n_layers
    return max_layers + 1


def _gen_task_data(task, n, seed):
    rng = random.Random(seed)
    problems = []
    if task == 'linear':
        for _ in range(n):
            a = rng.randint(1, 9)
            b = rng.randint(-9, 9)
            c = rng.randint(-20, 20)
            problems.append(build_linear_equation(float(a), float(b), float(c)))
    elif task == 'quadratic':
        count = 0
        while count < n:
            b_val = rng.randint(-9, 9)
            c_val = rng.randint(-20, 20)
            if b_val ** 2 - 4 * c_val < 0:
                continue
            problems.append(build_quadratic_equation(1.0, float(b_val), float(c_val)))
            count += 1
    elif task == 'simultaneous':
        count = 0
        while count < n:
            a1, b1 = rng.randint(-5, 5), rng.randint(-5, 5)
            c1 = rng.randint(-10, 10)
            a2, b2 = rng.randint(-5, 5), rng.randint(-5, 5)
            c2 = rng.randint(-10, 10)
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-8:
                continue
            p = build_simultaneous_equations(
                float(a1), float(b1), float(c1),
                float(a2), float(b2), float(c2),
            )
            if math.isnan(p.answer_value):
                continue
            problems.append(p)
            count += 1
    return problems


# ─────────────────────────────────────────────
#  比較実験
# ─────────────────────────────────────────────

def run_hgnn_comparison(n_samples=2000, n_epochs=500, lr=1e-3, seed=42):
    tasks = {
        'linear': build_linear_equation(2, 3, 7),
        'quadratic': build_quadratic_equation(1, -5, 6),
        'simultaneous': build_simultaneous_equations(1, 1, 5, 1, -1, 1),
    }

    # 従来手法の参照値
    refs = [
        ("Baseline",       "linear", 0.00,  0.0004, 6,  "SQ_bad"),
        ("Global pooling", "linear", 0.218, 0.0077, 1,  "OS_bad"),
        ("Graph rewiring", "linear", 0.889, 0.1279, 6,  "PASS"),
        ("Virtual node",   "linear", 0.229, 0.2971, 5,  "PASS"),
    ]

    # HGNN を各タスクで訓練・評価
    hgnn_results = []

    for task_name, test_prob in tasks.items():
        print(f"  HGNN [{task_name}] Training...")
        data = _gen_task_data(task_name, n_samples, seed)
        split = int(len(data) * 0.8)
        train_data = data[:split]

        solver = train_hgnn(train_data, n_epochs, lr, seed)

        # leaf_norm
        def fwd(x, prob, _s=solver):
            return _s.forward_for_jacobian(x, prob)
        leaf_raw, _ = measure_leaf_jacobian(fwd, test_prob)

        # 回帰誤差
        result = solver(test_prob)
        error = result['error']

        # OS開始層
        os_start = measure_os_start_hgnn(test_prob, seed=seed)

        leaf_ok = leaf_raw > 0.01
        os_ok = os_start >= 4
        err_ok = error < 0.05
        if leaf_ok and os_ok and err_ok:
            verdict = "PASS"
        elif leaf_ok and os_ok:
            verdict = "err_hi"
        elif leaf_ok:
            verdict = "OS_bad"
        elif os_ok:
            verdict = "SQ_bad"
        else:
            verdict = "FAIL"

        hgnn_results.append((task_name, leaf_raw, error, os_start, verdict))

    # 結果表
    print()
    print("=" * 72)
    print("  HGNN vs Prior Methods")
    print("=" * 72)
    print()
    print(
        f"  {'Method':<20} {'Task':<14} {'leaf_norm':>10} "
        f"{'error':>8} {'OS':>4} {'verdict':>8}"
    )
    print("  " + "-" * 68)
    for name, task, leaf, err, os_s, verd in refs:
        print(
            f"  {name:<20} {task:<14} {leaf:>10.2e} "
            f"{err:>8.4f} {os_s:>4} {verd:>8}"
        )
    print("  " + "-" * 68)
    for task_name, leaf, err, os_s, verd in hgnn_results:
        os_str = str(os_s) if os_s <= 8 else ">8"
        print(
            f"  {'HGNN':<20} {task_name:<14} {leaf:>10.2e} "
            f"{err:>8.4f} {os_str:>4} {verd:>8}"
        )
    print("  " + "-" * 68)
    print("  PASS: leaf_norm>0.01 AND OS>=4 AND error<0.05")
    print()

    return hgnn_results


# ─────────────────────────────────────────────
#  ボトルネック再測定
# ─────────────────────────────────────────────

def run_bottleneck_hgnn():
    tasks = {
        'linear': build_linear_equation(2, 3, 7),
        'quadratic': build_quadratic_equation(1, -5, 6),
        'simultaneous': build_simultaneous_equations(1, 1, 5, 1, -1, 1),
    }

    before = {
        'linear':       {'os': 6, 'leaf': 0.0},
        'quadratic':    {'os': 5, 'leaf': 0.2305},
        'simultaneous': {'os': 5, 'leaf': 0.1089},
    }

    print("=" * 72)
    print("  Bottleneck Re-measurement: HGNN")
    print("=" * 72)
    print()
    print(
        f"  {'':16} {'--- Baseline GNN ---':>22}  {'--- HGNN ---':>22}  {'verdict'}"
    )
    print(
        f"  {'Task':<16} {'OS':>6} {'leaf_norm':>12}    {'OS':>6} {'leaf_norm':>12}"
    )
    print("  " + "-" * 68)

    for task_name, prob in tasks.items():
        b = before[task_name]

        # HGNN OS
        os_start = measure_os_start_hgnn(prob, seed=42)

        # HGNN leaf_norm
        torch.manual_seed(42)
        solver = HGNNMathSolver(num_layers=4)
        solver.eval()

        def fwd(x, p, _s=solver):
            return _s.forward_for_jacobian(x, p)

        leaf_raw, _ = measure_leaf_jacobian(fwd, prob)

        os_str = str(os_start) if os_start <= 8 else ">8"
        leaf_ok = leaf_raw > 0.01
        os_ok = os_start >= 4
        verdict = "PASS" if (leaf_ok and os_ok) else (
            "OS_bad" if leaf_ok else ("SQ_bad" if os_ok else "FAIL")
        )

        print(
            f"  {task_name:<16} {b['os']:>6} {b['leaf']:>12.2e}    "
            f"{os_str:>6} {leaf_raw:>12.2e}  {verdict}"
        )

    print("  " + "-" * 68)
    print("  PASS: OS >= 4 AND leaf_norm > 0.01")
    print()


# ─────────────────────────────────────────────
#  メイン
# ─────────────────────────────────────────────

if __name__ == '__main__':
    hgnn_results = run_hgnn_comparison()
    run_bottleneck_hgnn()
