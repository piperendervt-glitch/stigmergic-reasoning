"""
multi_agent_solver.py
=====================
マルチエージェント + Stigmergy 環境層による数学問題ソルバー。

一次方程式で発生している Over-squashing（leaf_norm=0.0000）を
構造的に解決する。

アーキテクチャ:
  MathAgent（軽量）× N体
      ↕ 読み書き
  PheromoneEnvironment（微分可能環境層）
      ↕ GNN拡散

各エージェントは環境を介してのみ間接通信する（Stigmergy）。
root エージェントが堆積したフェロモンは 1 ステップ目から環境全体に
拡散するため、answer エージェントは 6 ホップ待つ必要がない。
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
    evaluate_information_flow,
    NODE_DIM,
    SCALE,
)
from core.root_layer_gnn import GNNDiffusionLayer


# ─────────────────────────────────────────────
#  MathAgent
# ─────────────────────────────────────────────

class MathAgent(nn.Module):
    """
    1ノードを担当する軽量エージェント。

    1ステップの処理:
      1. env から自分のフェロモンを読む       (pheromone_dim,)
      2. concat([node_feat, pheromone]) を encoder に通す
      3. 出力をフェロモンとして env に書く
    """

    def __init__(self, node_dim: int, pheromone_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(node_dim + pheromone_dim, pheromone_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        node_feat: torch.Tensor,    # (node_dim,)
        pheromone: torch.Tensor,     # (pheromone_dim,)
    ) -> torch.Tensor:
        """読み取ったフェロモンとノード特徴量から新しいフェロモンを生成する。"""
        combined = torch.cat([node_feat, pheromone], dim=-1)
        return self.encoder(combined)


# ─────────────────────────────────────────────
#  PheromoneEnvironment（微分可能版）
# ─────────────────────────────────────────────

class GatedGlobalBroadcast(nn.Module):
    """
    Stigmergy設計原則に従ったGated global broadcast。

    設計原則:
      エージェントは環境（フェロモン）とのみ通信する。
      ゲートの入力は local_pheromone と global_signal のみ。

    gate_input = concat([pheromone[i], global_signal])
    gate[i]    = sigmoid(W * gate_input + b)       # (pheromone_dim,)
    output[i]  = pheromone[i] + gate[i] * global_signal

    情報充実ノード → gate≈0 → global を遮断
    情報不足ノード → gate≈1 → global を受け取る
    """

    def __init__(self, pheromone_dim: int):
        super().__init__()
        self.gate_layer = nn.Linear(pheromone_dim * 2, pheromone_dim)

    def forward(self, pheromone: torch.Tensor) -> torch.Tensor:
        N = pheromone.size(0)
        global_signal = pheromone.mean(dim=0, keepdim=True)
        global_expand = global_signal.expand(N, -1)

        gate_input = torch.cat([pheromone, global_expand], dim=-1)
        gate = torch.sigmoid(self.gate_layer(gate_input))

        return pheromone + gate * global_expand


class PheromoneEnvironment(nn.Module):
    """
    微分可能なフェロモン環境層 + Gated Global Broadcast。

    処理順:
      1. エージェントからのフェロモン堆積
      2. GNN拡散
      3. Gated global broadcast（ノードごとにゲート制御）
      4. 蒸発
    """

    def __init__(
        self,
        pheromone_dim: int = 16,
        n_diffuse_steps: int = 2,
        evaporation: float = 0.1,
        diffusion_rate: float = 0.3,
    ):
        super().__init__()
        self.pheromone_dim = pheromone_dim
        self.evaporation = evaporation

        self.diffusion_layers = nn.ModuleList([
            GNNDiffusionLayer(pheromone_dim, diffusion_rate)
            for _ in range(n_diffuse_steps)
        ])

        self.gated_broadcast = GatedGlobalBroadcast(pheromone_dim)

    def step(
        self,
        pheromone: torch.Tensor,
        deposits: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        # 1. 堆積
        pheromone = pheromone + deposits

        # 2. GNN拡散
        for layer in self.diffusion_layers:
            pheromone = layer(pheromone, adj)

        # 3. Gated global broadcast
        pheromone = self.gated_broadcast(pheromone)

        # 4. 蒸発
        pheromone = pheromone * (1.0 - self.evaporation)

        return pheromone


# ─────────────────────────────────────────────
#  MultiAgentMathSolver
# ─────────────────────────────────────────────

class MultiAgentMathSolver(nn.Module):
    """
    複数 MathAgent + PheromoneEnvironment の統合システム。

    Parameters
    ----------
    node_dim       : ノード特徴量次元 (32)
    pheromone_dim  : フェロモン次元 (16)
    n_steps        : エージェントが動くステップ数 (4)
    evaporation    : フェロモン蒸発率 (0.1)
    diffusion_rate : フェロモン拡散率 (0.3)
    """

    def __init__(
        self,
        node_dim: int = NODE_DIM,
        pheromone_dim: int = 16,
        n_steps: int = 4,
        evaporation: float = 0.1,
        diffusion_rate: float = 0.3,
    ):
        super().__init__()
        self.node_dim = node_dim
        self.pheromone_dim = pheromone_dim
        self.n_steps = n_steps

        # 全エージェントが共有するエンコーダ（weight-sharing）
        self.agent = MathAgent(node_dim, pheromone_dim)

        # 環境層
        self.env = PheromoneEnvironment(
            pheromone_dim=pheromone_dim,
            n_diffuse_steps=2,
            evaporation=evaporation,
            diffusion_rate=diffusion_rate,
        )

        # 回帰デコーダ（答えエージェントのフェロモンから数値を予測）
        self.decoder = nn.Sequential(
            nn.Linear(pheromone_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        # ステップ履歴（可視化・分析用）
        self.step_history: list[torch.Tensor] = []

    def forward(self, problem: MathProblemGraph) -> dict:
        """
        問題グラフを受け取り、マルチエージェント推論を実行する。

        Returns
        -------
        {
          'predicted_value' : float   デコーダによる予測値
          'true_answer'     : float   正解
          'error'           : float   予測誤差
          'final_pheromone' : (N, pheromone_dim) 最終フェロモン場
          'step_history'    : list    各ステップのフェロモン場
        }
        """
        N = problem.x.size(0)
        node_feats = problem.x  # (N, node_dim)
        adj = problem.adj       # (N, N)
        ans_idx = problem.answer_node

        # フェロモン場をゼロ初期化
        pheromone = torch.zeros(N, self.pheromone_dim)
        self.step_history = [pheromone.detach().clone()]

        # n_steps 回ループ
        for step in range(self.n_steps):
            # a. 全エージェントが並列にフェロモンを読み、出力を計算
            deposits = self._all_agents_act(node_feats, pheromone)

            # b+c. 堆積 + 拡散 + 蒸発
            pheromone = self.env.step(pheromone, deposits, adj)

            self.step_history.append(pheromone.detach().clone())

        # 答えエージェントのフェロモンから予測
        answer_pheromone = pheromone[ans_idx]  # (pheromone_dim,)
        pred_scaled = self.decoder(answer_pheromone).squeeze(-1)
        predicted = pred_scaled.item() * SCALE

        return {
            'predicted_value': predicted,
            'true_answer': problem.answer_value,
            'error': abs(predicted - problem.answer_value),
            'final_pheromone': pheromone,
            'step_history': self.step_history,
        }

    def forward_train(self, problem: MathProblemGraph) -> tuple:
        """
        訓練用 forward。予測テンソル（勾配付き）とフェロモン場を返す。

        Returns: (pred_scaled, final_pheromone)
        """
        N = problem.x.size(0)
        node_feats = problem.x
        adj = problem.adj

        pheromone = torch.zeros(N, self.pheromone_dim)

        for step in range(self.n_steps):
            deposits = self._all_agents_act(node_feats, pheromone)
            pheromone = self.env.step(pheromone, deposits, adj)

        answer_pheromone = pheromone[problem.answer_node]
        pred_scaled = self.decoder(answer_pheromone).squeeze(-1)

        return pred_scaled, pheromone

    def _all_agents_act(
        self,
        node_feats: torch.Tensor,
        pheromone: torch.Tensor,
    ) -> torch.Tensor:
        """全エージェントが同時に行動する（バッチ処理）。"""
        # (N, node_dim) と (N, pheromone_dim) を結合して一括処理
        combined = torch.cat([node_feats, pheromone], dim=-1)  # (N, node_dim+pheromone_dim)
        deposits = self.agent.encoder(combined)                # (N, pheromone_dim)
        return deposits


# ─────────────────────────────────────────────
#  訓練
# ─────────────────────────────────────────────

def train_multi_agent(
    n_samples: int = 1000,
    n_epochs: int = 200,
    lr: float = 1e-3,
    lambda_contrastive: float = 0.5,
    seed: int = 42,
) -> tuple[MultiAgentMathSolver, dict]:
    """マルチエージェントソルバーを訓練する。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("=" * 65)
    print("Multi-Agent Stigmergy Solver — Training")
    print("=" * 65)
    print()

    print(f"Generating data (n={n_samples})...")
    problems = generate_training_data(n_samples, seed)
    split = int(len(problems) * 0.8)
    train_problems = problems[:split]
    test_problems = problems[split:]
    print(f"  Train: {len(train_problems)}, Test: {len(test_problems)}")
    print()

    solver = MultiAgentMathSolver()
    optimizer = torch.optim.Adam(solver.parameters(), lr=lr)

    history: dict[str, list] = {
        'epoch': [], 'reg_loss': [], 'con_loss': [], 'total_loss': [],
    }

    solver.train()
    for epoch in range(1, n_epochs + 1):
        epoch_reg = 0.0
        epoch_con = 0.0

        for prob in train_problems:
            optimizer.zero_grad()

            pred_scaled, final_pheromone = solver.forward_train(prob)

            true_scaled = torch.tensor(
                prob.answer_value / SCALE, dtype=torch.float32
            )

            reg_loss = answer_regression_loss(pred_scaled, true_scaled)
            con_loss = path_contrastive_loss(final_pheromone, prob)

            loss = reg_loss + lambda_contrastive * con_loss
            loss.backward()
            optimizer.step()

            epoch_reg += reg_loss.item()
            epoch_con += con_loss.item()

        n = len(train_problems)
        epoch_reg /= n
        epoch_con /= n

        if epoch % 20 == 0 or epoch == 1:
            history['epoch'].append(epoch)
            history['reg_loss'].append(epoch_reg)
            history['con_loss'].append(epoch_con)
            history['total_loss'].append(epoch_reg + lambda_contrastive * epoch_con)

            print(
                f"  Epoch {epoch:4d} | "
                f"Reg={epoch_reg:.4f}  Con={epoch_con:.4f}"
            )

    # テスト評価
    solver.eval()
    test_errors = []
    for prob in test_problems:
        result = solver(prob)
        test_errors.append(result['error'])

    mean_err = np.mean(test_errors)
    median_err = np.median(test_errors)
    print(f"\n  Test error: mean={mean_err:.4f}, median={median_err:.4f}")

    save_path = 'trained_multi_agent.pt'
    torch.save({
        'model_state_dict': solver.state_dict(),
        'history': history,
        'n_samples': n_samples,
        'n_epochs': n_epochs,
    }, save_path)
    print(f"  Saved to {save_path}")

    return solver, history


# ─────────────────────────────────────────────
#  比較実験
# ─────────────────────────────────────────────

def _measure_leaf_jacobian(model, problem, is_multi_agent=False):
    """答えノード出力の root ノード入力に対する Jacobian ノルムを測定する。"""
    model.eval()
    x = problem.x.clone().requires_grad_(True)
    ans_idx = problem.answer_node

    if is_multi_agent:
        # MultiAgentMathSolver: pheromone 空間での感度
        N = x.size(0)
        pheromone = torch.zeros(N, model.pheromone_dim)

        for step in range(model.n_steps):
            combined = torch.cat([x, pheromone], dim=-1)
            deposits = model.agent.encoder(combined)
            pheromone = model.env.step(pheromone, deposits, problem.adj)

        out = pheromone
    else:
        # MathSolverGNN: GNN output 空間での感度
        out = model.gnn(x, problem.adj)

    h_answer = out[ans_idx]  # (dim,)
    dim = h_answer.size(0)

    # root (Node 0) に対する Jacobian
    grads = []
    for d in range(dim):
        if x.grad is not None:
            x.grad.zero_()
        h_answer[d].backward(retain_graph=True)
        grads.append(x.grad[0].clone())  # Node 0 の勾配

    jac = torch.stack(grads)
    root_jac_norm = jac.norm().item()

    # 自己参照（正規化用）
    grads_self = []
    for d in range(dim):
        x.grad.zero_()
        h_answer[d].backward(retain_graph=True)
        grads_self.append(x.grad[ans_idx].clone())

    jac_self = torch.stack(grads_self)
    self_norm = jac_self.norm().item()

    if self_norm > 1e-8:
        normalized = root_jac_norm / self_norm
    else:
        normalized = 0.0

    return root_jac_norm, normalized


def _measure_convergence_steps(solver: MultiAgentMathSolver, problem):
    """answer エージェントのフェロモンが安定するまでのステップ数を測定する。"""
    solver.eval()
    N = problem.x.size(0)
    ans_idx = problem.answer_node

    max_steps = 20
    pheromone = torch.zeros(N, solver.pheromone_dim)
    prev_answer_ph = pheromone[ans_idx].clone()
    convergence_step = max_steps

    with torch.no_grad():
        for step in range(1, max_steps + 1):
            combined = torch.cat([problem.x, pheromone], dim=-1)
            deposits = solver.agent.encoder(combined)
            pheromone = solver.env.step(pheromone, deposits, problem.adj)

            answer_ph = pheromone[ans_idx]
            delta = (answer_ph - prev_answer_ph).norm().item()

            if delta < 0.01 and step > 1:
                convergence_step = step
                break

            prev_answer_ph = answer_ph.clone()

    return convergence_step


def _measure_all_jacobians(model, problem, is_multi_agent=False):
    """全ノードの Jacobian ノルムを測定する。"""
    model.eval()
    x = problem.x.clone().requires_grad_(True)
    ans_idx = problem.answer_node
    N = x.size(0)

    if is_multi_agent:
        pheromone = torch.zeros(N, model.pheromone_dim)
        for step in range(model.n_steps):
            combined = torch.cat([x, pheromone], dim=-1)
            deposits = model.agent.encoder(combined)
            pheromone = model.env.step(pheromone, deposits, problem.adj)
        out = pheromone
    else:
        out = model.gnn(x, problem.adj)

    h_answer = out[ans_idx]
    dim = h_answer.size(0)

    all_norms = []
    for node_i in range(N):
        grads = []
        for d in range(dim):
            if x.grad is not None:
                x.grad.zero_()
            h_answer[d].backward(retain_graph=True)
            grads.append(x.grad[node_i].clone())
        jac = torch.stack(grads)
        all_norms.append(jac.norm().item())

    self_norm = all_norms[ans_idx] if all_norms[ans_idx] > 1e-12 else 1.0
    normalized = [n / self_norm for n in all_norms]

    return all_norms, normalized


def measure_squashing_comparison(
    multi_solver: MultiAgentMathSolver,
    single_solver: MathSolverGNN = None,
):
    """単一GNN と マルチエージェントの Over-squashing 比較。"""
    prob = build_linear_equation(2, 3, 7)

    # 単一GNN
    if single_solver is None:
        torch.manual_seed(42)
        single_solver = MathSolverGNN(node_dim=NODE_DIM, hidden_dim=64, num_layers=4)

    s_raw, s_norm = _measure_all_jacobians(single_solver, prob, False)
    single_solver.eval()
    single_result = single_solver(prob)
    single_error = single_result['error']

    # マルチエージェント
    m_raw, m_norm = _measure_all_jacobians(multi_solver, prob, True)
    multi_solver.eval()
    multi_result = multi_solver(prob)
    multi_error = multi_result['error']

    # 収束ステップ
    conv_steps = _measure_convergence_steps(multi_solver, prob)

    # ── 全ノード感度マップ ─────────────────────────────
    print()
    print("=" * 72)
    print("  Over-squashing Comparison: 2x+3=7")
    print("=" * 72)

    G = nx.from_numpy_array(prob.adj.numpy())
    ans_idx = prob.answer_node

    print()
    print("  Per-node Jacobian sensitivity (raw |J|_F)")
    print(
        f"  {'Node':>5} {'Label':>10} {'hop':>4}  "
        f"{'SingleGNN':>12}  {'MultiAgent':>12}  {'ratio':>8}"
    )
    print("  " + "-" * 58)

    for i in range(prob.x.size(0)):
        try:
            hop = nx.shortest_path_length(G, i, ans_idx)
        except nx.NetworkXNoPath:
            hop = -1
        label = prob.node_labels[i][:10] if i < len(prob.node_labels) else f"n{i}"

        s_val = s_raw[i]
        m_val = m_raw[i]
        if s_val > 1e-12:
            ratio = m_val / s_val
        elif m_val > 1e-12:
            ratio = float('inf')
        else:
            ratio = 1.0

        # 科学表記で微小値も見える
        print(
            f"  {i:>5} {label:>10} {hop:>4}  "
            f"{s_val:>12.2e}  {m_val:>12.2e}  {ratio:>8.1f}x"
        )

    # ── サマリー表 ──────────────────────────────────────
    single_leaf_norm = s_norm[0]    # root = Node 0 (farthest)
    multi_leaf_norm = m_norm[0]

    single_verdict = "Squash" if single_leaf_norm < 0.01 else "OK"
    multi_verdict = "Squash" if multi_leaf_norm < 0.01 else "OK"

    print()
    print(f"  {'':24} {'Single GNN':>12}  {'Multi-Agent':>12}")
    print("  " + "-" * 52)
    print(f"  {'leaf_influence (norm)' :<24} {single_leaf_norm:>12.6f}  {multi_leaf_norm:>12.6f}")
    print(f"  {'leaf_influence (raw)'  :<24} {s_raw[0]:>12.2e}  {m_raw[0]:>12.2e}")
    print(f"  {'self_norm (raw)'       :<24} {s_raw[ans_idx]:>12.2e}  {m_raw[ans_idx]:>12.2e}")
    print(f"  {'regression_error'      :<24} {single_error:>12.4f}  {multi_error:>12.4f}")
    print(f"  {'convergence_steps'     :<24} {'N/A':>12}  {conv_steps:>12}")
    print(f"  {'verdict'               :<24} {single_verdict:>12}  {multi_verdict:>12}")
    print()

    # 改善倍率
    if s_raw[0] > 1e-12:
        improvement = m_raw[0] / s_raw[0]
        print(f"  Root sensitivity improvement: {improvement:.1f}x")
    elif m_raw[0] > 1e-12:
        print(f"  Root sensitivity improvement: inf (single=0)")
    else:
        print(f"  Root sensitivity: both near zero")

    return {
        'single_leaf_norm': single_leaf_norm,
        'multi_leaf_norm': multi_leaf_norm,
        'single_leaf_raw': s_raw[0],
        'multi_leaf_raw': m_raw[0],
        'single_error': single_error,
        'multi_error': multi_error,
        'convergence': conv_steps,
    }


# ─────────────────────────────────────────────
#  メイン
# ─────────────────────────────────────────────

def main():
    import os

    # マルチエージェント: 保存済みがあればロード、なければ訓練
    multi_path = 'trained_multi_agent.pt'
    if os.path.exists(multi_path):
        multi_solver = MultiAgentMathSolver()
        checkpoint = torch.load(multi_path, weights_only=False)
        multi_solver.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded trained multi-agent from {multi_path}")
    else:
        multi_solver, _ = train_multi_agent(
            n_samples=1000, n_epochs=200, lr=1e-3, seed=42,
        )

    # 単一GNN: 保存済みがあればロード
    single_solver = MathSolverGNN(node_dim=NODE_DIM, hidden_dim=64, num_layers=4)
    model_path = 'trained_model.pt'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, weights_only=False)
        single_solver.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded trained single GNN from {model_path}")

    # 比較実験
    measure_squashing_comparison(multi_solver, single_solver)


if __name__ == '__main__':
    main()
