"""
bottleneck_analysis.py
======================
GNN の Over-smoothing と Over-squashing を
タスク別（一次・二次・連立方程式）に定量測定する。

測定 1: Over-smoothing — 層数を増やしたときのノード表現の均一化
測定 2: Over-squashing — 答えノードへの各入力ノードの感度 (Jacobian)
測定 3: タスク別サマリー
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx

from core.gnn_stigmergy import GNNStigmergyLayer
from solvers.math_problem_gnn import (
    MathSolverGNN,
    MathProblemGraph,
    build_linear_equation,
    build_quadratic_equation,
    build_simultaneous_equations,
    NODE_DIM,
)


# ─────────────────────────────────────────────
#  測定 1: Over-smoothing
# ─────────────────────────────────────────────

def measure_over_smoothing(
    problem: MathProblemGraph,
    max_layers: int = 8,
    hidden_dim: int = 64,
    seed: int = 42,
) -> list[dict]:
    """
    num_layers を 1..max_layers と変えて Over-smoothing を測定する。

    各 num_layers で記録:
      - smoothing_score: 全ノードペアのコサイン類似度の平均 (→1.0 で深刻)
      - answer_variance: 答えノードの各次元の分散の平均 (→0 で表現死)
    """
    results = []

    for n_layers in range(1, max_layers + 1):
        torch.manual_seed(seed)
        model = MathSolverGNN(
            node_dim=NODE_DIM, hidden_dim=hidden_dim, num_layers=n_layers,
        )
        model.eval()

        with torch.no_grad():
            model.gnn(problem.x, problem.adj, record_history=True)

        final_h = model.gnn.signal_history[-1]  # (N, hidden_dim)
        N = final_h.size(0)

        # smoothing_score: 全ペアのコサイン類似度平均
        norms = final_h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normed = final_h / norms
        sim_matrix = normed @ normed.T  # (N, N)
        # 対角を除いた上三角の平均
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        smoothing_score = sim_matrix[mask].mean().item()

        # answer_variance: 答えノード表現の次元ごとの分散の平均
        ans_feat = final_h[problem.answer_node]
        answer_variance = ans_feat.var().item()

        results.append({
            'layers': n_layers,
            'smoothing': smoothing_score,
            'ans_var': answer_variance,
        })

    return results


# ─────────────────────────────────────────────
#  測定 2: Over-squashing
# ─────────────────────────────────────────────

def measure_over_squashing(
    problem: MathProblemGraph,
    num_layers: int = 4,
    hidden_dim: int = 64,
    seed: int = 42,
) -> list[dict]:
    """
    答えノード出力の各入力ノードに対する感度を Jacobian ノルムで測定する。

    jacobian_norm[i] = || d(h_answer) / d(x[i]) ||_F
    """
    torch.manual_seed(seed)
    model = MathSolverGNN(
        node_dim=NODE_DIM, hidden_dim=hidden_dim, num_layers=num_layers,
    )
    model.eval()

    x = problem.x.clone().requires_grad_(True)
    ans_idx = problem.answer_node
    N = x.size(0)

    # forward して答えノードの出力を取得
    out = model.gnn(x, problem.adj)  # (N, node_dim)
    h_answer = out[ans_idx]  # (node_dim,)

    # 各入力ノードに対する Jacobian ノルムを計算
    node_dim = h_answer.size(0)
    jacobian_norms = []

    for i in range(N):
        # d(h_answer) / d(x[i]) を要素ごとに計算
        grads = []
        for d in range(node_dim):
            if x.grad is not None:
                x.grad.zero_()
            h_answer[d].backward(retain_graph=True)
            grads.append(x.grad[i].clone())

        # (node_dim_out, node_dim_in) の Jacobian 行列
        jac = torch.stack(grads)  # (node_dim, node_dim)
        jacobian_norms.append(jac.norm().item())

    # 自己参照（答えノード）で正規化
    self_norm = jacobian_norms[ans_idx] if jacobian_norms[ans_idx] > 1e-8 else 1.0
    normalized = [n / self_norm for n in jacobian_norms]

    # ホップ距離を計算
    G = nx.from_numpy_array(problem.adj.numpy())
    results = []
    for i in range(N):
        try:
            hop = nx.shortest_path_length(G, i, ans_idx)
        except nx.NetworkXNoPath:
            hop = -1

        label = problem.node_labels[i] if i < len(problem.node_labels) else f"node{i}"
        results.append({
            'node': i,
            'label': label,
            'hop': hop,
            'jacobian_norm': jacobian_norms[i],
            'normalized': normalized[i],
            'degree': G.degree(i),
        })

    return results


# ─────────────────────────────────────────────
#  測定 3: タスク別サマリー
# ─────────────────────────────────────────────

def run_bottleneck_analysis():
    """全測定を実行してタスク別サマリーを出力する。"""

    problems = {
        'linear': build_linear_equation(2, 3, 7),
        'quadratic': build_quadratic_equation(1, -5, 6),
        'simultaneous': build_simultaneous_equations(1, 1, 5, 1, -1, 1),
    }

    # ── 測定 1: Over-smoothing ──────────────────────────────
    print("=" * 65)
    print("  Measurement 1: Over-smoothing")
    print("=" * 65)

    os_start_layers = {}

    for task_name, prob in problems.items():
        print(f"\n  [{task_name}] {prob.problem_str}")
        print(f"  {'layers':>8}  {'smoothing':>10}  {'ans_var':>10}")
        print("  " + "-" * 32)

        results = measure_over_smoothing(prob)
        os_start = None

        for r in results:
            marker = ""
            if os_start is None and r['smoothing'] > 0.9:
                os_start = r['layers']
                marker = "  <- Over-smoothing"
            print(
                f"  {r['layers']:>8}  "
                f"{r['smoothing']:>10.4f}  "
                f"{r['ans_var']:>10.6f}{marker}"
            )

        os_start_layers[task_name] = os_start if os_start else ">8"

    # ── 測定 2: Over-squashing ──────────────────────────────
    print()
    print("=" * 65)
    print("  Measurement 2: Over-squashing (Jacobian sensitivity)")
    print("=" * 65)

    squash_data = {}

    for task_name, prob in problems.items():
        print(f"\n  [{task_name}] {prob.problem_str}")
        print(f"  answer_node = {prob.answer_node}")
        print(
            f"  {'Node':>5} {'Label':>12}  {'hop':>4}  "
            f"{'|J|_F':>10}  {'normalized':>10}  {'deg':>4}"
        )
        print("  " + "-" * 52)

        results = measure_over_squashing(prob)

        # ブリッジノード（最大次数）と最遠葉ノードを特定
        max_deg_node = max(results, key=lambda r: r['degree'])
        farthest = max(
            (r for r in results if r['node'] != prob.answer_node),
            key=lambda r: r['hop'],
        )

        for r in results:
            marker = ""
            if r['node'] == max_deg_node['node']:
                marker = "  bridge"
            if r['node'] == farthest['node'] and r['node'] != max_deg_node['node']:
                marker = "  farthest"
            if r['node'] == prob.answer_node:
                marker = "  (self)"

            print(
                f"  {r['node']:>5} {r['label']:>12}  "
                f"{r['hop']:>4}  "
                f"{r['jacobian_norm']:>10.6f}  "
                f"{r['normalized']:>10.4f}"
                f"{r['degree']:>6}"
                f"{marker}"
            )

        squash_data[task_name] = {
            'bridge_norm': max_deg_node['normalized'],
            'leaf_norm': farthest['normalized'],
            'bridge_node': max_deg_node['node'],
            'leaf_node': farthest['node'],
            'squash_detected': farthest['normalized'] < 0.01,
        }

    # ── 測定 3: タスク別サマリー ─────────────────────────────
    print()
    print("=" * 65)
    print("  Measurement 3: Summary")
    print("=" * 65)
    print()
    print(
        f"  {'Task':<16} {'OS start':>8}  "
        f"{'bridge_norm':>12}  {'leaf_norm':>10}  {'Verdict'}"
    )
    print("  " + "-" * 62)

    for task_name in problems:
        os_layer = os_start_layers[task_name]
        sd = squash_data[task_name]

        verdicts = []
        if isinstance(os_layer, int) and os_layer <= 4:
            verdicts.append(f"OS@{os_layer}")
        if sd['squash_detected']:
            verdicts.append("Squash")
        if not verdicts:
            verdicts.append("OK")
        verdict = ", ".join(verdicts)

        os_str = str(os_layer) if isinstance(os_layer, int) else os_layer
        print(
            f"  {task_name:<16} {os_str:>8}  "
            f"{sd['bridge_norm']:>12.4f}  "
            f"{sd['leaf_norm']:>10.4f}  "
            f"{verdict}"
        )

    print()
    print("  Legend:")
    print("    OS start    : layer where smoothing_score > 0.9")
    print("    bridge_norm : Jacobian norm of highest-degree node (normalized)")
    print("    leaf_norm   : Jacobian norm of farthest leaf node (normalized)")
    print("    Squash      : leaf_norm < 0.01 (information lost)")
    print("    OS@N        : Over-smoothing begins at layer N (<= 4)")


if __name__ == '__main__':
    run_bottleneck_analysis()
