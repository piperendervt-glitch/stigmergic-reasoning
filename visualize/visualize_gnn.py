"""
GNN 動作の可視化ツール。

機能:
  A) 波面伝播アニメーション
     - 問題グラフ上でゲート値が layer ごとに広がる様子
     - matplotlib animation で保存 (wave_propagation.gif)

  B) Attention Weight ヒートマップ
     - 各エッジの Attention 重みを隣接行列形式で表示
     - 経路エッジが高い重みを持つかを視覚的に確認

  C) 問題タイプ別スコア比較バーチャート
     - 一次/二次/連立でどの評価軸が強い/弱いか
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import os
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from solvers.math_problem_gnn import (
    MathSolverGNN,
    MathProblemGraph,
    build_linear_equation,
    build_quadratic_equation,
    build_simultaneous_equations,
    generate_training_data,
    evaluate_information_flow,
    NODE_DIM,
)
from core.gnn_stigmergy import StigmergyMessagePassing


# ─────────────────────────────────────────────
#  A) 波面伝播アニメーション
# ─────────────────────────────────────────────

def create_wave_propagation_animation(
    solver: MathSolverGNN,
    problem: MathProblemGraph,
    save_path: str = 'wave_propagation.gif',
    fps: int = 2,
):
    """
    問題グラフ上でゲート値が layer ごとに広がる様子をアニメーション化する。
    """
    print(f"  Creating wave propagation animation...")

    solver.eval()
    with torch.no_grad():
        solver.gnn(problem.x, problem.adj, record_history=True)

    history = solver.gnn.signal_history
    n_layers = len(history)
    n_nodes = problem.x.size(0)

    G = nx.from_numpy_array(problem.adj.numpy())
    pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"Wave Propagation: {problem.problem_str}", fontsize=12, fontweight='bold')

    def update(frame):
        ax.clear()
        h = history[frame]
        norms = h.norm(dim=1).numpy()
        max_norm = norms.max() + 1e-8

        node_colors = norms / max_norm
        node_sizes = 300 + 400 * node_colors

        nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color='gray', alpha=0.5)
        nodes = nx.draw_networkx_nodes(
            G, pos=pos, ax=ax,
            node_color=node_colors.tolist(),
            node_size=node_sizes.tolist(),
            cmap=plt.cm.YlOrRd,
            vmin=0, vmax=1.0,
        )
        nx.draw_networkx_labels(G, pos=pos, ax=ax, font_size=8)

        active = (norms > norms.max() * 0.1).sum()
        ax.set_title(f"Layer {frame} / {n_layers - 1}  |  Active nodes: {active}/{n_nodes}")
        ax.axis('off')
        return nodes,

    anim = FuncAnimation(fig, update, frames=n_layers, interval=1000 // fps, blit=False)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────
#  B) Attention Weight ヒートマップ
# ─────────────────────────────────────────────

def create_attention_heatmap(
    solver: MathSolverGNN,
    problem: MathProblemGraph,
    save_path: str = 'attention_heatmap.png',
):
    """
    各エッジの Attention 重みを隣接行列形式で表示する。
    最終GNN層の Attention を抽出する。
    """
    print(f"  Creating attention heatmap...")

    solver.eval()
    n_nodes = problem.x.size(0)

    # 最終GNN層を取得して Attention 重みを抽出
    last_layer: StigmergyMessagePassing = solver.gnn.layers[-1]

    # forward を手動で実行して Attention weight を計算
    with torch.no_grad():
        # input_proj + 前の層を通す
        h = F.relu(solver.gnn.input_proj(problem.x))
        for layer in solver.gnn.layers[:-1]:
            h = layer(h, problem.adj)

        # 最終層の Attention を計算
        gate = last_layer._compute_gate(h)
        x_gated = F.normalize(h, p=2, dim=-1) * gate

        src, dst = last_layer._get_edges(problem.adj)
        E = src.size(0)

        if E == 0:
            print("  No edges found, skipping heatmap.")
            return

        edge_feat = torch.cat([x_gated[dst], x_gated[src]], dim=-1)
        edge_score = last_layer.att(edge_feat).squeeze(-1)

        src_gate = gate[src].squeeze(-1)
        edge_score = edge_score.masked_fill(src_gate < 0.5, float('-inf'))

        # Sparse softmax
        from gnn_stigmergy import _sparse_softmax
        att_weight = _sparse_softmax(edge_score, dst, n_nodes)

    # ヒートマップ行列を構築 (dst, src)
    att_matrix = np.zeros((n_nodes, n_nodes))
    for e_idx in range(E):
        s = src[e_idx].item()
        d = dst[e_idx].item()
        att_matrix[d][s] = att_weight[e_idx].item()

    # 経路エッジを特定
    G = nx.from_numpy_array(problem.adj.numpy())
    try:
        path = nx.shortest_path(G, 0, problem.answer_node)
        path_edges = set()
        for i in range(len(path) - 1):
            path_edges.add((path[i], path[i + 1]))
            path_edges.add((path[i + 1], path[i]))
    except nx.NetworkXNoPath:
        path_edges = set()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Attention Weights: {problem.problem_str}",
                 fontsize=12, fontweight='bold')

    # Heatmap
    im = ax1.imshow(att_matrix, cmap='YlOrRd', vmin=0, vmax=max(att_matrix.max(), 0.01))
    ax1.set_xlabel("Source node")
    ax1.set_ylabel("Destination node")
    ax1.set_title("Attention Weight Matrix (last layer)")

    labels = [l[:8] for l in problem.node_labels]
    ax1.set_xticks(range(n_nodes))
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax1.set_yticks(range(n_nodes))
    ax1.set_yticklabels(labels, fontsize=7)
    plt.colorbar(im, ax=ax1, fraction=0.046)

    # 経路エッジの Attention vs 非経路エッジ
    path_weights = []
    nonpath_weights = []
    for e_idx in range(E):
        s = src[e_idx].item()
        d = dst[e_idx].item()
        w = att_weight[e_idx].item()
        if (s, d) in path_edges:
            path_weights.append(w)
        else:
            nonpath_weights.append(w)

    categories = ['Path edges', 'Non-path edges']
    means = [
        np.mean(path_weights) if path_weights else 0,
        np.mean(nonpath_weights) if nonpath_weights else 0,
    ]
    colors = ['#4CAF50', '#F44336']
    ax2.bar(categories, means, color=colors, alpha=0.8)
    ax2.set_ylabel("Mean Attention Weight")
    ax2.set_title("Path vs Non-path Edge Attention")
    ax2.set_ylim(0, max(max(means) * 1.3, 0.01))

    for i, v in enumerate(means):
        ax2.text(i, v + 0.005, f"{v:.4f}", ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────
#  C) 問題タイプ別スコア比較バーチャート
# ─────────────────────────────────────────────

def create_score_comparison_chart(
    solver: MathSolverGNN,
    save_path: str = 'score_comparison.png',
    n_samples: int = 60,
):
    """問題タイプ別にどの評価軸が強い/弱いかをバーチャートで表示する。"""
    print(f"  Creating score comparison chart...")

    problems = generate_training_data(n_samples=n_samples, seed=123)
    solver.eval()

    results = {'linear': [], 'quadratic': [], 'simultaneous': []}

    for prob in problems:
        ev = evaluate_information_flow(solver, prob)
        if 'x^2' in prob.problem_str or 'x²' in prob.problem_str:
            ptype = 'quadratic'
        elif '/' in prob.problem_str:
            ptype = 'simultaneous'
        else:
            ptype = 'linear'
        results[ptype].append(ev)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Score Comparison by Problem Type", fontsize=14, fontweight='bold')

    metrics = ['Input Similarity', 'Dependency OK', 'Route OK']
    metric_keys = [
        lambda e: e['input_sim'],
        lambda e: float(e['dependency_ok']),
        lambda e: float(e['route_ok']),
    ]
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    for ax, metric_name, key_fn, color in zip(axes, metrics, metric_keys, colors):
        types = []
        means = []
        stds = []

        for ptype, evs in results.items():
            if not evs:
                continue
            vals = [key_fn(e) for e in evs]
            types.append(ptype)
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        x = np.arange(len(types))
        bars = ax.bar(x, means, yerr=stds, color=color, alpha=0.8, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(types, fontsize=9)
        ax.set_ylabel("Score")
        ax.set_title(metric_name)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)

        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{m:.3f}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────
#  メイン
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  GNN Stigmergy - Visualization Dashboard")
    print("=" * 60)
    print()

    solver = MathSolverGNN(node_dim=NODE_DIM, hidden_dim=64, num_layers=4)
    model_path = 'trained_model.pt'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, weights_only=False)
        solver.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded trained model from {model_path}")
    else:
        print(f"  No trained model found. Using untrained model.")

    solver.eval()

    prob = build_linear_equation(2, 3, 7)

    # A) 波面伝播アニメーション
    print()
    print("  [A] Wave Propagation Animation")
    create_wave_propagation_animation(solver, prob)

    # B) Attention Weight ヒートマップ
    print()
    print("  [B] Attention Weight Heatmap")
    create_attention_heatmap(solver, prob)

    # C) 問題タイプ別スコア比較
    print()
    print("  [C] Score Comparison by Problem Type")
    create_score_comparison_chart(solver)

    print()
    print("=" * 60)
    print("  All visualizations generated:")
    print("    - wave_propagation.gif")
    print("    - attention_heatmap.png")
    print("    - score_comparison.png")
    print("=" * 60)


if __name__ == '__main__':
    main()
