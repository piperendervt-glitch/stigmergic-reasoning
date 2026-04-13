"""
訓練済みモデルの評価スクリプト。

評価内容:
  1. テスト問題（訓練に使っていない問題）での3軸スコア
  2. 訓練前後の比較表（Anchor効果の改善度）
  3. 「2x + 3 = 7」を例に、Message Passing 各層での
     情報集約の変化を可視化（matplotlib）
  4. GNN vs LLM-like Anchor の Anchor Score 比較
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

from solvers.math_problem_gnn import (
    MathSolverGNN,
    MathProblemGraph,
    build_linear_equation,
    build_quadratic_equation,
    build_simultaneous_equations,
    generate_training_data,
    evaluate_information_flow,
    NODE_DIM,
    SCALE,
)
from experiments.comparison_experiment import (
    compute_anchor_effect_score,
    compute_signal_monotonicity,
    LLMLikeAnchor,
    BaselineMessagePassing,
)


def load_models(model_path: str = 'trained_model.pt'):
    """訓練済みモデルと未訓練モデルの両方を返す。"""
    untrained = MathSolverGNN(node_dim=NODE_DIM, hidden_dim=64, num_layers=4)
    torch.manual_seed(42)
    untrained_fresh = MathSolverGNN(node_dim=NODE_DIM, hidden_dim=64, num_layers=4)

    trained = MathSolverGNN(node_dim=NODE_DIM, hidden_dim=64, num_layers=4)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, weights_only=False)
        trained.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded trained model from {model_path}")
        n_epochs = checkpoint.get('n_epochs', '?')
        n_samples = checkpoint.get('n_samples', '?')
        print(f"  Training: {n_epochs} epochs, {n_samples} samples")
    else:
        print(f"  WARNING: No trained model at {model_path}. Using untrained.")
        trained = untrained_fresh

    return trained, untrained_fresh


# ─────────────────────────────────────────────
#  1. テスト問題での3軸スコア
# ─────────────────────────────────────────────

def eval_test_problems(solver: MathSolverGNN, n_samples: int = 100):
    """テスト問題で3軸評価を行う。"""
    print()
    print("=" * 65)
    print("  1. Test Problem Evaluation (3-axis)")
    print("=" * 65)
    print()

    problems = generate_training_data(n_samples=n_samples, seed=999)
    solver.eval()

    results_by_type = {'linear': [], 'quadratic': [], 'simultaneous': []}

    for prob in problems:
        ev = evaluate_information_flow(solver, prob)
        if 'x^2' in prob.problem_str or 'x²' in prob.problem_str:
            ptype = 'quadratic'
        elif '/' in prob.problem_str:
            ptype = 'simultaneous'
        else:
            ptype = 'linear'
        results_by_type[ptype].append(ev)

    print(f"  {'Type':<16} {'N':>4}  {'InputSim':>9}  {'DepOK':>7}  {'RouteOK':>8}")
    print("  " + "-" * 50)

    all_evals = []
    for ptype, evs in results_by_type.items():
        if not evs:
            continue
        n = len(evs)
        avg_input = np.mean([e['input_sim'] for e in evs])
        avg_dep = np.mean([e['dependency_ok'] for e in evs])
        avg_route = np.mean([e['route_ok'] for e in evs])
        print(f"  {ptype:<16} {n:>4}  {avg_input:>9.4f}  {avg_dep:>7.0%}  {avg_route:>8.0%}")
        all_evals.extend(evs)

    print("  " + "-" * 50)
    total = len(all_evals)
    avg_input = np.mean([e['input_sim'] for e in all_evals])
    avg_dep = np.mean([e['dependency_ok'] for e in all_evals])
    avg_route = np.mean([e['route_ok'] for e in all_evals])
    print(f"  {'TOTAL':<16} {total:>4}  {avg_input:>9.4f}  {avg_dep:>7.0%}  {avg_route:>8.0%}")

    return all_evals


# ─────────────────────────────────────────────
#  2. 訓練前後の比較
# ─────────────────────────────────────────────

def eval_before_after(trained: MathSolverGNN, untrained: MathSolverGNN):
    """訓練前後の3軸スコアを比較する。"""
    print()
    print("=" * 65)
    print("  2. Before/After Training Comparison")
    print("=" * 65)
    print()

    test_problems = [
        build_linear_equation(2, 3, 7),
        build_linear_equation(3, -6, 9),
        build_linear_equation(5, 2, 12),
        build_quadratic_equation(1, -5, 6),
        build_quadratic_equation(1, -3, 2),
        build_simultaneous_equations(1, 1, 5, 1, -1, 1),
        build_simultaneous_equations(2, 3, 12, 1, -1, 1),
    ]

    print(f"  {'Problem':<30} {'':>4} {'InputSim':>9}  {'DepOK':>7}  {'RouteOK':>8}  {'Anchor':>8}")
    print("  " + "-" * 72)

    for prob in test_problems:
        trained.eval()
        untrained.eval()
        ev_t = evaluate_information_flow(trained, prob)
        ev_u = evaluate_information_flow(untrained, prob)

        G = nx.from_numpy_array(prob.adj.numpy())
        with torch.no_grad():
            out_t = trained.gnn(prob.x, prob.adj)
            out_u = untrained.gnn(prob.x, prob.adj)
        anchor_t = compute_anchor_effect_score(out_t, G)
        anchor_u = compute_anchor_effect_score(out_u, G)

        name = prob.name[:28]
        print(f"  {name:<30} {'pre':>4} {ev_u['input_sim']:>9.4f}  "
              f"{'Y' if ev_u['dependency_ok'] else 'N':>7}  "
              f"{'Y' if ev_u['route_ok'] else 'N':>8}  {anchor_u:>8.4f}")
        print(f"  {'':30} {'post':>4} {ev_t['input_sim']:>9.4f}  "
              f"{'Y' if ev_t['dependency_ok'] else 'N':>7}  "
              f"{'Y' if ev_t['route_ok'] else 'N':>8}  {anchor_t:>8.4f}")
        print()


# ─────────────────────────────────────────────
#  3. Message Passing 各層の可視化
# ─────────────────────────────────────────────

def visualize_message_passing_layers(
    solver: MathSolverGNN,
    save_path: str = 'message_passing_layers.png',
):
    """「2x + 3 = 7」を例に、各GNN層での情報集約の変化を可視化する。"""
    print()
    print("=" * 65)
    print("  3. Message Passing Layer Visualization")
    print("=" * 65)
    print()

    prob = build_linear_equation(2, 3, 7)
    solver.eval()
    with torch.no_grad():
        solver.gnn(prob.x, prob.adj, record_history=True)

    history = solver.gnn.signal_history
    n_layers = len(history)
    ans_idx = prob.answer_node

    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))
    if n_layers == 1:
        axes = [axes]

    fig.suptitle("Message Passing: 2x + 3 = 7 (per-layer answer-node similarity)",
                 fontsize=12, fontweight='bold')

    for layer_idx, (h, ax) in enumerate(zip(history, axes)):
        ans_vec = h[ans_idx]
        sims = []
        for i in range(h.size(0)):
            if i == ans_idx:
                sims.append(1.0)
                continue
            s = F.cosine_similarity(
                h[i].unsqueeze(0), ans_vec.unsqueeze(0)
            ).item()
            sims.append(s)

        colors = ['#FF6B6B' if i == ans_idx else
                  plt.cm.YlOrRd(max(0, s)) for i, s in enumerate(sims)]

        G = nx.from_numpy_array(prob.adj.numpy())
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx(
            G, pos=pos, ax=ax,
            node_color=sims,
            node_size=500,
            cmap=plt.cm.YlOrRd,
            vmin=-0.2, vmax=1.0,
            with_labels=True,
            font_size=8,
            edge_color='gray',
        )
        ax.set_title(f"Layer {layer_idx}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

    # テキスト出力
    print()
    print(f"  {'Layer':>6}  ", end='')
    for i, label in enumerate(prob.node_labels):
        short = label[:8]
        print(f"{short:>9}", end='')
    print()
    print("  " + "-" * (8 + 9 * len(prob.node_labels)))

    for layer_idx, h in enumerate(history):
        ans_vec = h[ans_idx]
        print(f"  {layer_idx:>6}  ", end='')
        for i in range(h.size(0)):
            if i == ans_idx:
                s = 1.0
            else:
                s = F.cosine_similarity(
                    h[i].unsqueeze(0), ans_vec.unsqueeze(0)
                ).item()
            print(f"{s:>9.4f}", end='')
        print()


# ─────────────────────────────────────────────
#  4. GNN vs LLM-like Anchor Score 比較
# ─────────────────────────────────────────────

def eval_gnn_vs_llm_anchor(trained: MathSolverGNN):
    """GNN と LLM-like Anchor の Anchor Score を比較する。"""
    print()
    print("=" * 65)
    print("  4. GNN vs LLM-like Anchor Score")
    print("=" * 65)
    print()

    torch.manual_seed(42)
    llm = LLMLikeAnchor(node_dim=NODE_DIM, num_layers=4)
    baseline = BaselineMessagePassing(node_dim=NODE_DIM, num_layers=4)

    test_problems = [
        build_linear_equation(2, 3, 7),
        build_linear_equation(4, -2, 10),
        build_quadratic_equation(1, -5, 6),
        build_simultaneous_equations(1, 1, 5, 1, -1, 1),
    ]

    print(f"  {'Problem':<30} {'GNN':>8} {'LLM-like':>10} {'Baseline':>10} {'GNN<LLM?':>10}")
    print("  " + "-" * 72)

    for prob in test_problems:
        trained.eval()
        llm.eval()
        baseline.eval()

        with torch.no_grad():
            out_gnn = trained.gnn(prob.x, prob.adj)
            out_llm = llm(prob.x, prob.adj)
            out_base = baseline(prob.x, prob.adj)

        G = nx.from_numpy_array(prob.adj.numpy())
        a_gnn = compute_anchor_effect_score(out_gnn, G)
        a_llm = compute_anchor_effect_score(out_llm, G)
        a_base = compute_anchor_effect_score(out_base, G)

        better = "Y" if a_gnn < a_llm else "N"
        name = prob.name[:28]
        print(f"  {name:<30} {a_gnn:>8.4f} {a_llm:>10.4f} {a_base:>10.4f} {better:>10}")

    print()
    print("  GNN < LLM-like = GNN has less Anchor effect (better)")


# ─────────────────────────────────────────────
#  メイン
# ─────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  GNN Stigmergy - Trained Model Evaluation")
    print("=" * 65)

    trained, untrained = load_models()

    # 1. テスト問題での3軸スコア
    eval_test_problems(trained)

    # 2. 訓練前後の比較
    eval_before_after(trained, untrained)

    # 3. Message Passing 各層の可視化
    visualize_message_passing_layers(trained)

    # 4. GNN vs LLM-like Anchor Score
    eval_gnn_vs_llm_anchor(trained)

    print()
    print("=" * 65)
    print("  Evaluation complete.")
    print("=" * 65)


if __name__ == '__main__':
    main()
