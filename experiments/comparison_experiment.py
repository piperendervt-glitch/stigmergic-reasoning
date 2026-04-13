"""
comparison_experiment.py
========================
GNN Stigmergy vs 従来Baseline の比較実験

比較対象:
  - Baseline: 単純な平均集約（GNN Attentionなし）
  - GNN Stigmergy: Attention付きMessage Passing
  - LLM-like: Node 0の影響が全ノードに伝播し続けるモデル（Anchor効果の模擬）

評価指標:
  1. Anchor Effect Score      : Node 0の信号が遠隔ノードに残存する度合い
  2. Signal Monotonicity      : 信号強度がホップ距離に対して単調減少するか
  3. Diffusion Uniformity     : 信号が空間的に均等に広がるか
  4. Recovery Speed           : ノイズからの回復速度

検証仮説:
  H1: GNN Stigmergyは従来より低いAnchor Effect Scoreを示す
  H2: GNN StigmergyはLLM-likeより高いSignal Monotonicity を示す
  H3: Attention機構によって不要な信号が自然に減衰する
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional

from core.gnn_stigmergy import GNNStigmergyLayer, build_test_graph
from core.root_layer_gnn import RootLayerGNN


# ─────────────────────────────────────────────
#  Baseline モデル
# ─────────────────────────────────────────────

class BaselineMessagePassing(nn.Module):
    """
    Attention重みなしの単純平均集約Baseline。
    全ての近傍ノードを等しく扱う。
    """
    def __init__(self, node_dim: int = 16, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([
            nn.Linear(node_dim, node_dim)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(node_dim) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x.clone()
        degree = adj.sum(dim=1, keepdim=True).clamp(min=1)
        norm_adj = adj / degree

        for layer, norm in zip(self.layers, self.norms):
            h_neigh = torch.matmul(norm_adj, h)
            h = F.relu(norm(layer(h + h_neigh)))
        return h


class LLMLikeAnchor(nn.Module):
    """
    LLMのAnchor効果を模擬するモデル。
    Node 0の信号が全ての処理に残存し続ける。
    （LLMにおける「文脈の先頭部分が後続に強く影響する」現象）
    """
    def __init__(self, node_dim: int = 16, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        # Node 0の「アンカー」を全ノードに注入するための重み
        self.anchor_weight = nn.Parameter(torch.ones(1) * 0.5)
        self.layers = nn.ModuleList([
            nn.Linear(node_dim * 2, node_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x.clone()
        anchor = x[0:1].expand(x.size(0), -1)  # Node 0を全ノードに broadcast

        degree = adj.sum(dim=1, keepdim=True).clamp(min=1)
        norm_adj = adj / degree

        for layer in self.layers:
            h_neigh = torch.matmul(norm_adj, h)
            # アンカーを強制的に混入（LLMの先頭文脈残存の模擬）
            h_with_anchor = torch.cat([h_neigh, anchor * self.anchor_weight], dim=-1)
            h = F.relu(layer(h_with_anchor))
        return h


# ─────────────────────────────────────────────
#  評価指標
# ─────────────────────────────────────────────

@dataclass
class ExperimentResult:
    model_name: str
    anchor_effect_score: float      # 低いほど良い（Anchor効果が少ない）
    signal_monotonicity: float      # 高いほど良い（距離に対して単調減少）
    diffusion_uniformity: float     # 高いほど良い（均等な拡散）
    recovery_speed: float           # 高いほど良い（ノイズからの回復）
    notes: str = ""


def compute_anchor_effect_score(
    out: torch.Tensor,
    G: nx.Graph,
    source_node: int = 0,
    x_source: torch.Tensor = None,
) -> float:
    """
    Anchor Effect Score を計算する（v2: コサイン類似度ベース）。

    【旧実装の問題】
      「信号ノルム × ホップ距離」の相関を取っていた。
      しかしSparse Attentionでは「ゼロノードのL2正規化フラット化」により
      遠いノードのノルムが均一に増加し、正の相関が生じた。
      つまり旧指標は「ノルムの大きさ」を測っており、
      「Node 0の情報がどれだけ残っているか」を測っていなかった。

    【新実装の設計】
      直接問いに答える：出力ベクトルが「Node 0の出力」とどれだけ似ているか。

      AnchorScore(i) = cosine_similarity(out[i], out[source])

      これを全ノードで集計し、ホップ距離で重み付けして平均する。
      遠いノードほど重みを大きくすることで、
      「近傍は似ていて当然、遠くにどれだけ残存するか」を正しく測る。

      Score = Σ hop(i) × cos(out[i], out[source])
              ─────────────────────────────────────
                      Σ hop(i)

      高いほど遠いノードにもNode 0の情報が残存 → Anchor効果が強い（悪い）。

    Parameters
    ----------
    out         : (N, dim) モデル出力
    G           : networkxグラフ
    source_node : Anchorノードのインデックス（デフォルト0）
    x_source    : 未使用（互換性のため残存）

    Returns
    -------
    score : float [0, 1] 高いほどAnchor効果が強い
    """
    out_detached = out.detach()
    source_vec = out_detached[source_node]          # (dim,)
    source_norm = source_vec.norm().item()

    if source_norm < 1e-8:
        return 0.0

    n_nodes = len(G.nodes())
    weighted_sim_sum = 0.0
    weight_sum = 0.0

    for target in range(n_nodes):
        if target == source_node:
            continue
        try:
            hop = nx.shortest_path_length(G, source_node, target)
        except nx.NetworkXNoPath:
            continue

        target_vec  = out_detached[target]
        target_norm = target_vec.norm().item()

        if target_norm < 1e-8:
            cos_sim = 0.0
        else:
            cos_sim = float(
                torch.dot(source_vec, target_vec).item()
                / (source_norm * target_norm)
            )

        # hop が大きいほど重みを増やす（遠くへの残存を重視）
        weight = float(hop)
        weighted_sim_sum += weight * cos_sim
        weight_sum       += weight

    if weight_sum < 1e-8:
        return 0.0

    raw_score = weighted_sim_sum / weight_sum   # [-1, 1]
    # [-1, 1] → [0, 1] に線形変換
    return float((raw_score + 1.0) / 2.0)


def compute_signal_monotonicity(
    out: torch.Tensor,
    G: nx.Graph,
    source_node: int = 0
) -> float:
    """
    Signal Monotonicity を計算する（v2: コサイン類似度ベース）。

    ホップ距離が増えるにつれて「Node 0との類似度」が
    単調減少しているかを評価する。

    旧版はノルムの単調性を測っていたが、
    Sparse Attentionのフラット化によって誤判定が生じていた。
    新版は「情報の残存量（コサイン類似度）」の単調性を直接測る。

    Returns
    -------
    score : float [0, 1] 高いほど単調減少（良い）
    """
    out_detached = out.detach()
    source_vec  = out_detached[source_node]
    source_norm = source_vec.norm().item()

    hop_to_sims: dict[int, list[float]] = {}

    for target in range(len(G.nodes())):
        if target == source_node:
            continue
        try:
            hop = nx.shortest_path_length(G, source_node, target)
        except nx.NetworkXNoPath:
            continue

        t_vec  = out_detached[target]
        t_norm = t_vec.norm().item()
        if source_norm < 1e-8 or t_norm < 1e-8:
            cos_sim = 0.0
        else:
            cos_sim = float(torch.dot(source_vec, t_vec).item()
                            / (source_norm * t_norm))

        hop_to_sims.setdefault(hop, []).append(cos_sim)

    if len(hop_to_sims) < 2:
        return 1.0

    sorted_hops = sorted(hop_to_sims.keys())
    avg_by_hop  = [float(np.mean(hop_to_sims[h])) for h in sorted_hops]

    monotone_count = sum(
        1 for i in range(len(avg_by_hop) - 1)
        if avg_by_hop[i] >= avg_by_hop[i + 1]
    )
    return float(monotone_count) / max(len(avg_by_hop) - 1, 1)


def compute_diffusion_uniformity(out: torch.Tensor) -> float:
    """
    Diffusion Uniformity を計算する。

    信号が空間的に均等に分布しているかを評価。
    変動係数の逆数（低い変動 → 高いスコア）。

    Returns
    -------
    score : float [0, 1] 高いほど均等（偏りが少ない）
    """
    strengths = out.norm(dim=1).detach().numpy()
    mean = strengths.mean()
    std = strengths.std()

    if mean < 1e-8:
        return 0.0

    cv = std / mean  # 変動係数
    # CV=0（完全均等）→ 1.0、CV大→ 0.0 に変換
    uniformity = 1.0 / (1.0 + cv)
    return float(uniformity)


def compute_recovery_speed(
    model: nn.Module,
    x_clean: torch.Tensor,
    adj: torch.Tensor,
    noise_level: float = 0.5
) -> float:
    """
    Recovery Speed を計算する。

    ノイズを加えた入力に対して、ノイズなしと近い出力を
    得られるかを評価する（ロバスト性の代理指標）。

    Returns
    -------
    score : float [0, 1] 高いほどロバスト
    """
    model.eval()
    with torch.no_grad():
        out_clean = model(x_clean, adj)
        x_noisy = x_clean + torch.randn_like(x_clean) * noise_level
        out_noisy = model(x_noisy, adj)

    diff = (out_clean - out_noisy).norm().item()
    clean_norm = out_clean.norm().item() + 1e-8
    relative_diff = diff / clean_norm

    # 差が小さいほど高スコア
    score = 1.0 / (1.0 + relative_diff)
    return float(score)


# ─────────────────────────────────────────────
#  比較実験
# ─────────────────────────────────────────────

def run_comparison_experiment(
    n_nodes: int = 10,
    node_dim: int = 16,
    num_layers: int = 3,
    n_trials: int = 5,
    seed: int = 42
) -> list[ExperimentResult]:
    """
    3モデルの比較実験を実行する。

    Parameters
    ----------
    n_nodes    : ノード数
    node_dim   : 特徴次元
    num_layers : GNN層数
    n_trials   : 試行回数（平均を取る）
    seed       : 乱数シード
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 70)
    print("GNN Stigmergy vs Baseline 比較実験")
    print("=" * 70)
    print(f"ノード数={n_nodes}, 特徴次元={node_dim}, 層数={num_layers}, 試行数={n_trials}")
    print()

    # グラフ構築
    adj, G = build_test_graph(n_nodes, 'path')

    # モデル初期化
    models = {
        'GNN Stigmergy': GNNStigmergyLayer(node_dim, 32, num_layers, dropout=0.0),
        'Baseline (Mean Agg)': BaselineMessagePassing(node_dim, num_layers),
        'LLM-like Anchor': LLMLikeAnchor(node_dim, num_layers),
    }

    results_all: dict[str, list[dict]] = {name: [] for name in models}

    for trial in range(n_trials):
        # Node 0のみ活性化した初期状態
        x = torch.zeros(n_nodes, node_dim)
        x[0] = torch.ones(node_dim) + torch.randn(node_dim) * 0.1

        for name, model in models.items():
            model.eval()
            with torch.no_grad():
                out = model(x, adj)

            anchor = compute_anchor_effect_score(out, G)
            monotone = compute_signal_monotonicity(out, G)
            uniform = compute_diffusion_uniformity(out)
            recovery = compute_recovery_speed(model, x, adj)

            results_all[name].append({
                'anchor': anchor,
                'monotone': monotone,
                'uniform': uniform,
                'recovery': recovery
            })

    # 平均を計算
    final_results = []
    for name, trial_results in results_all.items():
        avg = ExperimentResult(
            model_name=name,
            anchor_effect_score=np.mean([r['anchor'] for r in trial_results]),
            signal_monotonicity=np.mean([r['monotone'] for r in trial_results]),
            diffusion_uniformity=np.mean([r['uniform'] for r in trial_results]),
            recovery_speed=np.mean([r['recovery'] for r in trial_results]),
        )
        final_results.append(avg)

    return final_results


def print_results(results: list[ExperimentResult]):
    """比較結果を表形式で表示する。"""
    print("─" * 70)
    print("比較結果サマリー  （v2指標: Anchor/単調性はコサイン類似度ベース）")
    print("─" * 70)
    print("  Anchor↓  : 遠いノードへのNode 0情報残存量（低いほどAnchor効果なし）")
    print("  単調性↑  : 距離が増えるほど類似度が下がるか（高いほど局所的影響）")
    print("  均一性↑  : 全ノードのノルムが均等か（高いほど空間的に偏りなし）")
    print("  回復性↑  : ノイズ入力への耐性（高いほどロバスト）")
    print("─" * 70)
    print(f"{'モデル':<25} {'Anchor↓':>10} {'単調性↑':>10} {'均一性↑':>10} {'回復性↑':>10}")
    print("─" * 70)

    for r in results:
        print(
            f"{r.model_name:<25} "
            f"{r.anchor_effect_score:>10.4f} "
            f"{r.signal_monotonicity:>10.4f} "
            f"{r.diffusion_uniformity:>10.4f} "
            f"{r.recovery_speed:>10.4f}"
        )

    print("─" * 70)
    print()

    # 仮説検証
    print("【仮説検証】")
    gnn  = next(r for r in results if 'GNN'      in r.model_name)
    llm  = next(r for r in results if 'LLM'      in r.model_name)
    base = next(r for r in results if 'Baseline' in r.model_name)

    h1 = gnn.anchor_effect_score < llm.anchor_effect_score
    h2 = gnn.signal_monotonicity > llm.signal_monotonicity
    h3 = gnn.recovery_speed      > base.recovery_speed

    print(f"  H1 (GNN Anchor効果 < LLM-like):  {'✓ 支持' if h1 else '✗ 棄却'}")
    print(f"     GNN={gnn.anchor_effect_score:.4f}  LLM={llm.anchor_effect_score:.4f}"
          f"  差={llm.anchor_effect_score - gnn.anchor_effect_score:+.4f}")
    print(f"     解釈: GNNでは遠いノードにNode 0の方向情報が"
          f"{'残りにくい' if h1 else '残りやすい（要改善）'}")
    print()
    print(f"  H2 (GNN 単調性 > LLM-like):      {'✓ 支持' if h2 else '✗ 棄却'}")
    print(f"     GNN={gnn.signal_monotonicity:.4f}  LLM={llm.signal_monotonicity:.4f}"
          f"  差={gnn.signal_monotonicity - llm.signal_monotonicity:+.4f}")
    print(f"     解釈: GNNではNode 0の類似度が距離に対して"
          f"{'単調に減衰している' if h2 else '単調に減衰しない（要改善）'}")
    print()
    print(f"  H3 (GNN 回復性 > Baseline):       {'✓ 支持' if h3 else '✗ 棄却'}")
    print(f"     GNN={gnn.recovery_speed:.4f}  Baseline={base.recovery_speed:.4f}"
          f"  差={gnn.recovery_speed - base.recovery_speed:+.4f}")
    print()


def visualize_results(results: list[ExperimentResult], save_path: str = None):
    """比較結果をレーダーチャートとバーチャートで可視化する。"""
    try:
        metrics = ['Anchor↓\n(低いほど良)', '単調性↑\n(高いほど良)',
                   '均一性↑\n(高いほど良)', '回復性↑\n(高いほど良)']

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("GNN Stigmergy vs Baseline 比較実験", fontsize=14, fontweight='bold')

        colors = ['#2196F3', '#4CAF50', '#F44336']

        # バーチャート
        ax = axes[0]
        x = np.arange(len(metrics))
        width = 0.25

        for i, (result, color) in enumerate(zip(results, colors)):
            values = [
                result.anchor_effect_score,
                result.signal_monotonicity,
                result.diffusion_uniformity,
                result.recovery_speed
            ]
            ax.bar(x + i * width, values, width, label=result.model_name, color=color, alpha=0.8)

        ax.set_xticks(x + width)
        ax.set_xticklabels(metrics, fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('スコア')
        ax.set_title('評価指標別スコア比較')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        # 改善率（GNN vs LLM-like）
        ax2 = axes[1]
        gnn = results[0]
        llm = results[2]

        improvements = [
            (llm.anchor_effect_score - gnn.anchor_effect_score),   # 低いほど良いので差が正 = 改善
            (gnn.signal_monotonicity - llm.signal_monotonicity),
            (gnn.diffusion_uniformity - llm.diffusion_uniformity),
            (gnn.recovery_speed - llm.recovery_speed),
        ]
        metric_labels = ['Anchor低減', '単調性向上', '均一性向上', '回復性向上']

        bar_colors = ['#4CAF50' if v >= 0 else '#F44336' for v in improvements]
        ax2.barh(metric_labels, improvements, color=bar_colors, alpha=0.8)
        ax2.axvline(0, color='black', linewidth=0.8, linestyle='--')
        ax2.set_xlabel('改善量（GNN - LLM-like）')
        ax2.set_title('GNN Stigmergy の改善効果\n（LLM-likeとの比較）')
        ax2.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"図を保存しました: {save_path}")
        else:
            plt.show()
        plt.close()

    except Exception as e:
        print(f"可視化をスキップ（{e}）")


if __name__ == '__main__':
    # 比較実験を実行
    results = run_comparison_experiment(
        n_nodes=10,
        node_dim=16,
        num_layers=3,
        n_trials=5
    )

    # 結果表示
    print_results(results)

    # 可視化
    visualize_results(results, save_path='comparison_results.png')

    print("=" * 70)
    print("比較実験が完了しました。")
    print("次のステップ:")
    print("  1. 数学問題（方程式）をグラフ化したテストを追加する")
    print("  2. 実際のLLMエージェントとGNN環境層を統合する")
    print("  3. Groupthink/Reflectionのバランス評価実験を追加する")
    print("=" * 70)
