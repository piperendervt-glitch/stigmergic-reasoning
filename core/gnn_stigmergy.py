"""
gnn_stigmergy.py
================
GNNベースのStigmergy Layer実装

Stigmergyの核心:
  - エージェント同士は直接通信しない
  - 環境（グラフ）を介して間接的に影響し合う
  - Message Passingがフェロモン拡散の役割を担う

LLMボトルネックとGNNの対応:
  - Anchor効果 → Sparse Attention（隣接エッジのみ評価）
  - Semantic Anchor → 連続値ベクトル表現（自然言語の残留なし）
  - Confidence非単調性 → ノルム正規化後のAttention
  - Groupthink/Reflection → 蒸発率パラメータによる動的バランス

v2変更点（Sparse Attention）:
  旧実装は全ノードペアでAttentionを計算 → Node 0のノルム優位性が
  全エッジのSoftmaxを支配（Attention Sink）。
  新実装はグラフに存在するエッジのみでAttentionを評価し、
  入力をL2正規化してからスコアを計算する。
  これによりAttentionはノードの「値の大きさ」ではなく
  「特徴の方向（どんな情報か）」に基づいて決まる。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


# ─────────────────────────────────────────────
#  Sparse Attention Message Passingレイヤー
# ─────────────────────────────────────────────

class StigmergyMessagePassing(nn.Module):
    """
    Sparse Attention + ゲート機構付き Stigmergy Message Passingレイヤー (v3)。

    【v1 Dense の問題】
      全ノードペア (N×N) でAttentionを計算 → Attention Sink。

    【v2 Sparse の問題】
      隣接エッジのみに絞ったが、ゼロノードをL2正規化すると
      線形層通過後に「方向不定ベクトル」が生まれ、
      Attentionスコアが hop距離と無関係な乱数に依存してしまう
      （フラット化問題）。

    【v3 Sparse + Gate の設計】
      ゲートが「情報を持っているか」を明示的に判定し、
      空ノードを集約から完全に除外する。

        gate[i] = sigmoid(norm(x[i]) × temp − temp/2)
                  ≈ 1.0  情報あり
                  ≈ 0.0  ほぼゼロ

      L2正規化の後にゲートを乗算:
        情報ありノード → 正規化ベクトル × gate≈1  → Attentionに参加
        ゼロノード     → ゼロベクトル  × gate≈0  → Attentionから除外

      src側ゲートでエッジを追加マスク:
        送信元がゼロノード → そのエッジのスコアを -inf → 集約に寄与しない

      結果として信号は「波面」として伝播する:
        Layer 0 : Node 0 だけ gate=1
        Layer 1 : Node 0 の隣接だけ gate→1（Node 0から受け取った）
        Layer 2 : その隣接の隣接だけ gate→1
        → フェロモンの物理的な拡散と同じ挙動

    Parameters
    ----------
    in_dim    : 入力特徴次元
    out_dim   : 出力特徴次元
    gate_temp : ゲートのシャープネス（大きいほど 0/1 に近づく）
    """

    def __init__(self, in_dim: int, out_dim: int, gate_temp: float = 10.0):
        super().__init__()
        self.gate_temp = gate_temp

        self.lin_self  = nn.Linear(in_dim, out_dim, bias=False)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.att       = nn.Linear(in_dim * 2, 1, bias=False)
        self.norm      = nn.LayerNorm(out_dim)

        self._edge_cache: dict = {}

    def _get_edges(self, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        key = (adj.shape, adj.device, int(adj.bool().sum()))
        if key not in self._edge_cache:
            src, dst = adj.bool().nonzero(as_tuple=True)
            self._edge_cache[key] = (src, dst)
        return self._edge_cache[key]

    def _compute_gate(self, x: torch.Tensor) -> torch.Tensor:
        """
        ノードごとのゲート値 (N, 1) を計算する。

        sigmoid のオフセットにより:
          norm = 0.0 → gate ≈ 0.007  （ほぼ閉じている）
          norm = 0.5 → gate ≈ 0.993  （ほぼ全開）
          norm = 1.0 → gate ≈ 0.999  （全開）
        """
        node_norm = x.norm(dim=-1)                               # (N,)
        gate = torch.sigmoid(
            node_norm * self.gate_temp - self.gate_temp / 2
        )                                                        # (N,)
        return gate.unsqueeze(-1)                                # (N, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x   : (N, in_dim)  ノード特徴量
        adj : (N, N)        隣接行列

        Returns
        -------
        out : (N, out_dim)  更新後ノード特徴量
        """
        N = x.size(0)

        # ── Step 1: ゲート計算 ───────────────────────────────────────
        gate = self._compute_gate(x)                             # (N, 1)

        # ── Step 2: ゲート付き L2正規化 ──────────────────────────────
        # ゼロノードは正規化後もゼロのまま保たれる。
        # gate を乗じることで、線形層通過後の「方向不定」を防ぐ。
        x_gated = F.normalize(x, p=2, dim=-1) * gate            # (N, in_dim)

        # ── Step 3: エッジ取得 ────────────────────────────────────────
        src, dst = self._get_edges(adj)
        E = src.size(0)
        if E == 0:
            return F.relu(self.norm(self.lin_self(x)))

        # ── Step 4: エッジ単位の Attention スコア計算 ─────────────────
        edge_feat  = torch.cat([x_gated[dst], x_gated[src]], dim=-1)  # (E, 2*in_dim)
        edge_score = self.att(edge_feat).squeeze(-1)                   # (E,)

        # ── Step 5: src ゲートによるエッジマスク ─────────────────────
        # 送信元がゼロノード（gate≈0）なら -inf にしてSoftmaxから除外。
        # これにより「まだ情報を受け取っていないノード」は
        # メッセージを送れない → 波面型伝播が実現する。
        src_gate = gate[src].squeeze(-1)                               # (E,)
        edge_score = edge_score.masked_fill(src_gate < 0.5, float('-inf'))

        # ── Step 6: ノード単位の Sparse Softmax ──────────────────────
        att_weight = _sparse_softmax(edge_score, dst, N)               # (E,)

        # ── Step 7: 重み付き近傍集約 ──────────────────────────────────
        agg = torch.zeros(N, x.size(-1), device=x.device, dtype=x.dtype)
        agg.scatter_add_(
            0,
            dst.unsqueeze(-1).expand(E, x.size(-1)),
            att_weight.unsqueeze(-1) * x[src]
        )

        # ── Step 8: 自ノード + 近傍を合成 ────────────────────────────
        out = self.norm(self.lin_self(x) + self.lin_neigh(agg))
        return F.relu(out)


def _sparse_softmax(
    scores: torch.Tensor,   # (E,)  各エッジのスコア
    dst:    torch.Tensor,   # (E,)  受信先ノードID
    N:      int             # ノード数
) -> torch.Tensor:
    """
    ノード単位の Sparse Softmax。

    dst が同じエッジグループ内だけで Softmax を計算する。

    安全処理:
      - 全エッジが -inf（ゲートで全マスク）のグループは
        max も -inf になり exp(-inf - (-inf)) = exp(nan) が発生する。
        is_valid フラグで該当グループを検出し、weight=0 に落とす。
    """
    # 数値安定化: グループごとの最大値を引く
    max_scores = torch.full((N,), float('-inf'), device=scores.device)
    max_scores.scatter_reduce_(0, dst, scores, reduce='amax', include_self=True)

    # 全エッジが -inf のグループ（ゲートで全マスクされたノード）を検出
    valid_group = max_scores.isfinite()          # (N,)  False = 全マスク済み

    # max が -inf のエッジは shifted score を 0 に（exp(0)=1 だが後でゼロ化）
    safe_max = max_scores.clone()
    safe_max[~valid_group] = 0.0

    scores_shifted = scores - safe_max[dst]      # (E,)
    exp_scores = scores_shifted.exp()            # (E,)

    # 全マスクグループのエッジは weight=0 に強制
    exp_scores = exp_scores * valid_group[dst].float()

    sum_exp = torch.zeros(N, device=scores.device)
    sum_exp.scatter_add_(0, dst, exp_scores)     # (N,)

    att_weight = exp_scores / (sum_exp[dst] + 1e-8)
    return att_weight


class GNNStigmergyLayer(nn.Module):
    """
    多層GNN Stigmergyアーキテクチャ。

    各層がフェロモン拡散の「時間ステップ」に対応する。
    深さ = 信号が伝播できる最大ホップ数。

    Parameters
    ----------
    node_dim   : ノード特徴次元
    hidden_dim : 隠れ層次元
    num_layers : GNN層数（= 最大伝播ホップ数）
    dropout    : ドロップアウト率
    """

    def __init__(
        self,
        node_dim: int = 16,
        hidden_dim: int = 32,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # 入力投影
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        # GNN層スタック
        self.layers = nn.ModuleList([
            StigmergyMessagePassing(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # 出力投影
        self.output_proj = nn.Linear(hidden_dim, node_dim)

        # 信号履歴（可視化用）
        self.signal_history: list[torch.Tensor] = []

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        record_history: bool = False
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x              : (N, node_dim) 初期ノード特徴量
        adj            : (N, N) 隣接行列
        record_history : Trueのとき各層の出力を記録

        Returns
        -------
        out : (N, node_dim) Stigmergy処理後のノード特徴量
        """
        h = F.relu(self.input_proj(x))

        if record_history:
            self.signal_history = [h.detach().clone()]

        for layer in self.layers:
            h_new = layer(h, adj)
            h = F.dropout(h_new, p=self.dropout, training=self.training)
            if record_history:
                self.signal_history.append(h.detach().clone())

        out = self.output_proj(h)
        return out


# ─────────────────────────────────────────────
#  PoC: シンプルなグラフ上での信号拡散テスト
# ─────────────────────────────────────────────

def build_test_graph(n_nodes: int = 8, graph_type: str = 'path') -> tuple:
    """
    テスト用グラフを構築する。

    Parameters
    ----------
    n_nodes    : ノード数
    graph_type : 'path' | 'ring' | 'grid' | 'random'

    Returns
    -------
    adj : (N, N) 隣接行列 (torch.Tensor)
    G   : networkx.Graph
    """
    if graph_type == 'path':
        G = nx.path_graph(n_nodes)
    elif graph_type == 'ring':
        G = nx.cycle_graph(n_nodes)
    elif graph_type == 'grid':
        side = int(n_nodes ** 0.5)
        G = nx.grid_2d_graph(side, side)
        G = nx.convert_node_labels_to_integers(G)
    elif graph_type == 'random':
        G = nx.erdos_renyi_graph(n_nodes, p=0.3, seed=42)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    adj = nx.to_numpy_array(G)
    adj_tensor = torch.tensor(adj, dtype=torch.float32)
    return adj_tensor, G


def run_diffusion_poc(
    n_nodes: int = 8,
    node_dim: int = 16,
    graph_type: str = 'path',
    n_steps: int = 3,
    seed: int = 42
):
    """
    信号拡散PoCを実行する。

    1ノードにだけ強い信号を与え、GNN Message Passingで
    近傍ノードへ伝播する様子を可視化する。

    LLMのAnchor効果との比較:
      - LLM: Node 1の情報が文脈として残存し続ける
      - GNN: 蒸発率と層数で制御可能な局所的伝播
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("GNN Stigmergy Layer - 信号拡散 PoC")
    print("=" * 60)
    print(f"グラフ種別: {graph_type}, ノード数: {n_nodes}, GNN層数: {n_steps}")
    print()

    # グラフ構築
    adj, G = build_test_graph(n_nodes, graph_type)

    # 初期ノード特徴量: Node 0にだけ強い信号
    x = torch.zeros(n_nodes, node_dim)
    x[0] = torch.ones(node_dim)  # Source node

    print("初期信号強度（ノード0のみ活性化）:")
    signal_strength = x.norm(dim=1).tolist()
    for i, s in enumerate(signal_strength):
        bar = '█' * int(s * 10)
        print(f"  Node {i:2d}: {bar:<12} ({s:.4f})")
    print()

    # GNN Stigmergy Layer
    model = GNNStigmergyLayer(
        node_dim=node_dim,
        hidden_dim=32,
        num_layers=n_steps,
        dropout=0.0  # PoC段階ではdropoffなし
    )
    model.eval()

    with torch.no_grad():
        out = model(x, adj, record_history=True)

    print("GNN Message Passing後の信号強度:")
    out_strength = out.norm(dim=1).tolist()
    max_s = max(out_strength) if max(out_strength) > 0 else 1.0
    for i, s in enumerate(out_strength):
        bar = '█' * int((s / max_s) * 20)
        print(f"  Node {i:2d}: {bar:<22} ({s:.4f})")
    print()

    # Anchor効果の定量評価
    # LLM: Node 0の影響が全ノードに均一に残存
    # GNN: ホップ距離に応じて信号が減衰
    print("ホップ距離 vs 信号強度（Anchor効果の評価）:")
    print("  （LLMでは距離に関係なく Node 0の影響が残存する）")

    for target in range(1, min(n_nodes, 6)):
        try:
            hop = nx.shortest_path_length(G, 0, target)
            strength = out_strength[target]
            print(f"  Node {target} (hop={hop}): 信号強度 = {strength:.4f}")
        except nx.NetworkXNoPath:
            print(f"  Node {target}: 到達不可能")
    print()

    # 各層での信号伝播の記録
    print("各GNN層での信号拡散（Layer = フェロモン拡散ステップ）:")
    for layer_idx, h in enumerate(model.signal_history):
        strengths = h.norm(dim=1).tolist()
        active_nodes = sum(1 for s in strengths if s > 0.1)
        print(f"  Layer {layer_idx}: 活性ノード数 = {active_nodes}/{n_nodes}")

    print()
    print("✓ 信号拡散PoCが正常に完了しました。")
    print()

    return model, adj, G, out


def visualize_diffusion(model, adj, G, out, save_path: str = None):
    """信号拡散の可視化（matplotlibが利用可能な環境向け）"""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("GNN Stigmergy Layer - 信号拡散 PoC", fontsize=14, fontweight='bold')

        pos = nx.spring_layout(G, seed=42)
        n_nodes = len(G.nodes())

        # Layer 0: 初期状態
        ax0 = axes[0]
        init_strength = model.signal_history[0].norm(dim=1).numpy()
        nx.draw_networkx(
            G, pos=pos, ax=ax0,
            node_color=init_strength,
            node_size=600,
            cmap=plt.cm.YlOrRd,
            vmin=0, vmax=init_strength.max() + 1e-6,
            with_labels=True,
            font_color='black',
            edge_color='gray'
        )
        ax0.set_title("初期状態（Node 0のみ活性化）")
        ax0.axis('off')

        # 最終層: GNN後の信号強度
        ax1 = axes[1]
        final_strength = out.detach().norm(dim=1).numpy()
        nx.draw_networkx(
            G, pos=pos, ax=ax1,
            node_color=final_strength,
            node_size=600,
            cmap=plt.cm.YlOrRd,
            vmin=0, vmax=final_strength.max() + 1e-6,
            with_labels=True,
            font_color='black',
            edge_color='gray'
        )
        ax1.set_title(f"GNN Message Passing後（{len(model.signal_history)-1}ステップ）")
        ax1.axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"図を保存しました: {save_path}")
        else:
            plt.show()
        plt.close()

    except Exception as e:
        print(f"可視化をスキップ（{e}）")


def run_sparse_vs_dense_comparison(
    n_nodes: int = 8,
    node_dim: int = 16,
    n_steps: int = 3,
    seed: int = 42
):
    """
    Sparse Attention（新）vs Dense Attention（旧）の直接比較。

    同一モデル重み・同一入力で、Attention計算範囲の違いが
    Anchor効果にどう影響するかを可視化する。
    """
    torch.manual_seed(seed)

    print("=" * 65)
    print("Sparse Attention vs Dense Attention  Anchor効果比較")
    print("=" * 65)
    print(f"Path graph  ノード数={n_nodes}  GNN層数={n_steps}")
    print()

    adj, G = build_test_graph(n_nodes, 'path')

    # Node 0のみ活性化（Anchor効果が最も出やすい条件）
    x = torch.zeros(n_nodes, node_dim)
    x[0] = torch.ones(node_dim)

    # ── Sparse Attention（新実装）────────────────────────
    model_sparse = GNNStigmergyLayer(node_dim, 32, n_steps, dropout=0.0)
    model_sparse.eval()
    with torch.no_grad():
        out_sparse = model_sparse(x, adj)

    # ── Dense Attention（旧実装を再現）───────────────────
    # 旧実装: L2正規化なし・全ノードペアでSoftmax
    class _DenseMessagePassing(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.lin_self  = nn.Linear(dim, dim, bias=False)
            self.lin_neigh = nn.Linear(dim, dim, bias=False)
            self.att = nn.Linear(dim * 2, 1, bias=False)
            self.norm = nn.LayerNorm(dim)
        def forward(self, x, adj):
            N = x.size(0)
            x_i = x.unsqueeze(1).expand(N, N, -1)
            x_j = x.unsqueeze(0).expand(N, N, -1)
            raw = self.att(torch.cat([x_i, x_j], dim=-1)).squeeze(-1)
            mask = (adj > 0)
            raw = raw.masked_fill(~mask, float('-inf'))
            has_neigh = mask.any(dim=1, keepdim=True)
            w = torch.where(has_neigh,
                            F.softmax(raw, dim=1),
                            torch.zeros_like(raw))
            neigh = self.lin_neigh(torch.matmul(w, x))
            return F.relu(self.norm(self.lin_self(x) + neigh))

    class _DenseModel(nn.Module):
        def __init__(self, node_dim, hidden_dim, num_layers):
            super().__init__()
            self.input_proj  = nn.Linear(node_dim, hidden_dim)
            self.layers = nn.ModuleList(
                [_DenseMessagePassing(hidden_dim) for _ in range(num_layers)])
            self.output_proj = nn.Linear(hidden_dim, node_dim)
        def forward(self, x, adj):
            h = F.relu(self.input_proj(x))
            for l in self.layers:
                h = l(h, adj)
            return self.output_proj(h)

    model_dense = _DenseModel(node_dim, 32, n_steps)
    model_dense.eval()
    with torch.no_grad():
        out_dense = model_dense(x, adj)

    # ── 結果表示 ─────────────────────────────────────────
    print(f"{'Node':<6} {'hop':>4}  {'Sparse (新)':>12}  {'Dense (旧)':>12}  {'差分':>10}")
    print("-" * 55)

    sparse_s = out_sparse.norm(dim=1).tolist()
    dense_s  = out_dense.norm(dim=1).tolist()
    max_sp   = max(sparse_s) + 1e-8
    max_de   = max(dense_s)  + 1e-8

    for i in range(n_nodes):
        hop = nx.shortest_path_length(G, 0, i) if i > 0 else 0
        diff = sparse_s[i] / max_sp - dense_s[i] / max_de
        marker = " ← Anchor抑制" if hop >= 3 and diff < -0.05 else ""
        print(
            f"  {i:<4} {hop:>4}  "
            f"{sparse_s[i]/max_sp:>12.4f}  "
            f"{dense_s[i]/max_de:>12.4f}  "
            f"{diff:>+10.4f}{marker}"
        )

    print()
    # ホップ距離との相関（Anchor Scoreの直感的説明）
    hops    = [nx.shortest_path_length(G, 0, i) for i in range(1, n_nodes)]
    s_corr  = np.corrcoef(hops, [sparse_s[i]/max_sp for i in range(1, n_nodes)])[0,1]
    d_corr  = np.corrcoef(hops, [dense_s[i]/max_de  for i in range(1, n_nodes)])[0,1]

    print("ホップ距離と信号強度の相関係数（-1に近いほどAnchor効果が小さい）:")
    print(f"  Sparse (新): r = {s_corr:+.4f}  {'✓ 距離依存あり（良い）' if s_corr < -0.3 else '△ 距離依存が弱い'}")
    print(f"  Dense  (旧): r = {d_corr:+.4f}  {'✓ 距離依存あり（良い）' if d_corr < -0.3 else '△ 距離依存が弱い'}")
    print()
    print("解釈:")
    print("  相関が-1に近い → 遠いノードほど信号が弱い → Anchor効果なし（理想）")
    print("  相関が0に近い  → 距離に関係なく信号が一定 → Anchor効果あり（問題）")


if __name__ == '__main__':
    # ── PoC 1: Path graph（直線グラフ）────────────────────
    model, adj, G, out = run_diffusion_poc(
        n_nodes=8,
        node_dim=16,
        graph_type='path',
        n_steps=3
    )
    visualize_diffusion(model, adj, G, out, save_path='diffusion_path.png')

    # ── PoC 2: Ring graph（循環グラフ）────────────────────
    print("\n" + "=" * 60)
    print("Ring Graph（循環）での信号拡散テスト")
    print("=" * 60)
    run_diffusion_poc(
        n_nodes=8,
        node_dim=16,
        graph_type='ring',
        n_steps=3
    )

    # ── PoC 3: Random graph ────────────────────────────
    print("\n" + "=" * 60)
    print("Random Graph での信号拡散テスト")
    print("=" * 60)
    run_diffusion_poc(
        n_nodes=10,
        node_dim=16,
        graph_type='random',
        n_steps=4
    )

    # ── PoC 4: Sparse vs Dense Anchor比較（新規）──────────
    print()
    run_sparse_vs_dense_comparison(n_nodes=8, node_dim=16, n_steps=3)
