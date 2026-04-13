"""
root_layer_gnn.py
=================
GNN版 Root Layer / Pheromone Layer

役割:
  - フェロモン（信号）の堆積・蒸発・強化を管理する「環境層」
  - エージェントはこの層への読み書きのみを行う（直接通信なし）
  - GNNのMessage Passingをフェロモン拡散の物理メカニズムとして使用

Stigmergy原則の実装:
  1. 堆積 (Deposit)    : エージェントが訪問したノードにフェロモンを置く
  2. 拡散 (Diffusion)  : GNN Message Passingで近傍に広がる
  3. 蒸発 (Evaporation): 時間とともにフェロモンが減衰する
  4. 強化 (Amplification): 良い経路のフェロモンを強化する
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional


class PheromoneField(nn.Module):
    """
    フェロモン場（Pheromone Field）。

    グラフ上の各ノードが持つフェロモン濃度を管理する。
    蒸発率と強化率で信号の持続性を制御できる。

    Parameters
    ----------
    n_nodes      : ノード数
    pheromone_dim: フェロモンの次元数（異なる種類のフェロモンを同時管理）
    evaporation  : 蒸発率 (0.0 ~ 1.0, 大きいほど速く消える)
    """

    def __init__(
        self,
        n_nodes: int,
        pheromone_dim: int = 4,
        evaporation: float = 0.1
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.pheromone_dim = pheromone_dim
        self.evaporation = evaporation

        # フェロモン濃度テンソル（学習パラメータではなく状態変数）
        self.register_buffer(
            'pheromone',
            torch.zeros(n_nodes, pheromone_dim)
        )
        # 時間ステップカウンタ
        self.register_buffer('timestep', torch.tensor(0))

    def deposit(self, node_ids: list[int], amounts: torch.Tensor):
        """
        指定ノードにフェロモンを堆積させる。

        Parameters
        ----------
        node_ids : フェロモンを置くノードのインデックスリスト
        amounts  : (len(node_ids), pheromone_dim) 堆積量
        """
        for i, nid in enumerate(node_ids):
            self.pheromone[nid] += amounts[i]

    def evaporate(self):
        """1タイムステップ分のフェロモン蒸発を適用する。"""
        self.pheromone *= (1.0 - self.evaporation)
        self.timestep += 1

    def read(self, node_id: int) -> torch.Tensor:
        """指定ノードのフェロモン濃度を読み取る。"""
        return self.pheromone[node_id].clone()

    def total_concentration(self) -> float:
        """全ノードの総フェロモン濃度。"""
        return self.pheromone.sum().item()


class GNNDiffusionLayer(nn.Module):
    """
    フェロモン拡散を担うGNN層。

    グラフ上でフェロモンを物理的に拡散させる。
    隣接ノードへの拡散係数はエッジの重みで制御。

    Parameters
    ----------
    pheromone_dim : フェロモン次元
    diffusion_rate: 隣接ノードへの拡散率 (0.0 ~ 1.0)
    """

    def __init__(self, pheromone_dim: int = 4, diffusion_rate: float = 0.3):
        super().__init__()
        self.diffusion_rate = diffusion_rate
        # 拡散変換（フェロモンの質を保ちながら広げる）
        self.diffusion_transform = nn.Linear(
            pheromone_dim, pheromone_dim, bias=False
        )
        # 単位行列に近い初期化（拡散が均一になるように）
        nn.init.eye_(self.diffusion_transform.weight)

    def forward(
        self,
        pheromone: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pheromone : (N, pheromone_dim)
        adj       : (N, N) 隣接行列

        Returns
        -------
        diffused  : (N, pheromone_dim)
        """
        # 度数正規化（各ノードの近傍数で正規化）
        degree = adj.sum(dim=1, keepdim=True).clamp(min=1)
        norm_adj = adj / degree

        # 近傍からの流入フェロモン
        neighbor_pheromone = torch.matmul(norm_adj, pheromone)
        neighbor_transformed = self.diffusion_transform(neighbor_pheromone)

        # 自ノードのフェロモン保持 + 近傍からの流入
        diffused = (
            (1 - self.diffusion_rate) * pheromone
            + self.diffusion_rate * neighbor_transformed
        )
        return F.relu(diffused)


class RootLayerGNN(nn.Module):
    """
    GNN版 Root Layer（Pheromone Layer）。

    Stigmergyの「環境」として機能する中核コンポーネント。
    エージェントはこの層を介してのみ間接的に通信する。

    設計思想:
      LLMのSemantic Anchorと異なり、フェロモンは連続値ベクトルであり
      「文脈として保持される自然言語」ではない。
      蒸発・拡散パラメータによって信号の寿命と影響範囲を明示的に制御できる。

    Parameters
    ----------
    n_nodes       : ノード数
    pheromone_dim : フェロモン次元
    n_diffuse_steps: 1タイムステップあたりの拡散回数
    evaporation   : 蒸発率
    diffusion_rate: 拡散率
    """

    def __init__(
        self,
        n_nodes: int,
        pheromone_dim: int = 4,
        n_diffuse_steps: int = 2,
        evaporation: float = 0.1,
        diffusion_rate: float = 0.3
    ):
        super().__init__()

        self.n_nodes = n_nodes
        self.pheromone_dim = pheromone_dim
        self.n_diffuse_steps = n_diffuse_steps

        # フェロモン場
        self.field = PheromoneField(n_nodes, pheromone_dim, evaporation)

        # GNN拡散層（複数ステップ）
        self.diffusion_layers = nn.ModuleList([
            GNNDiffusionLayer(pheromone_dim, diffusion_rate)
            for _ in range(n_diffuse_steps)
        ])

        # 隣接行列（登録してデバイス移動に対応）
        self.register_buffer('adj', torch.zeros(n_nodes, n_nodes))

        # 履歴ログ
        self.history: list[dict] = []

    def set_graph(self, adj: torch.Tensor):
        """グラフ構造を設定する。"""
        self.adj = adj.float()

    def step(
        self,
        agent_deposits: Optional[dict[int, torch.Tensor]] = None,
        record: bool = True
    ) -> torch.Tensor:
        """
        1タイムステップを実行する。

        処理順序:
          1. エージェントからのフェロモン堆積
          2. GNN拡散
          3. 蒸発

        Parameters
        ----------
        agent_deposits : {node_id: deposit_amount} エージェントの堆積情報
        record         : 履歴を記録するか

        Returns
        -------
        current_pheromone : (N, pheromone_dim) 現在のフェロモン場
        """
        # 1. 堆積
        if agent_deposits:
            for node_id, amount in agent_deposits.items():
                if amount.dim() == 1:
                    amount = amount.unsqueeze(0)
                self.field.deposit([node_id], amount)

        # 2. GNN拡散
        pheromone = self.field.pheromone.clone()
        with torch.no_grad():
            for layer in self.diffusion_layers:
                pheromone = layer(pheromone, self.adj)
        self.field.pheromone.copy_(pheromone)

        # 3. 蒸発
        self.field.evaporate()

        if record:
            self.history.append({
                'timestep': self.field.timestep.item(),
                'pheromone_snapshot': self.field.pheromone.clone(),
                'total_concentration': self.field.total_concentration()
            })

        return self.field.pheromone.clone()

    def get_signal(self, node_id: int) -> torch.Tensor:
        """エージェントが環境から信号を読み取る。"""
        return self.field.read(node_id)

    def reset(self):
        """フェロモン場をリセットする。"""
        self.field.pheromone.zero_()
        self.field.timestep.zero_()
        self.history.clear()


# ─────────────────────────────────────────────
#  PoC: Pheromone Layerの動作確認
# ─────────────────────────────────────────────

def run_pheromone_poc():
    """
    フェロモン層の動作確認PoC。

    シナリオ:
      - 8ノードのpath graph
      - Node 0と Node 7に異なるエージェントがフェロモンを置く
      - 時間経過とともに拡散・蒸発する様子を観察
      - Anchor効果の観測: LLMとGNNの振る舞いを比較
    """
    print("=" * 60)
    print("Root Layer GNN - Pheromone Layer PoC")
    print("=" * 60)

    n_nodes = 8
    pheromone_dim = 4

    # グラフ構築
    G = nx.path_graph(n_nodes)
    adj = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32)

    # Root Layerの初期化
    root_layer = RootLayerGNN(
        n_nodes=n_nodes,
        pheromone_dim=pheromone_dim,
        n_diffuse_steps=2,
        evaporation=0.15,
        diffusion_rate=0.3
    )
    root_layer.set_graph(adj)

    print("\n【Anchor効果テスト】")
    print("Node 0 に強いフェロモンを置き、時間とともにどう変化するか観察")
    print(f"パラメータ: evaporation=0.15, diffusion_rate=0.3")
    print()

    # 初期堆積: Node 0に強い信号
    initial_deposit = torch.ones(1, pheromone_dim) * 5.0
    root_layer.field.deposit([0], initial_deposit)

    print("タイムステップ | Node 0 | Node 1 | Node 2 | Node 3 | Node 4 | 合計濃度")
    print("-" * 75)

    for t in range(10):
        # 追加堆積なし（初期フェロモンだけを観察）
        pheromone = root_layer.step(agent_deposits=None, record=True)

        strengths = pheromone.norm(dim=1).tolist()
        total = sum(strengths)
        row = f"  t={t+1:2d}         | "
        row += " | ".join(f"{strengths[i]:6.3f}" for i in range(5))
        row += f" | {total:7.3f}"
        print(row)

    print()
    print("観察ポイント:")
    print("  ✓ Node 0の信号が時間とともに減衰（LLMのAnchor効果と対照的）")
    print("  ✓ 近傍ノードへの拡散が距離に応じて減衰")
    print("  ✓ 蒸発率で信号の寿命を明示的に制御可能")

    print()
    print("─" * 60)
    print()

    # ── テスト2: 2エージェントの間接通信 ──
    root_layer.reset()
    print("【間接通信テスト（Stigmergy）】")
    print("Agent A (Node 0) と Agent B (Node 7) が環境を介して情報共有")
    print()

    pheromone_A = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # エージェントAの信号
    pheromone_B = torch.tensor([[0.0, 0.0, 1.0, 0.0]])  # エージェントBの信号

    for t in range(8):
        deposits = {}
        # エージェントAが偶数ステップで堆積
        if t % 2 == 0:
            deposits[0] = pheromone_A
        # エージェントBが奇数ステップで堆積
        if t % 2 == 1:
            deposits[7] = pheromone_B

        pheromone = root_layer.step(agent_deposits=deposits, record=False)

        # 中央ノード（Node 3, 4）での信号混合を観察
        node3_signal = pheromone[3]
        a_component = node3_signal[0].item()  # Agent Aのフェロモン成分
        b_component = node3_signal[2].item()  # Agent Bのフェロモン成分
        print(f"  t={t+1}: Node 3 - A成分={a_component:.4f}, B成分={b_component:.4f}")

    print()
    print("  ✓ 2エージェントの信号が中央ノードで混合 → Stigmergy成立")
    print("  ✓ 直接通信なしで間接的な情報共有が実現")
    print()
    print("✓ Pheromone Layer PoCが正常に完了しました。")


if __name__ == '__main__':
    run_pheromone_poc()
