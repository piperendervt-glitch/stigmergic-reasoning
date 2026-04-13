"""
math_problem_gnn.py
===================
数学問題をグラフ構造で表現し、GNN Stigmergy Layerで推論するシステム。

設計思想:
  - 問題の「解法ステップ」を各ノードに割り当てる
  - エッジは「このステップの後にこのステップが来る」依存関係を表す
  - GNNのMessage Passingで各ノードが文脈を受け取りながら推論する
  - 最終的に「答えノード」の出力を分類器で解釈して答えを出す

LLMとの構造的な違い:
  LLM : 全ステップを逐次的に処理。先頭の情報が後続に残存（Anchor効果）
  GNN : 各ステップが並列にMessage Passingで情報交換。
        依存関係だけが情報の流れを規定する（グラフの構造が制約）

対応問題:
  1. 一次方程式  (例: 2x + 3 = 7)
  2. 二次方程式  (例: x^2 - 5x + 6 = 0)
  3. 連立方程式  (例: x + y = 5, x - y = 1)
  4. 文章題      (例: リンゴが3個...)
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import random
import math

from core.gnn_stigmergy import GNNStigmergyLayer


# ─────────────────────────────────────────────
#  特徴量エンコーダ
# ─────────────────────────────────────────────

# ノード特徴量レイアウト (dim=32)
# [ 0- 3] ノードタイプ (one-hot 4種)
# [ 4- 9] 数値スロット (coeff_a, coeff_b, const, rhs, answer, unused)
# [10-13] 演算フラグ  (add, sub, mul, div)
# [14-17] 変数フラグ  (x存在, y存在, x^2存在, 複数変数)
# [18-21] ステップ種別 (simplify, isolate, substitute, verify)
# [22-31] 予備スロット（拡張用）

NODE_DIM = 32

NODE_TYPES = {
    'problem_root': 0,   # 問題全体を表すルートノード
    'expression':   1,   # 数式・項
    'operation':    2,   # 変形操作（両辺に何かする）
    'answer':       3,   # 答えノード
}

OPS = {'add': 10, 'sub': 11, 'mul': 12, 'div': 13}
VARS = {'x': 14, 'y': 15, 'x2': 16, 'multi': 17}
STEPS = {'simplify': 18, 'isolate': 19, 'substitute': 20, 'verify': 21}

SCALE = 10.0  # 数値を正規化するスケール


def make_node(
    node_type: str,
    coeff_a:   float = 0.0,
    coeff_b:   float = 0.0,
    const:     float = 0.0,
    rhs:       float = 0.0,
    answer:    float = 0.0,
    ops:       list[str] = None,
    variables: list[str] = None,
    step_type: str = None,
) -> torch.Tensor:
    """ノード特徴量ベクトルを生成する。"""
    vec = torch.zeros(NODE_DIM)
    vec[NODE_TYPES[node_type]] = 1.0
    vec[4]  = coeff_a / SCALE
    vec[5]  = coeff_b / SCALE
    vec[6]  = const   / SCALE
    vec[7]  = rhs     / SCALE
    vec[8]  = answer  / SCALE
    for op in (ops or []):
        if op in OPS:
            vec[OPS[op]] = 1.0
    for v in (variables or []):
        if v in VARS:
            vec[VARS[v]] = 1.0
    if step_type in STEPS:
        vec[STEPS[step_type]] = 1.0
    return vec


# ─────────────────────────────────────────────
#  問題グラフビルダー
# ─────────────────────────────────────────────

@dataclass
class MathProblemGraph:
    """
    数学問題のグラフ表現。

    Attributes
    ----------
    name        : 問題名
    problem_str : 問題の文字列表現
    x           : (N, NODE_DIM) ノード特徴量
    adj         : (N, N)        隣接行列
    answer_node : 答えノードのインデックス
    answer_value: 正解の値（評価用）
    node_labels : 可視化用のノードラベル
    """
    name:         str
    problem_str:  str
    x:            torch.Tensor
    adj:          torch.Tensor
    answer_node:  int
    answer_value: float
    node_labels:  list[str] = field(default_factory=list)


def build_linear_equation(a: float, b: float, c: float) -> MathProblemGraph:
    """
    一次方程式 ax + b = c のグラフを構築する。

    グラフ構造:
      0 (root: ax+b=c)
      ├─ 1 (expr: ax+b, 左辺)
      │   └─ 3 (op: -b, 定数を右辺へ移項)
      ├─ 2 (expr: c, 右辺)
      │   └─ 3
      │        └─ 4 (expr: ax=c-b, 中間式)
      │               └─ 5 (op: /a, 係数で割る)
      │                     └─ 6 (answer: x = (c-b)/a)
    """
    true_answer = (c - b) / a

    nodes = [
        make_node('problem_root', coeff_a=a, const=b, rhs=c,
                  variables=['x'], step_type='simplify'),
        make_node('expression',   coeff_a=a, const=b,
                  variables=['x']),
        make_node('expression',   rhs=c),
        make_node('operation',    const=b, ops=['sub'],
                  step_type='isolate'),
        make_node('expression',   coeff_a=a, rhs=c - b,
                  variables=['x']),
        make_node('operation',    coeff_a=a, ops=['div'],
                  step_type='isolate'),
        make_node('answer',       answer=true_answer,
                  variables=['x']),
    ]

    edges = [(0,1),(0,2),(1,3),(2,3),(3,4),(4,5),(5,6)]
    N = len(nodes)
    adj = torch.zeros(N, N)
    for s, d in edges:
        adj[s][d] = adj[d][s] = 1.0

    labels = [
        f"{a}x+{b}={c}",
        f"{a}x+{b}",
        f"{c}",
        f"-{b}",
        f"{a}x={c-b}",
        f"÷{a}",
        f"x={true_answer:.2f}",
    ]

    return MathProblemGraph(
        name=f"一次方程式: {a}x + {b} = {c}",
        problem_str=f"{a}x + {b} = {c}",
        x=torch.stack(nodes),
        adj=adj,
        answer_node=6,
        answer_value=true_answer,
        node_labels=labels,
    )


def build_quadratic_equation(a: float, b: float, c: float) -> MathProblemGraph:
    """
    二次方程式 ax^2 + bx + c = 0 のグラフを構築する（判別式→因数分解ルート）。

    グラフ構造:
      0 (root)
      ├─ 1 (discriminant: b^2 - 4ac)
      ├─ 2 (root1: (-b + √D) / 2a)
      └─ 3 (root2: (-b - √D) / 2a)
           └─ 4 (answer: x1, x2)
    """
    D = b**2 - 4*a*c
    if D < 0:
        x1, x2 = float('nan'), float('nan')
    elif D == 0:
        x1 = x2 = -b / (2*a)
    else:
        x1 = (-b + D**0.5) / (2*a)
        x2 = (-b - D**0.5) / (2*a)

    nodes = [
        make_node('problem_root', coeff_a=a, coeff_b=b, const=c,
                  variables=['x', 'x2'], step_type='simplify'),
        make_node('expression',   coeff_a=b, coeff_b=4*a,
                  rhs=D / SCALE,
                  step_type='simplify'),                        # 判別式
        make_node('expression',   answer=x1 if not np.isnan(x1) else 0,
                  variables=['x'], step_type='isolate'),        # 解1
        make_node('expression',   answer=x2 if not np.isnan(x2) else 0,
                  variables=['x'], step_type='isolate'),        # 解2
        make_node('answer',
                  coeff_a=x1 if not np.isnan(x1) else 0,
                  coeff_b=x2 if not np.isnan(x2) else 0,
                  variables=['x']),
    ]

    edges = [(0,1),(1,2),(1,3),(2,4),(3,4)]
    N = len(nodes)
    adj = torch.zeros(N, N)
    for s, d in edges:
        adj[s][d] = adj[d][s] = 1.0

    labels = [
        f"{a}x²+{b}x+{c}=0",
        f"D={D:.1f}",
        f"x₁={x1:.2f}" if not np.isnan(x1) else "解なし",
        f"x₂={x2:.2f}" if not np.isnan(x2) else "解なし",
        f"答え",
    ]

    return MathProblemGraph(
        name=f"二次方程式: {a}x² + {b}x + {c} = 0",
        problem_str=f"{a}x² + {b}x + {c} = 0",
        x=torch.stack(nodes),
        adj=adj,
        answer_node=4,
        answer_value=x1,
        node_labels=labels,
    )


def build_simultaneous_equations(
    a1: float, b1: float, c1: float,
    a2: float, b2: float, c2: float,
) -> MathProblemGraph:
    """
    連立方程式 a1*x + b1*y = c1 / a2*x + b2*y = c2 のグラフ。

    グラフ構造:
      0 (root)
      ├─ 1 (eq1: a1x + b1y = c1)
      ├─ 2 (eq2: a2x + b2y = c2)
      ├─ 3 (op: 消去、代入)
      ├─ 4 (intermediate: x値)
      ├─ 5 (intermediate: y値)
      └─ 6 (answer: x=?, y=?)
    """
    det = a1*b2 - a2*b1
    if abs(det) < 1e-8:
        x_ans = y_ans = float('nan')
    else:
        x_ans = (c1*b2 - c2*b1) / det
        y_ans = (a1*c2 - a2*c1) / det

    nodes = [
        make_node('problem_root', coeff_a=a1, coeff_b=b1,
                  variables=['x', 'y', 'multi'], step_type='simplify'),
        make_node('expression',   coeff_a=a1, coeff_b=b1, rhs=c1,
                  variables=['x', 'y']),
        make_node('expression',   coeff_a=a2, coeff_b=b2, rhs=c2,
                  variables=['x', 'y']),
        make_node('operation',    ops=['sub'], step_type='substitute'),
        make_node('expression',
                  answer=x_ans if not np.isnan(x_ans) else 0,
                  variables=['x'], step_type='isolate'),
        make_node('expression',
                  answer=y_ans if not np.isnan(y_ans) else 0,
                  variables=['y'], step_type='isolate'),
        make_node('answer',
                  coeff_a=x_ans if not np.isnan(x_ans) else 0,
                  coeff_b=y_ans if not np.isnan(y_ans) else 0,
                  variables=['x', 'y', 'multi']),
    ]

    edges = [(0,1),(0,2),(1,3),(2,3),(3,4),(3,5),(4,6),(5,6)]
    N = len(nodes)
    adj = torch.zeros(N, N)
    for s, d in edges:
        adj[s][d] = adj[d][s] = 1.0

    labels = [
        "連立方程式",
        f"{a1}x+{b1}y={c1}",
        f"{a2}x+{b2}y={c2}",
        "消去",
        f"x={x_ans:.2f}" if not np.isnan(x_ans) else "?",
        f"y={y_ans:.2f}" if not np.isnan(y_ans) else "?",
        "答え",
    ]

    return MathProblemGraph(
        name=f"連立方程式: {a1}x+{b1}y={c1}, {a2}x+{b2}y={c2}",
        problem_str=f"{a1}x + {b1}y = {c1}  /  {a2}x + {b2}y = {c2}",
        x=torch.stack(nodes),
        adj=adj,
        answer_node=6,
        answer_value=x_ans,
        node_labels=labels,
    )


# ─────────────────────────────────────────────
#  答えデコーダ
# ─────────────────────────────────────────────

class AnswerDecoder(nn.Module):
    """
    GNNの答えノード出力を数値に変換するデコーダ。

    GNNは連続ベクトルを出力するので、それを回帰で数値に変換する。
    訓練データなしの zero-shot での動作を確認するため、
    ここでは特徴量スロット（answer=スロット）を直接読み取る
    「解釈型デコーダ」と、GNN出力から学習する「回帰型デコーダ」を用意する。
    """

    def __init__(self, node_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, answer_node_feat: torch.Tensor) -> torch.Tensor:
        """(node_dim,) → scalar"""
        return self.regressor(answer_node_feat).squeeze(-1)

    @staticmethod
    def decode_direct(
        answer_node_feat: torch.Tensor,
        answer_slot: int = 8,
    ) -> float:
        """
        GNN出力を通さず、特徴量スロットを直接読む「解釈型」デコーダ。

        GNNのMessage Passingを経た後、答えノードの特徴量スロット[8]が
        どれだけ正解値を保持しているかを確認する。
        """
        return float(answer_node_feat[answer_slot].item() * SCALE)


# ─────────────────────────────────────────────
#  GNN推論エンジン
# ─────────────────────────────────────────────

class MathSolverGNN(nn.Module):
    """
    GNN Stigmergy Layerを使った数学問題ソルバー。

    処理フロー:
      1. 問題グラフを構築（各ステップがノード）
      2. GNN Stigmergy Layerでグラフ全体のMessage Passing
      3. 答えノードの出力をデコーダで数値に変換
      4. 正解と比較して評価

    このシステムの意義:
      単純なforward passで解くのではなく、
      「解法手順の依存関係グラフ」上でGNNが
      情報を集約する過程が推論そのもの。
    """

    def __init__(
        self,
        node_dim:   int = NODE_DIM,
        hidden_dim: int = 64,
        num_layers: int = 4,
    ):
        super().__init__()
        self.gnn = GNNStigmergyLayer(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.0,
        )
        self.decoder = AnswerDecoder(node_dim, hidden_dim)

    def forward(self, problem: MathProblemGraph) -> dict:
        """
        問題グラフを受け取り、GNN推論を実行して結果を返す。

        Returns
        -------
        {
          'gnn_output'      : (N, node_dim) 全ノードのGNN出力,
          'answer_feat'     : (node_dim,)   答えノードの出力,
          'predicted_value' : float         デコーダによる予測値,
          'direct_read'     : float         スロット直読みによる予測値,
          'true_answer'     : float         正解,
          'error'           : float         予測誤差,
        }
        """
        self.eval()
        with torch.no_grad():
            out = self.gnn(problem.x, problem.adj, record_history=True)

        answer_feat = out[problem.answer_node]

        # デコーダによる予測（SCALE で元のスケールに復元）
        predicted = float(self.decoder(answer_feat).item()) * SCALE

        # スロット直読み（特徴量エンコーディングをGNNが保持しているか）
        direct = AnswerDecoder.decode_direct(answer_feat)

        return {
            'gnn_output':       out,
            'answer_feat':      answer_feat,
            'predicted_value':  predicted,
            'direct_read':      direct,
            'true_answer':      problem.answer_value,
            'error':            abs(predicted - problem.answer_value),
        }

    def analyze_message_passing(
        self, problem: MathProblemGraph
    ) -> None:
        """
        Message Passingの過程を可視化する。
        各層で答えノードに向かって情報が集まっていく様子を表示。
        """
        self.eval()
        with torch.no_grad():
            _ = self.gnn(problem.x, problem.adj, record_history=True)

        print(f"  問題: {problem.problem_str}")
        print(f"  グラフ: {problem.x.shape[0]}ノード, "
              f"{int(problem.adj.sum().item()//2)}エッジ")
        print()

        ans_idx = problem.answer_node
        ans_true = problem.answer_value

        print(f"  各GNN層での答えノード(Node {ans_idx})の状態変化:")
        print(f"  {'Layer':>6}  {'ノルム':>8}  {'answer_slot':>12}  "
              f"{'slot×{SCALE}':>10}  {'誤差':>8}")
        print("  " + "─" * 55)

        for layer_idx, h in enumerate(self.gnn.signal_history):
            feat   = h[ans_idx]
            norm   = feat.norm().item()
            slot   = feat[8].item()             # answerスロット
            decoded = slot * SCALE
            error  = abs(decoded - ans_true)
            print(f"  {layer_idx:>6}  {norm:>8.4f}  {slot:>12.4f}  "
                  f"{decoded:>10.4f}  {error:>8.4f}")

        print()

        # 各ノードの最終状態
        final_h = self.gnn.signal_history[-1]
        print(f"  最終層での全ノード状態:")
        print(f"  {'Node':>5}  {'ラベル':>14}  {'ノルム':>8}  {'answer_slot×{SCALE}':>16}")
        print("  " + "─" * 50)
        for i, label in enumerate(problem.node_labels):
            feat  = final_h[i]
            norm  = feat.norm().item()
            slot  = feat[8].item() * SCALE
            marker = " ← 答えノード" if i == ans_idx else ""
            print(f"  {i:>5}  {label:>14}  {norm:>8.4f}  {slot:>16.4f}{marker}")


# ─────────────────────────────────────────────
#  実験実行
# ─────────────────────────────────────────────

def evaluate_information_flow(
    solver: 'MathSolverGNN',
    prob:    MathProblemGraph,
) -> dict:
    """
    GNNが問題を「正しく解いている」かを3軸で評価する。

    評価軸:
      A) 入力保存率  : 入力スロット値が出力にどれだけ保持されているか
      B) 依存構造整合: 答えノードの情報が「直接の親 > ルート」か
      C) 情報集約度  : 答えノードが解に必要なノードを高く参照しているか
    """
    solver.eval()
    with torch.no_grad():
        out = solver.gnn(prob.x, prob.adj, record_history=True)

    final_h = solver.gnn.signal_history[-1]
    ans_idx = prob.answer_node
    ans_vec = final_h[ans_idx]                        # (hidden_dim,)

    # A) 入力保存率
    #    output_proj済みの出力(node_dim) と 入力特徴(node_dim) を比較
    #    signal_history は hidden_dim なので output_proj を通した out を使う
    with torch.no_grad():
        out_proj = solver.gnn(prob.x, prob.adj)       # (N, node_dim)
    ans_proj      = out_proj[ans_idx]                 # (node_dim,)
    input_ans_vec = prob.x[ans_idx]                   # (node_dim,)
    input_sim = F.cosine_similarity(
        ans_proj.unsqueeze(0), input_ans_vec.unsqueeze(0)
    ).item()

    # B) 依存構造整合 (直接親 vs ルートの類似度比較)
    #    グラフのanswerノードへのエッジ元を親とする
    parents = prob.adj[:, ans_idx].nonzero(as_tuple=True)[0].tolist()
    root_sim = F.cosine_similarity(
        final_h[0].unsqueeze(0), ans_vec.unsqueeze(0)
    ).item()
    parent_sims = [
        F.cosine_similarity(
            final_h[p].unsqueeze(0), ans_vec.unsqueeze(0)
        ).item()
        for p in parents
    ]
    max_parent_sim = max(parent_sims) if parent_sims else 0.0
    dependency_ok = max_parent_sim > root_sim

    # C) 解法経路スコア
    #    答えへの経路ノード（root→...→answer）の類似度が
    #    非経路ノードより高いか
    # 経路ノードを BFS で特定
    G = nx.from_numpy_array(prob.adj.numpy())
    try:
        path_nodes = set(nx.shortest_path(G, 0, ans_idx))
    except nx.NetworkXNoPath:
        path_nodes = {0, ans_idx}

    path_sims    = []
    nonpath_sims = []
    for i in range(prob.x.shape[0]):
        if i == ans_idx:
            continue
        s = F.cosine_similarity(
            final_h[i].unsqueeze(0), ans_vec.unsqueeze(0)
        ).item()
        (path_sims if i in path_nodes else nonpath_sims).append(s)

    path_mean    = float(np.mean(path_sims))    if path_sims    else 0.0
    nonpath_mean = float(np.mean(nonpath_sims)) if nonpath_sims else 0.0
    route_ok = path_mean >= nonpath_mean

    return {
        'input_sim':       input_sim,
        'root_sim':        root_sim,
        'max_parent_sim':  max_parent_sim,
        'dependency_ok':   dependency_ok,
        'path_mean':       path_mean,
        'nonpath_mean':    nonpath_mean,
        'route_ok':        route_ok,
        'final_h':         final_h,
        'out':             out,
    }


def run_math_experiments():
    torch.manual_seed(42)

    print("=" * 65)
    print("GNN Stigmergy — 数学問題グラフ推論")
    print("=" * 65)
    print()
    print("評価の考え方:")
    print("  GNNは未訓練なので「数値を計算」する能力は持たない。")
    print("  ここで検証するのは:")
    print("  ① 解法経路に沿った情報集約が起きているか")
    print("  ② ルート（問題文）より直接の親ノードが答えに影響するか")
    print("     （LLMのAnchor効果の有無）")
    print("  ③ 問題の係数が正しくエンコードされ保持されているか")
    print()

    solver = MathSolverGNN(node_dim=NODE_DIM, hidden_dim=64, num_layers=4)

    problems = [
        build_linear_equation(2,  3,  7),    # x=2
        build_linear_equation(3, -6,  9),    # x=5
        build_linear_equation(5,  2, 12),    # x=2
        build_quadratic_equation(1, -5, 6),  # x=3,2
        build_quadratic_equation(1, -3, 2),  # x=2,1
        build_simultaneous_equations(1, 1, 5, 1, -1, 1),    # x=3,y=2
        build_simultaneous_equations(2, 3, 12, 1, -1, 1),   # x=3,y=2
    ]

    print(f"{'問題':<34} {'入力保存':>8} {'依存構造':>8} {'経路整合':>8}")
    print("─" * 65)

    all_evals = []
    for prob in problems:
        ev = evaluate_information_flow(solver, prob)
        a = "✓" if ev['input_sim']    > 0.5 else "✗"
        b = "✓" if ev['dependency_ok']       else "✗"
        c = "✓" if ev['route_ok']            else "✗"
        name = prob.name[:32]
        print(f"  {name:<32} {a} {ev['input_sim']:>5.3f}  "
              f"{b} {ev['max_parent_sim']:>4.3f}/{ev['root_sim']:>4.3f}  "
              f"{c} {ev['path_mean']:>4.3f}/{ev['nonpath_mean']:>4.3f}")
        all_evals.append(ev)

    print("─" * 65)
    print("  (入力保存: cos(入力,出力))  "
          "(依存構造: 親sim/根sim)  (経路整合: 経路/非経路)")

    a_rate = np.mean([e['input_sim']    > 0.5 for e in all_evals])
    b_rate = np.mean([e['dependency_ok']       for e in all_evals])
    c_rate = np.mean([e['route_ok']            for e in all_evals])
    print(f"\n  達成率  入力保存={a_rate:.0%}  依存構造={b_rate:.0%}  経路整合={c_rate:.0%}")

    # ── 詳細分析: 一次方程式 ────────────────────────────
    print()
    print("=" * 65)
    print("【詳細分析】 2x + 3 = 7  — Message Passing 過程")
    print("=" * 65)
    solver.analyze_message_passing(problems[0])

    # ── 情報源追跡 ───────────────────────────────────────
    print("=" * 65)
    print("【情報源追跡】 答えノードはどのノードと最も類似しているか")
    print("=" * 65)
    print()
    prob = problems[0]
    ev   = evaluate_information_flow(solver, prob)
    fh   = ev['final_h']
    ans  = fh[prob.answer_node]

    G = nx.from_numpy_array(prob.adj.numpy())
    path_nodes = set(nx.shortest_path(G, 0, prob.answer_node))

    print(f"  正解経路: {sorted(path_nodes)}")
    print()
    print(f"  {'Node':<5} {'ラベル':>12}  {'cos_sim':>8}  {'経路?':>6}")
    print("  " + "─" * 38)
    sims = []
    for i, label in enumerate(prob.node_labels):
        if i == prob.answer_node:
            continue
        s = F.cosine_similarity(fh[i].unsqueeze(0), ans.unsqueeze(0)).item()
        on_path = "✓" if i in path_nodes else " "
        sims.append((i, label, s, on_path))

    for i, label, s, on_path in sorted(sims, key=lambda x: -x[2]):
        print(f"  {i:<5} {label:>12}  {s:>8.4f}  {on_path:>6}")

    print()
    path_avg    = np.mean([s for _, _, s, p in sims if p == "✓"])
    nonpath_avg = np.mean([s for _, _, s, p in sims if p == " "])
    print(f"  経路ノード平均類似度    : {path_avg:.4f}")
    print(f"  非経路ノード平均類似度  : {nonpath_avg:.4f}")
    result = "✓ 経路ノードが優先" if path_avg >= nonpath_avg else "✗ 非経路が混入"
    print(f"  判定: {result}")


# ─────────────────────────────────────────────
#  Task 2: 訓練データ生成
# ─────────────────────────────────────────────

def generate_training_data(n_samples: int = 500, seed: int = 42) -> list[MathProblemGraph]:
    """
    正解付き問題グラフを大量生成する。

    生成する問題タイプ:
      - 一次方程式 ax + b = c  (a in [1,9], b in [-9,9], c in [-20,20])
      - 二次方程式 x^2 + bx + c = 0  (判別式 D >= 0 のもの)
      - 連立方程式 (行列式 != 0 のもの)

    Returns: List[MathProblemGraph]
    """
    rng = random.Random(seed)
    problems: list[MathProblemGraph] = []

    # 一次方程式: 50%
    n_linear = n_samples // 2
    for _ in range(n_linear):
        a = rng.randint(1, 9)
        b = rng.randint(-9, 9)
        c = rng.randint(-20, 20)
        problems.append(build_linear_equation(float(a), float(b), float(c)))

    # 二次方程式: 25%
    n_quad = n_samples // 4
    count = 0
    while count < n_quad:
        a = 1.0
        b_val = rng.randint(-9, 9)
        c_val = rng.randint(-20, 20)
        D = b_val ** 2 - 4 * a * c_val
        if D < 0:
            continue
        problems.append(build_quadratic_equation(a, float(b_val), float(c_val)))
        count += 1

    # 連立方程式: 25%
    n_sim = n_samples - n_linear - n_quad
    count = 0
    while count < n_sim:
        a1 = rng.randint(-5, 5)
        b1 = rng.randint(-5, 5)
        c1 = rng.randint(-10, 10)
        a2 = rng.randint(-5, 5)
        b2 = rng.randint(-5, 5)
        c2 = rng.randint(-10, 10)
        det = a1 * b2 - a2 * b1
        if abs(det) < 1e-8:
            continue
        prob = build_simultaneous_equations(
            float(a1), float(b1), float(c1),
            float(a2), float(b2), float(c2),
        )
        if math.isnan(prob.answer_value):
            continue
        problems.append(prob)
        count += 1

    rng.shuffle(problems)
    return problems


# ─────────────────────────────────────────────
#  Task 2: 損失関数
# ─────────────────────────────────────────────

def answer_regression_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
) -> torch.Tensor:
    """MSE loss for answer value prediction."""
    return F.mse_loss(pred, true)


def path_contrastive_loss(
    final_h: torch.Tensor,
    problem: MathProblemGraph,
    margin: float = 0.1,
) -> torch.Tensor:
    """
    経路ノードの類似度 > 非経路ノードの類似度 + margin
    になるよう答えノードの表現を学習するコントラスト損失。

    LLM の Anchor 効果に対応する構造:
    「問題文（root）より直接の操作ノード（parent）が
     答えノードに強く影響する」状態を学習目標とする。
    """
    ans_idx = problem.answer_node
    ans_vec = final_h[ans_idx]  # (hidden_dim,)
    n_nodes = final_h.size(0)

    # 経路ノードを特定
    G = nx.from_numpy_array(problem.adj.numpy())
    try:
        path_nodes = set(nx.shortest_path(G, 0, ans_idx))
    except nx.NetworkXNoPath:
        path_nodes = {0, ans_idx}
    path_nodes.discard(ans_idx)

    if len(path_nodes) == 0 or n_nodes - len(path_nodes) - 1 <= 0:
        return torch.tensor(0.0, device=final_h.device, requires_grad=True)

    # 経路・非経路ノードの類似度を計算
    path_sims = []
    nonpath_sims = []
    for i in range(n_nodes):
        if i == ans_idx:
            continue
        sim = F.cosine_similarity(
            final_h[i].unsqueeze(0), ans_vec.unsqueeze(0)
        )
        if i in path_nodes:
            path_sims.append(sim)
        else:
            nonpath_sims.append(sim)

    if not path_sims or not nonpath_sims:
        return torch.tensor(0.0, device=final_h.device, requires_grad=True)

    path_mean = torch.stack(path_sims).mean()
    nonpath_mean = torch.stack(nonpath_sims).mean()

    # hinge loss: path_mean should exceed nonpath_mean + margin
    loss = F.relu(nonpath_mean - path_mean + margin)
    return loss


def dependency_loss(
    out: torch.Tensor,
    problem: MathProblemGraph,
    margin: float = 0.05,
) -> torch.Tensor:
    """
    答えノードに対して「直接の親 > ルート」を強制する損失。

    対象:
      - answer_node  : 評価対象ノード
      - direct_parent: adj[:, answer_node] で隣接するノード
      - root         : Node 0（問題全体のルート）

    損失:
      root が direct_parent でもある場合はスキップする。
      それ以外:
        loss = max(0, cos(answer, root) - cos(answer, parent) + margin)
        を全 parent について平均する。
    """
    ans_idx = problem.answer_node
    ans_vec = out[ans_idx]

    # 直接の親ノードを取得
    parents = problem.adj[:, ans_idx].nonzero(as_tuple=True)[0].tolist()
    # root が親に含まれる場合、この問題ではスキップ
    parents_no_root = [p for p in parents if p != 0 and p != ans_idx]

    if not parents_no_root:
        return torch.tensor(0.0, device=out.device, requires_grad=True)

    root_sim = F.cosine_similarity(
        out[0].unsqueeze(0), ans_vec.unsqueeze(0)
    )

    losses = []
    for p in parents_no_root:
        parent_sim = F.cosine_similarity(
            out[p].unsqueeze(0), ans_vec.unsqueeze(0)
        )
        # hinge: root_sim should be below parent_sim by at least margin
        losses.append(F.relu(root_sim - parent_sim + margin))

    return torch.stack(losses).mean()


# ─────────────────────────────────────────────
#  Task 2: 訓練ループ
# ─────────────────────────────────────────────

def train(
    solver: MathSolverGNN,
    problems: list[MathProblemGraph],
    n_epochs: int = 100,
    lr: float = 1e-3,
    lambda_contrastive: float = 0.5,
    lambda_dependency: float = 1.0,
) -> dict:
    """
    Adam optimizer で訓練する。

    10 epoch ごとに以下を記録・表示:
      - 回帰損失 (answer MSE)
      - コントラスト損失
      - 依存構造損失
      - 評価3軸スコア（入力保存・依存構造・経路整合）の平均

    Returns
    -------
    history : dict with keys 'epoch', 'reg_loss', 'con_loss', 'dep_loss', ...
    """
    optimizer = torch.optim.Adam(solver.parameters(), lr=lr)
    history: dict[str, list] = {
        'epoch': [],
        'reg_loss': [],
        'con_loss': [],
        'dep_loss': [],
        'total_loss': [],
        'input_sim': [],
        'dependency_ok': [],
        'route_ok': [],
    }

    solver.train()

    for epoch in range(1, n_epochs + 1):
        epoch_reg = 0.0
        epoch_con = 0.0
        epoch_dep = 0.0
        epoch_total = 0.0

        for prob in problems:
            optimizer.zero_grad()

            # Forward: GNN + decoder
            out = solver.gnn(prob.x, prob.adj, record_history=True)
            answer_feat = out[prob.answer_node]
            pred_value = solver.decoder(answer_feat)

            true_value = torch.tensor(
                prob.answer_value / SCALE, dtype=torch.float32
            )

            # 回帰損失
            reg_loss = answer_regression_loss(pred_value, true_value)

            # コントラスト損失 (output space — out は勾配あり)
            # 注: signal_history は .detach().clone() で保存されているため
            # 勾配が流れない。out を直接使う。
            con_loss = path_contrastive_loss(out, prob)

            # 依存構造損失 (parent > root)
            dep_loss = dependency_loss(out, prob)

            loss = (reg_loss
                    + lambda_contrastive * con_loss
                    + lambda_dependency * dep_loss)
            loss.backward()
            optimizer.step()

            epoch_reg += reg_loss.item()
            epoch_con += con_loss.item()
            epoch_dep += dep_loss.item()
            epoch_total += loss.item()

        n = len(problems)
        epoch_reg /= n
        epoch_con /= n
        epoch_dep /= n
        epoch_total /= n

        # 10 epoch ごとに評価
        if epoch % 10 == 0 or epoch == 1:
            solver.eval()
            eval_sample = problems[:min(20, len(problems))]
            input_sims = []
            dep_oks = []
            route_oks = []

            for prob in eval_sample:
                ev = evaluate_information_flow(solver, prob)
                input_sims.append(ev['input_sim'])
                dep_oks.append(float(ev['dependency_ok']))
                route_oks.append(float(ev['route_ok']))

            avg_input = np.mean(input_sims)
            avg_dep = np.mean(dep_oks)
            avg_route = np.mean(route_oks)

            history['epoch'].append(epoch)
            history['reg_loss'].append(epoch_reg)
            history['con_loss'].append(epoch_con)
            history['dep_loss'].append(epoch_dep)
            history['total_loss'].append(epoch_total)
            history['input_sim'].append(avg_input)
            history['dependency_ok'].append(avg_dep)
            history['route_ok'].append(avg_route)

            print(
                f"  Epoch {epoch:4d} | "
                f"Reg={epoch_reg:.4f}  Con={epoch_con:.4f}  "
                f"Dep={epoch_dep:.4f}  "
                f"Total={epoch_total:.4f} | "
                f"DepOK={avg_dep:.0%}  "
                f"RouteOK={avg_route:.0%}"
            )

            solver.train()

    return history


def run_training(
    n_samples: int = 200,
    n_epochs: int = 100,
    lr: float = 1e-3,
    lambda_contrastive: float = 0.5,
    lambda_dependency: float = 1.0,
    seed: int = 42,
    save_model: bool = True,
) -> tuple[MathSolverGNN, dict]:
    """
    訓練を実行して、モデルと履歴を返す便利関数。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print("=" * 65)
    print("GNN Stigmergy — 訓練ループ")
    print("=" * 65)
    print(f"  lambda_contrastive = {lambda_contrastive}")
    print(f"  lambda_dependency  = {lambda_dependency}")
    print()

    print(f"データ生成中 (n={n_samples})...")
    problems = generate_training_data(n_samples, seed)
    print(f"  生成完了: {len(problems)} 問題")

    # 訓練/テスト分割 (80/20)
    split = int(len(problems) * 0.8)
    train_problems = problems[:split]
    test_problems = problems[split:]
    print(f"  訓練: {len(train_problems)}, テスト: {len(test_problems)}")
    print()

    solver = MathSolverGNN(node_dim=NODE_DIM, hidden_dim=64, num_layers=4)

    print("訓練開始...")
    history = train(
        solver, train_problems,
        n_epochs=n_epochs, lr=lr,
        lambda_contrastive=lambda_contrastive,
        lambda_dependency=lambda_dependency,
    )
    print()

    # テスト評価
    solver.eval()
    print("テスト評価:")
    test_evals = []
    for prob in test_problems:
        ev = evaluate_information_flow(solver, prob)
        test_evals.append(ev)

    a_rate = np.mean([e['input_sim'] > 0.5 for e in test_evals])
    b_rate = np.mean([e['dependency_ok'] for e in test_evals])
    c_rate = np.mean([e['route_ok'] for e in test_evals])
    print(f"  入力保存={a_rate:.0%}  依存構造={b_rate:.0%}  経路整合={c_rate:.0%}")

    # モデル保存
    if save_model:
        save_path = 'trained_model.pt'
        torch.save({
            'model_state_dict': solver.state_dict(),
            'history': history,
            'n_samples': n_samples,
            'n_epochs': n_epochs,
            'lambda_contrastive': lambda_contrastive,
            'lambda_dependency': lambda_dependency,
        }, save_path)
        print(f"\n  モデルを保存しました: {save_path}")

    return solver, history


def run_sweep(
    lambdas_dep: list[float] = None,
    lambda_contrastive: float = 0.5,
    n_samples: int = 200,
    n_epochs: int = 100,
    lr: float = 1e-3,
    seed: int = 42,
):
    """
    lambda_dependency のスイープ実験。
    lambda_contrastive は固定し、lambda_dependency を変えて比較する。
    同じデータ・同じ乱数シードで各値を訓練・評価する。
    """
    if lambdas_dep is None:
        lambdas_dep = [0.5, 1.0, 2.0, 5.0]

    print("=" * 65)
    print("Lambda Dependency Sweep")
    print("=" * 65)
    print(f"  lambda_contrastive = {lambda_contrastive} (fixed)")
    print(f"  lambda_dependency  = {lambdas_dep}")
    print(f"  n_samples={n_samples}, n_epochs={n_epochs}, seed={seed}")
    print()

    # データを1回だけ生成（全値で共有）
    problems = generate_training_data(n_samples, seed)
    split = int(len(problems) * 0.8)
    train_problems = problems[:split]
    test_problems = problems[split:]
    print(f"  データ: {len(train_problems)} 訓練, {len(test_problems)} テスト")
    print()

    results = []

    for lam_dep in lambdas_dep:
        print("-" * 65)
        print(f"  lambda_dep = {lam_dep}")
        print("-" * 65)

        # 毎回モデルを初期化し直す（公平な比較）
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        solver = MathSolverGNN(node_dim=NODE_DIM, hidden_dim=64, num_layers=4)

        history = train(
            solver, train_problems,
            n_epochs=n_epochs, lr=lr,
            lambda_contrastive=lambda_contrastive,
            lambda_dependency=lam_dep,
        )

        # テスト評価
        solver.eval()
        test_evals = []
        for prob in test_problems:
            ev = evaluate_information_flow(solver, prob)
            test_evals.append(ev)

        final_reg = history['reg_loss'][-1]
        final_con = history['con_loss'][-1]
        final_dep = history['dep_loss'][-1]
        dep_rate = np.mean([e['dependency_ok'] for e in test_evals])
        route_rate = np.mean([e['route_ok'] for e in test_evals])

        results.append({
            'lambda_dep': lam_dep,
            'reg_loss': final_reg,
            'con_loss': final_con,
            'dep_loss': final_dep,
            'dep_ok': dep_rate,
            'route_ok': route_rate,
            'solver': solver,
            'history': history,
        })
        print()

    # 一覧表示
    print("=" * 65)
    print("Sweep Results  (lambda_contrastive="
          f"{lambda_contrastive} fixed)")
    print("=" * 65)
    print()
    for r in results:
        print(
            f"  lambda_dep={r['lambda_dep']:<4.1f} | "
            f"Reg={r['reg_loss']:.4f}  "
            f"Con={r['con_loss']:.4f}  "
            f"Dep={r['dep_loss']:.4f}  "
            f"DepOK={r['dep_ok']:.0%}  "
            f"RouteOK={r['route_ok']:.0%}"
        )
    print()

    # ベストを保存
    best = max(results, key=lambda r: r['dep_ok'] + r['route_ok'])
    save_path = 'trained_model.pt'
    torch.save({
        'model_state_dict': best['solver'].state_dict(),
        'history': best['history'],
        'n_samples': n_samples,
        'n_epochs': n_epochs,
        'lambda_contrastive': lambda_contrastive,
        'lambda_dependency': best['lambda_dep'],
    }, save_path)
    print(
        f"  Best: lambda_dep={best['lambda_dep']} "
        f"(DepOK={best['dep_ok']:.0%}, RouteOK={best['route_ok']:.0%})"
    )
    print(f"  Saved to {save_path}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="GNN Stigmergy Math Problem Solver")
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--sweep', action='store_true', help='Run lambda sweep experiment')
    parser.add_argument('--lambda_contrastive', type=float, default=0.5,
                        help='Contrastive loss weight (default: 0.5)')
    parser.add_argument('--lambda_dependency', type=float, default=1.0,
                        help='Dependency loss weight (default: 1.0)')
    parser.add_argument('--n_samples', type=int, default=200)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if args.sweep:
        run_sweep(
            lambda_contrastive=args.lambda_contrastive,
            n_samples=args.n_samples,
            n_epochs=args.n_epochs,
            lr=args.lr,
            seed=args.seed,
        )
    elif args.train:
        run_training(
            n_samples=args.n_samples,
            n_epochs=args.n_epochs,
            lr=args.lr,
            lambda_contrastive=args.lambda_contrastive,
            lambda_dependency=args.lambda_dependency,
            seed=args.seed,
        )
    else:
        run_math_experiments()
