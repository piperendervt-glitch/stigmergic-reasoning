# stigmergic-reasoning

**Stigmergy原則に基づくHypergraph Neural Networkによる推論システム**

エージェントが環境（フェロモン場）を介してのみ間接通信する
Stigmergyアーキテクチャで、GNNベースの推論システムにおける
構造的ボトルネックをStigmergy + HGNNで解消するPoC。

---

## 概要

GNNベースの推論システムで構造的に発生する
以下のボトルネックを測定・解消することを目的としたPoCです。

設計の動機はLLMのマルチエージェント設計で観測された
Anchor効果・Semantic Anchor・Groupthink問題にあります。
ただし現段階ではGNN単体での実装・検証に集中しています。

| ボトルネック | 解決手法 | 結果 |
|---|---|---|
| Attention Sink（ノルム優位性） | Sparse Attention + Gate機構 | Anchor Score 0.948 → 0.562 |
| Over-squashing（情報消失） | HGNN超エッジ設計 | leaf_norm 0.00 → 3.32 |
| Over-smoothing（表現均一化） | 超エッジサイズを2〜4ノードに制限 | OS開始層 >8 を達成 |
| 間接通信の欠如 | フェロモン場（蒸発・拡散・堆積） | エージェント間の直接通信なし |

---

## アーキテクチャ

```
[MathAgent × N]          ← 軽量エージェント（担当ノード1つ）
      ↕ 読み書き
[PheromoneEnvironment]   ← 動的フェロモン場
      ↕ 超エッジ拡散
[HyperedgeLayer]         ← HGNN（超エッジで直径短縮）
```

### 設計原則

- **Stigmergy**: エージェントは環境とのみ通信する（直接通信なし）
- **HGNN**: 問題の依存関係を超エッジで表現（Over-squashing解消）
- **Gate機構**: ゼロノードを集約から除外（波面伝播を実現）

---

## 既存アーキテクチャとの位置づけ

このプロジェクトは以下の3つの交差点にあります。

```
Neural Algorithmic Reasoning  ← 推論プロセスをグラフで学習
        ×
Hypergraph Neural Network     ← 高次依存関係を超エッジで表現
        ×
Stigmergy (MAS)               ← 環境介在型間接通信
```

| 比較対象 | 共通点 | 相違点 |
|---|---|---|
| Transformer/LLM | 設計の比較対象・動機 | 本リポジトリでは未使用 |
| 標準GNN (GCN/GAT) | グラフ上のMessage Passing | Over-squashing・静的エッジ |
| 標準HGNN | 超エッジによる高次関係 | 静的超エッジ・エージェントなし |
| Neural Operator | グラフ上の学習 | 関数写像が目的・Stigmergyなし |
| 従来MAS | マルチエージェント | 直接通信・動的環境なし |

---

## セットアップ

```bash
git clone https://github.com/piperendervt-glitch/stigmergic-reasoning
cd stigmergic-reasoning
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 使い方

```bash
# 問題を解く
python solvers/solve.py --eq "2x+3=7"
python solvers/solve.py --eq "x^2-5x+6=0"
python solvers/solve.py --sim "x+y=5,x-y=1"
python solvers/solve.py --interactive

# 訓練
python solvers/math_problem_gnn.py --train

# ボトルネック測定
python experiments/bottleneck_analysis.py

# 比較実験
python experiments/comparison_experiment.py

# テスト
python -m pytest tests/ -v
```

---

## 実験結果

### ボトルネック測定（HGNN導入後）

| タスク | leaf_norm | 回帰誤差 | OS開始層 | 判定 |
|---|---|---|---|---|
| 一次方程式 | 3.32 | 0.047 | >8 | PASS |
| 二次方程式 | 0.88 | 0.008 | >8 | PASS |
| 連立方程式 | 0.035 | 0.618 | >8 | 誤差改善中 |

### Anchor Score の変化

| アーキテクチャ | Anchor Score | DepOK | RouteOK |
|---|---|---|---|
| LLM-like | 1.000 | 0% | 50% |
| GNN v3 (Sparse+Gate) | 0.562 | 100% | 100% |
| HGNN | 測定予定 | 100% | 100% |

---

## ファイル構成

```
stigmergic-reasoning/
├── core/
│   ├── gnn_stigmergy.py      # Sparse Attention + Gate機構 (v3)
│   ├── root_layer_gnn.py     # フェロモン場（蒸発・拡散・堆積）
│   └── hgnn_solver.py        # HGNNソルバー（超エッジ設計）
├── solvers/
│   ├── math_problem_gnn.py   # 問題グラフ構築・訓練ループ
│   ├── multi_agent_solver.py # マルチエージェントソルバー
│   └── solve.py              # CLIインターフェース
├── experiments/
│   ├── comparison_experiment.py  # GNN vs Baseline比較
│   ├── bottleneck_analysis.py    # Over-squashing/smoothing測定
│   ├── squash_fix_comparison.py  # 解消手法の比較
│   └── eval_trained_model.py     # 訓練済みモデル評価
├── visualize/
│   └── visualize_gnn.py      # 波面アニメーション・可視化
└── tests/
    └── test_gnn_stigmergy.py # 28テスト
```

---

## 残課題

- [ ] 連立方程式の回帰誤差改善（n_epochs 200 → 500）
- [ ] HGNN導入後のAnchor Score測定
- [ ] HGNNのAnchor ScoreをLLM-likeと比較
- [ ] マルチエージェント設計の本格統合
- [ ] LLMエージェントとStigmergy環境層の統合（次フェーズ）
- [ ] LLMエージェントでのAnchor Score実測とGNNとの比較

---

## 開発経緯

このプロジェクトはLLMのStigmergy型マルチエージェント設計で
観測されたAnchor効果を構造的に回避することを動機として
開始しました。

現在はGNN単体でのStigmergy + HGNNアーキテクチャの
実装・検証フェーズにあります。
LLMエージェントとの統合は次フェーズの課題です。

GNN → Sparse Attention + Gate → HGNN という
段階的な改良を通じてボトルネックを順番に解消しました。
