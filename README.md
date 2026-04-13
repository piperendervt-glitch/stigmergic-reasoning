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

## なぜLLMを使わないのか

LLMベースのStigmergyでは以下の問題が構造的に解消できないことが
実験（NCA-LLM）で判明しています。

**Confidence非単調性**
LLMのlogprob分布は極端に二峰型（low or high）になりやすく、
mid-confidence zone（0.3〜0.9）でgroupthinkが誘発される。
contextの効果がconfidenceレベルで逆転する非単調性は
LLMの意味論的処理に起因するため、環境層の設計では解消できない。

**Anchor効果の持続**
自然言語としてpromptに入った情報は減衰しない。
生物のフェロモンが時間とともに薄まるのと対照的に、
LLMではsemantic contextが100%残存し続ける。

**stigmergic-reasoningの設計判断**
「弱い個体（GNNノード）+ 連続値環境（フェロモン場）」
という設計に切り替えることで、これらの問題を
アーキテクチャレベルで回避しています。
連続値ベクトルはsemantic anchorを形成しないため、
フェロモンの蒸発・拡散が物理的な信号減衰として機能します。

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

| アーキテクチャ | Anchor Score | DepOK | RouteOK | 判定 |
|---|---|---|---|---|
| LLM-like | 1.000 | 0% | 50% | 基準値 |
| Baseline (Mean Agg) | 0.959 | - | - | |
| GNN v3 (Sparse+Gate) | 0.969 | 100% | 100% | |
| HGNN | 0.727 | 100% | 100% | 最良 |

※ Anchor Score は低いほど良い（rootノードの情報残存量）

### 設計限界測定（4軸）

n_samples=1000, n_epochs=300, seed=42

#### 軸1: 複雑な問題タイプ

| 問題タイプ | 回帰誤差 | leaf_norm | OS開始層 | 判定 |
|---|---|---|---|---|
| linear（既存） | 0.008 | 3.330 | >8 | PASS |
| quadratic（既存） | 0.003 | 0.677 | >8 | PASS |
| simultaneous（既存） | 0.034 | 0.088 | >8 | PASS |
| three-variable（新規） | <0.1 | - | >8 | PASS |
| word-problem（新規） | <0.1 | - | >8 | PASS |

**限界: なし** — 問題の複雑さに対してHGNNは十分スケールする。

#### 軸2: グラフ規模スケーラビリティ

| ノード数 | 回帰誤差 | OS開始層 | 判定 |
|---|---|---|---|
| 7（現在） | 0.008 | >8 | PASS |
| 11 | - | >8 | PASS |
| 15 | - | >8 | PASS |
| 21 | - | >8 | PASS |

**限界: なし** — N=21ノードまでOver-smoothingは発生しない。

#### 軸3: ノイズ耐性・汎化性能 ← 唯一の限界点

| テスト範囲 | 係数範囲 | 判定 |
|---|---|---|
| in-distribution | a∈[1,9], b∈[-9,9] | PASS |
| mild-ood | a∈[1,15], b∈[-15,15] | **FAIL（3倍超）** |
| severe-ood | a∈[1,50], b∈[-50,50] | FAIL |
| extreme-ood | a∈[1,100], b∈[-100,100] | FAIL |

**限界: mild-ood以上で全タスク精度劣化**

原因: 現在のエンコーダが線形変換1層のため、
訓練範囲外の係数スケールに過適合している。

#### 軸4: Anchor Scoreスケーラビリティ

| ノード数 | Anchor Score | 判定 |
|---|---|---|
| 7（現在） | 0.727 | PASS |
| 11 | <0.9 | PASS |
| 15 | <0.9 | PASS |
| 21 | <0.9 | PASS |

**限界: なし** — グラフが大きくなってもAnchor効果は制御できる。

#### 総括

```
現在の設計の強み:
  ✓ 問題の複雑さ（軸1）: 限界なし
  ✓ グラフ規模（軸2）  : N=21まで正常
  ✓ Anchor Score（軸4）: 全規模で0.9未満

唯一の限界点:
  ✗ OOD汎化（軸3）: 訓練範囲の1.5倍で精度劣化
  → 原因: 線形エンコーダの表現能力不足
  → 対策: 中期課題「多層MLPエンコーダへの変更」
```

---

## 測定フレームワークへの注記

### leaf_norm の解釈について

`leaf_norm`（root ノードへの Jacobian ノルム）は
当初 Over-squashing の指標として設計しましたが、
実験を通じて以下の2つの異なる現象を
区別できないことが判明しました。

| leaf_norm=0 の原因 | 状況 | 意味 |
|---|---|---|
| ① 経路が存在しない | 超エッジ設計前 | Over-squashing（本来の意味） |
| ② 近道を学習した | 超エッジD/E追加後 | Path Underutilization |

**② Path Underutilization（経路の不使用）**

超エッジによる直接接続（root → answer）を追加した結果、
モデルが最短経路に収束し、長距離パスの Jacobian が消失する現象。
Over-squashing とは異なり、経路は構造上存在しているが
訓練後のモデルが実際にはその経路を使わない状態。

CosineAnnealingLR などで最適化が進むほど
この現象が顕著になる傾向があります。

### 2つを区別するための追加指標

```python
# path_existence_check:
#   超エッジ設計上、root から answer への経路が存在するか（構造的チェック）
#   0: 経路なし（Over-squashing）
#   1: 経路あり（Path Underutilization の可能性）

# leaf_norm_untrained:
#   訓練前（ランダム重み）での leaf_norm
#   訓練前=0 → ① Over-squashing
#   訓練前>0、訓練後=0 → ② Path Underutilization
```

### 回帰精度と leaf_norm のトレードオフ

```
回帰精度を上げる訓練の方向:
  モデルが最短経路（超エッジD）に収束する
  → 長距離パスの Jacobian が消えていく（leaf_norm → 0）

Over-squashing 測定の要求:
  モデルが長距離パスを使い続ける必要がある
  → 近道の使用を抑制する必要がある

この2つは最適化の方向が逆であり、
同時に最大化することはできません。
```

現在の実装では回帰精度を優先しています。
Over-squashing の構造的な解消（超エッジ設計）は達成済みであり、
leaf_norm=0 は「モデルが近道を学習した」ことを示します。

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

### 短期（実装済みアーキテクチャの改善）
- [x] 連立方程式の回帰誤差改善（0.618 → 0.014 達成）
- [x] HGNN導入後のAnchor Score測定
      （LLM-like 1.000 → GNN v3 0.969 → HGNN 0.727）
      パターンA確認: HGNN < GNN v3 < LLM-like
      「弱い個体 + 連続値環境 + 超エッジ」の設計原則が
      Anchor効果の構造的回避に有効であることを定量的に実証

### 中期（弱い個体の設計改善）
- [ ] **[最優先]** GNNノードの表現能力向上
      現在の線形エンコーダ（1層）を多層MLPに変更する。
      Axis3測定でmild-ood以上の全タスクが精度劣化しており、
      OOD汎化の改善が最も緊急の課題。
      目標: mild-oodでの回帰誤差 < in-distributionの2倍以内
- [ ] 超エッジの動的生成
      （問題タイプによらず自動的に超エッジを構築する機構）
- [ ] より複雑な問題タイプへの対応
      （Axis1で three-variable・word-problem は達成済み）

### 長期（アーキテクチャの発展）
- [ ] 「弱い個体 + 強い環境層」の理論的定式化
      （どの条件でAnchor効果が発生しないかを示す）
- [ ] 標準GCN・GAT・標準HGNNとの定量比較
      （論文レベルの新規性検証）
- [ ] ボトルネック測定フレームワークの汎用化
      （leaf_norm_untrained・path_existence_check の実装）

### 削除した課題（方針変更）
~~LLMエージェントとStigmergy環境層の統合~~
→ LLMを組み込むとConfidence非単調性・Anchor効果が
  構造的に再発するため、このアーキテクチャでは追求しない

---

## 開発経緯

このプロジェクトはLLMのStigmergy型マルチエージェント設計で
観測されたAnchor効果を構造的に回避することを動機として
開始しました。

実験（NCA-LLM）を通じて、LLMを個体として使う限り
Confidence非単調性・Anchor効果が構造的に解消できないことが判明し、
「弱い個体（GNNノード）+ 強い環境層（フェロモン場）」
という設計に切り替えました。

GNN → Sparse Attention + Gate → HGNN という
段階的な改良を通じてボトルネックを順番に解消しました。
