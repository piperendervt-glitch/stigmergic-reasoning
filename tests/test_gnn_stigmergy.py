"""
pytest によるユニットテスト。

テスト項目:
  - StigmergyMessagePassing: nan が出ないこと
  - ゲート機構: norm=0 のノードで gate < 0.1 になること
  - 波面伝播: N層で最大Nホップまで活性化すること
  - _sparse_softmax: 全-inf グループで weight=0 になること
  - PheromoneField: 蒸発後に総濃度が減少すること
  - compute_anchor_effect_score: LLM-like > Baseline の順になること
  - MathProblemGraph: 各問題タイプでグラフが正しく構築されること
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import pytest

from core.gnn_stigmergy import (
    StigmergyMessagePassing,
    GNNStigmergyLayer,
    _sparse_softmax,
    build_test_graph,
)
from core.root_layer_gnn import PheromoneField, GNNDiffusionLayer, RootLayerGNN
from experiments.comparison_experiment import (
    compute_anchor_effect_score,
    compute_signal_monotonicity,
    BaselineMessagePassing,
    LLMLikeAnchor,
)
from solvers.math_problem_gnn import (
    build_linear_equation,
    build_quadratic_equation,
    build_simultaneous_equations,
    MathSolverGNN,
    generate_training_data,
    answer_regression_loss,
    path_contrastive_loss,
    NODE_DIM,
)


# ─────────────────────────────────────────────
#  StigmergyMessagePassing テスト
# ─────────────────────────────────────────────

class TestStigmergyMessagePassing:
    """StigmergyMessagePassing レイヤーのテスト。"""

    def test_no_nan_output(self):
        """出力に nan が含まれないこと。"""
        torch.manual_seed(42)
        layer = StigmergyMessagePassing(16, 16)
        adj, _ = build_test_graph(8, 'path')
        x = torch.zeros(8, 16)
        x[0] = torch.ones(16)

        out = layer(x, adj)

        assert not torch.isnan(out).any(), "Output contains NaN"
        assert not torch.isinf(out).any(), "Output contains Inf"

    def test_no_nan_all_zero_input(self):
        """全ノードがゼロの場合でも nan が出ないこと。"""
        torch.manual_seed(42)
        layer = StigmergyMessagePassing(16, 16)
        adj, _ = build_test_graph(8, 'path')
        x = torch.zeros(8, 16)

        out = layer(x, adj)

        assert not torch.isnan(out).any(), "Output contains NaN for all-zero input"

    def test_no_nan_random_input(self):
        """ランダム入力でも nan が出ないこと。"""
        torch.manual_seed(42)
        layer = StigmergyMessagePassing(16, 16)
        adj, _ = build_test_graph(8, 'path')
        x = torch.randn(8, 16)

        out = layer(x, adj)

        assert not torch.isnan(out).any()

    def test_output_shape(self):
        """出力の形状が正しいこと。"""
        layer = StigmergyMessagePassing(16, 32)
        adj, _ = build_test_graph(8, 'path')
        x = torch.randn(8, 16)

        out = layer(x, adj)

        assert out.shape == (8, 32)

    def test_no_edges_graph(self):
        """エッジのないグラフでも動作すること。"""
        layer = StigmergyMessagePassing(16, 16)
        adj = torch.zeros(4, 4)
        x = torch.randn(4, 16)

        out = layer(x, adj)

        assert out.shape == (4, 16)
        assert not torch.isnan(out).any()


# ─────────────────────────────────────────────
#  ゲート機構テスト
# ─────────────────────────────────────────────

class TestGateMechanism:
    """ゲート機構のテスト。"""

    def test_zero_node_gate_low(self):
        """norm=0 のノードで gate < 0.1 になること。"""
        layer = StigmergyMessagePassing(16, 16, gate_temp=10.0)
        x = torch.zeros(4, 16)
        # Node 0 はゼロ、Node 1-3 に信号あり
        x[1] = torch.ones(16)
        x[2] = torch.ones(16) * 0.5
        x[3] = torch.randn(16)

        gate = layer._compute_gate(x)  # (4, 1)

        assert gate[0].item() < 0.1, f"Zero node gate should be < 0.1, got {gate[0].item()}"

    def test_active_node_gate_high(self):
        """信号ありのノードで gate > 0.9 になること。"""
        layer = StigmergyMessagePassing(16, 16, gate_temp=10.0)
        x = torch.zeros(4, 16)
        x[1] = torch.ones(16)

        gate = layer._compute_gate(x)

        assert gate[1].item() > 0.9, f"Active node gate should be > 0.9, got {gate[1].item()}"

    def test_gate_shape(self):
        """ゲート出力の形状が (N, 1) であること。"""
        layer = StigmergyMessagePassing(16, 16)
        x = torch.randn(5, 16)
        gate = layer._compute_gate(x)
        assert gate.shape == (5, 1)


# ─────────────────────────────────────────────
#  波面伝播テスト
# ─────────────────────────────────────────────

class TestWavePropagation:
    """波面伝播のテスト。"""

    def test_wave_front_expansion(self):
        """N層で最大Nホップまで活性化すること（ゲート波面）。"""
        torch.manual_seed(42)
        n_nodes = 8
        node_dim = 16

        adj, _ = build_test_graph(n_nodes, 'path')
        x = torch.zeros(n_nodes, node_dim)
        x[0] = torch.ones(node_dim)

        model = GNNStigmergyLayer(
            node_dim=node_dim,
            hidden_dim=32,
            num_layers=4,
            dropout=0.0,
        )
        model.eval()

        with torch.no_grad():
            model(x, adj, record_history=True)

        # Layer 0 (input_proj直後): Node 0 のみ活性
        h0 = model.signal_history[0]
        norms0 = h0.norm(dim=1)
        assert norms0[0].item() > 0.1, "Node 0 should be active at layer 0"

        # 各層で活性ノード数が単調に増加（少なくとも減少しない）
        prev_active = 0
        for h in model.signal_history:
            norms = h.norm(dim=1)
            active = (norms > 0.1).sum().item()
            assert active >= prev_active, "Active nodes should not decrease"
            prev_active = active


# ─────────────────────────────────────────────
#  _sparse_softmax テスト
# ─────────────────────────────────────────────

class TestSparseSoftmax:
    """_sparse_softmax のテスト。"""

    def test_all_inf_group_zero_weight(self):
        """全-inf グループで weight=0 になること。"""
        scores = torch.tensor([float('-inf'), float('-inf'), 1.0, 2.0])
        dst = torch.tensor([0, 0, 1, 1])
        N = 2

        weights = _sparse_softmax(scores, dst, N)

        # グループ0 (dst=0) は全 -inf → weight=0
        assert weights[0].item() == pytest.approx(0.0, abs=1e-6)
        assert weights[1].item() == pytest.approx(0.0, abs=1e-6)

        # グループ1 (dst=1) は正常
        assert weights[2].item() > 0
        assert weights[3].item() > 0

    def test_softmax_sums_to_one(self):
        """各グループの重み合計が1になること。"""
        scores = torch.tensor([1.0, 2.0, 3.0, 0.5, 1.5])
        dst = torch.tensor([0, 0, 0, 1, 1])
        N = 2

        weights = _sparse_softmax(scores, dst, N)

        group0_sum = weights[:3].sum().item()
        group1_sum = weights[3:].sum().item()
        assert group0_sum == pytest.approx(1.0, abs=1e-5)
        assert group1_sum == pytest.approx(1.0, abs=1e-5)

    def test_no_nan_output(self):
        """出力に nan がないこと。"""
        scores = torch.tensor([float('-inf'), 1.0, float('-inf')])
        dst = torch.tensor([0, 1, 1])
        N = 2

        weights = _sparse_softmax(scores, dst, N)
        assert not torch.isnan(weights).any()


# ─────────────────────────────────────────────
#  PheromoneField テスト
# ─────────────────────────────────────────────

class TestPheromoneField:
    """PheromoneField のテスト。"""

    def test_evaporation_decreases_concentration(self):
        """蒸発後に総濃度が減少すること。"""
        field = PheromoneField(n_nodes=4, pheromone_dim=4, evaporation=0.2)
        deposit = torch.ones(1, 4) * 5.0
        field.deposit([0], deposit)

        before = field.total_concentration()
        field.evaporate()
        after = field.total_concentration()

        assert after < before, f"Concentration should decrease: {before} -> {after}"

    def test_evaporation_rate(self):
        """蒸発率が正しく適用されること。"""
        evap = 0.3
        field = PheromoneField(n_nodes=2, pheromone_dim=2, evaporation=evap)
        deposit = torch.ones(1, 2) * 10.0
        field.deposit([0], deposit)

        before = field.total_concentration()
        field.evaporate()
        after = field.total_concentration()

        expected = before * (1.0 - evap)
        assert after == pytest.approx(expected, rel=1e-5)

    def test_deposit_and_read(self):
        """堆積と読み取りが正しく動作すること。"""
        field = PheromoneField(n_nodes=4, pheromone_dim=2)
        deposit = torch.tensor([[3.0, 5.0]])
        field.deposit([1], deposit)

        read = field.read(1)
        assert read[0].item() == pytest.approx(3.0, abs=1e-5)
        assert read[1].item() == pytest.approx(5.0, abs=1e-5)

    def test_initial_concentration_zero(self):
        """初期状態の総濃度がゼロであること。"""
        field = PheromoneField(n_nodes=4, pheromone_dim=4)
        assert field.total_concentration() == pytest.approx(0.0, abs=1e-8)


# ─────────────────────────────────────────────
#  Anchor Effect Score テスト
# ─────────────────────────────────────────────

class TestAnchorEffectScore:
    """compute_anchor_effect_score のテスト。"""

    def test_llm_like_higher_than_gnn(self):
        """LLM-like Anchor > GNN Stigmergy の順になること。"""
        torch.manual_seed(42)
        n_nodes = 10
        node_dim = 16
        adj, G = build_test_graph(n_nodes, 'path')
        x = torch.zeros(n_nodes, node_dim)
        x[0] = torch.ones(node_dim)

        gnn = GNNStigmergyLayer(node_dim, 32, 3, dropout=0.0)
        llm = LLMLikeAnchor(node_dim, 3)

        gnn.eval()
        llm.eval()
        with torch.no_grad():
            out_gnn = gnn(x, adj)
            out_llm = llm(x, adj)

        score_gnn = compute_anchor_effect_score(out_gnn, G)
        score_llm = compute_anchor_effect_score(out_llm, G)

        assert score_llm >= score_gnn, (
            f"LLM-like ({score_llm:.4f}) should have higher anchor than GNN ({score_gnn:.4f})"
        )

    def test_score_range(self):
        """スコアが [0, 1] の範囲内であること。"""
        torch.manual_seed(42)
        adj, G = build_test_graph(8, 'path')
        x = torch.zeros(8, 16)
        x[0] = torch.ones(16)

        model = GNNStigmergyLayer(16, 32, 3, dropout=0.0)
        model.eval()
        with torch.no_grad():
            out = model(x, adj)

        score = compute_anchor_effect_score(out, G)
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"


# ─────────────────────────────────────────────
#  MathProblemGraph テスト
# ─────────────────────────────────────────────

class TestMathProblemGraph:
    """各問題タイプでグラフが正しく構築されること。"""

    def test_linear_equation(self):
        """一次方程式のグラフが正しいこと。"""
        prob = build_linear_equation(2.0, 3.0, 7.0)

        assert prob.answer_value == pytest.approx(2.0, abs=1e-5)
        assert prob.x.shape == (7, NODE_DIM)
        assert prob.adj.shape == (7, 7)
        assert prob.answer_node == 6
        assert prob.adj.sum() > 0  # エッジが存在する

    def test_quadratic_equation(self):
        """二次方程式のグラフが正しいこと。"""
        prob = build_quadratic_equation(1.0, -5.0, 6.0)

        assert prob.answer_value == pytest.approx(3.0, abs=1e-5)
        assert prob.x.shape == (5, NODE_DIM)
        assert prob.adj.shape == (5, 5)
        assert prob.answer_node == 4

    def test_simultaneous_equations(self):
        """連立方程式のグラフが正しいこと。"""
        prob = build_simultaneous_equations(1.0, 1.0, 5.0, 1.0, -1.0, 1.0)

        assert prob.answer_value == pytest.approx(3.0, abs=1e-5)
        assert prob.x.shape == (7, NODE_DIM)
        assert prob.adj.shape == (7, 7)
        assert prob.answer_node == 6

    def test_adjacency_symmetric(self):
        """隣接行列が対称であること。"""
        prob = build_linear_equation(3.0, 1.0, 10.0)
        assert torch.allclose(prob.adj, prob.adj.T)

    def test_quadratic_no_real_roots(self):
        """判別式 < 0 の場合でもエラーにならないこと。"""
        prob = build_quadratic_equation(1.0, 0.0, 1.0)
        assert prob.x.shape[0] == 5  # ノード数は同じ


# ─────────────────────────────────────────────
#  訓練関連テスト
# ─────────────────────────────────────────────

class TestTraining:
    """訓練機能のテスト。"""

    def test_generate_training_data(self):
        """データ生成が正しい数を返すこと。"""
        problems = generate_training_data(n_samples=20, seed=42)
        assert len(problems) == 20

    def test_generate_training_data_no_nan(self):
        """生成されたデータに nan answer がないこと。"""
        import math
        problems = generate_training_data(n_samples=50, seed=42)
        for prob in problems:
            assert not math.isnan(prob.answer_value), (
                f"NaN answer in {prob.name}"
            )

    def test_regression_loss(self):
        """回帰損失が計算できること。"""
        pred = torch.tensor(2.0, requires_grad=True)
        true = torch.tensor(3.0)
        loss = answer_regression_loss(pred, true)
        assert loss.item() == pytest.approx(1.0, abs=1e-5)

    def test_contrastive_loss_runs(self):
        """コントラスト損失がエラーなく計算できること。"""
        torch.manual_seed(42)
        prob = build_linear_equation(2.0, 3.0, 7.0)
        solver = MathSolverGNN(node_dim=NODE_DIM, hidden_dim=64, num_layers=4)

        out = solver.gnn(prob.x, prob.adj, record_history=True)
        final_h = solver.gnn.signal_history[-1]

        loss = path_contrastive_loss(final_h, prob)
        assert not torch.isnan(loss), "Contrastive loss is NaN"
        assert loss.item() >= 0.0, "Contrastive loss should be non-negative"

    def test_solver_forward(self):
        """MathSolverGNN の forward が正常に動作すること。"""
        torch.manual_seed(42)
        solver = MathSolverGNN(node_dim=NODE_DIM, hidden_dim=64, num_layers=4)
        prob = build_linear_equation(2.0, 3.0, 7.0)

        result = solver(prob)

        assert 'predicted_value' in result
        assert 'direct_read' in result
        assert 'true_answer' in result
        assert result['true_answer'] == pytest.approx(2.0, abs=1e-5)
