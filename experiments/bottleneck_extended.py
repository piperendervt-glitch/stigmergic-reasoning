"""
bottleneck_extended.py
======================
HGNN 設計限界を4軸で測定する。

軸1: より複雑な問題タイプへの対応
軸2: グラフ規模のスケーラビリティ
軸3: ノイズ耐性・汎化性能
軸4: Anchor Score のスケーラビリティ
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import time
import random
import math
import logging
from pathlib import Path
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx


def setup_logger(log_file: str = "logs/experiment_progress.log"):
    """
    ファイルとコンソールの両方に出力するロガーを設定する。
    SSH接続時に tail -f で確認できるようにする。
    """
    Path("logs").mkdir(exist_ok=True)

    logger = logging.getLogger("experiment")
    logger.setLevel(logging.INFO)

    # ファイル出力（追記モード）
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)

    # コンソール出力
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "[%(asctime)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = logging.getLogger("experiment")

from solvers.math_problem_gnn import (
    MathProblemGraph, make_node,
    build_linear_equation, build_quadratic_equation,
    build_simultaneous_equations,
    NODE_DIM, SCALE,
)
from core.hgnn_solver import (
    train_hgnn, _gen_task_data, get_hyperedges,
    HGNNMathSolver, HGNNStigmergyLayer, HypergraphBuilder,
    measure_os_start_hgnn,
)
from experiments.squash_fix_comparison import measure_leaf_jacobian
from experiments.comparison_experiment import (
    compute_anchor_effect_score, compute_signal_monotonicity,
)


# ─────────────────────────────────────────────
#  新規問題タイプ
# ────────────────────────��────────────────────

def build_three_variable_equations(
    a1, b1, c1, d1,
    a2, b2, c2, d2,
    a3, b3, c3, d3,
) -> tuple[MathProblemGraph, list[list[int]]]:
    """
    三元連立方程式 a1x+b1y+c1z=d1, a2x+b2y+c2z=d2, a3x+b3y+c3z=d3。

    ノード構成:
      0: root, 1: eq1, 2: eq2, 3: eq3,
      4: 消去1(z消去), 5: 消去2(y消去),
      6: x解, 7: y解, 8: z解, 9: answer
    """
    A = np.array([[a1,b1,c1],[a2,b2,c2],[a3,b3,c3]], dtype=float)
    b_vec = np.array([d1,d2,d3], dtype=float)
    det = np.linalg.det(A)
    if abs(det) < 1e-8:
        x_ans = y_ans = z_ans = 0.0
    else:
        sol = np.linalg.solve(A, b_vec)
        x_ans, y_ans, z_ans = float(sol[0]), float(sol[1]), float(sol[2])

    nodes = [
        make_node('problem_root', coeff_a=a1, coeff_b=b1,
                  variables=['x', 'y', 'multi'], step_type='simplify'),
        make_node('expression', coeff_a=a1, coeff_b=b1, rhs=d1, variables=['x', 'y']),
        make_node('expression', coeff_a=a2, coeff_b=b2, rhs=d2, variables=['x', 'y']),
        make_node('expression', coeff_a=a3, coeff_b=b3, rhs=d3, variables=['x', 'y']),
        make_node('operation', ops=['sub'], step_type='substitute'),
        make_node('operation', ops=['sub'], step_type='isolate'),
        make_node('expression', answer=x_ans, variables=['x'], step_type='isolate'),
        make_node('expression', answer=y_ans, variables=['y'], step_type='isolate'),
        make_node('expression', answer=z_ans, variables=['x'], step_type='isolate'),
        make_node('answer', coeff_a=x_ans, coeff_b=y_ans,
                  variables=['x', 'y', 'multi']),
    ]

    edges = [(0,1),(0,2),(0,3),(1,4),(2,4),(3,4),(4,5),(5,6),(5,7),(6,9),(7,9),(7,8),(8,9)]
    N = len(nodes)
    adj = torch.zeros(N, N, dtype=torch.float32)
    for s, d in edges:
        adj[s][d] = adj[d][s] = 1.0

    labels = [
        "三元連立", f"{a1}x+{b1}y+{c1}z={d1}",
        f"{a2}x+{b2}y+{c2}z={d2}", f"{a3}x+{b3}y+{c3}z={d3}",
        "z消去", "y消去",
        f"x={x_ans:.2f}", f"y={y_ans:.2f}", f"z={z_ans:.2f}", "答え",
    ]

    hyperedges = [
        [0, 4, 9],        # root -> 消去 -> answer
        [1, 2, 3, 4],     # 3式 -> 消去ノード
        [4, 5, 6],        # 消去 -> 変数分離 -> x
        [5, 7, 8, 9],     # 変数分離 -> y, z -> answer
        [0, 9],           # root -> answer 直接補強
    ]

    prob = MathProblemGraph(
        name="三元連立方程式",
        problem_str=f"{a1}x+{b1}y+{c1}z={d1} / {a2}x+{b2}y+{c2}z={d2} / {a3}x+{b3}y+{c3}z={d3}",
        x=torch.stack(nodes).float(), adj=adj,
        answer_node=9, answer_value=x_ans,
        node_labels=labels,
    )
    return prob, hyperedges


def build_word_problem(multiplier: float, total: float) -> tuple[MathProblemGraph, list[list[int]]]:
    """
    二段階文章題: AはBのk倍、A+B=total のとき Aは？

    ノード構成:
      0: root, 1: 関係式(A=kB), 2: 合計式(A+B=total),
      3: 代入操作, 4: 中間式, 5: answer
    """
    b_ans = total / (multiplier + 1)
    a_ans = multiplier * b_ans

    nodes = [
        make_node('problem_root', coeff_a=multiplier, rhs=total,
                  variables=['x', 'y'], step_type='simplify'),
        make_node('expression', coeff_a=multiplier, variables=['x', 'y']),
        make_node('expression', rhs=total, variables=['x', 'y']),
        make_node('operation', ops=['sub'], step_type='substitute'),
        make_node('expression', coeff_a=multiplier + 1, rhs=total,
                  variables=['x'], step_type='isolate'),
        make_node('answer', answer=a_ans, variables=['x']),
    ]

    edges = [(0,1),(0,2),(1,3),(2,3),(3,4),(4,5)]
    N = len(nodes)
    adj = torch.zeros(N, N, dtype=torch.float32)
    for s, d in edges:
        adj[s][d] = adj[d][s] = 1.0

    labels = [
        "文章題", f"A={multiplier}B", f"A+B={total}",
        "代入", f"{multiplier+1}B={total}", f"A={a_ans:.2f}",
    ]

    hyperedges = [
        [0, 1, 2, 3],  # root + 2式 + 代入
        [1, 2, 3],      # 2式 -> 代入
        [3, 4, 5],      # 代入 -> 中間 -> answer
        [0, 5],          # root -> answer 直接補強
    ]

    prob = MathProblemGraph(
        name="文章題",
        problem_str=f"A={multiplier}B, A+B={total}",
        x=torch.stack(nodes).float(), adj=adj,
        answer_node=5, answer_value=a_ans,
        node_labels=labels,
    )
    return prob, hyperedges


def build_extended_linear(a: float, b: float, c: float, n_nodes: int = 7):
    """
    一次方程式のノード数を拡張する。
    n_nodes=7: 標準（既存）
    n_nodes>7: 中間ステップを追加して解法を細分化する。

    Returns: (MathProblemGraph, hyperedges)
    """
    if n_nodes == 7:
        prob = build_linear_equation(a, b, c)
        return prob, get_hyperedges(prob)

    true_answer = (c - b) / a

    # 中間ノードを追加して経路を延長する
    # 基本: root(0) -> expr(1) -> expr(2) -> op(3) -> ...中間... -> answer(N-1)
    nodes = [
        make_node('problem_root', coeff_a=a, const=b, rhs=c,
                  variables=['x'], step_type='simplify'),
        make_node('expression', coeff_a=a, const=b, variables=['x']),
        make_node('expression', rhs=c),
        make_node('operation', const=b, ops=['sub'], step_type='isolate'),
    ]
    labels = [f"{a}x+{b}={c}", f"{a}x+{b}", f"{c}", f"-{b}"]

    # 中間ノードを追加
    n_mid = n_nodes - 5  # 標準の5ノード(root,expr,expr,op,answer) + 中間
    intermediate_val = c - b
    step = (true_answer - intermediate_val / a) / max(n_mid, 1)

    for i in range(n_mid):
        frac = (i + 1) / (n_mid + 1)
        val = intermediate_val * (1 - frac) + true_answer * frac
        nodes.append(
            make_node('expression', coeff_a=a * (1 - frac), rhs=val,
                      variables=['x'], step_type='isolate')
        )
        labels.append(f"mid{i}")

    # 最後の割り算ノードと答えノード
    nodes.append(make_node('operation', coeff_a=a, ops=['div'], step_type='isolate'))
    labels.append(f"÷{a}")
    nodes.append(make_node('answer', answer=true_answer, variables=['x']))
    labels.append(f"x={true_answer:.2f}")

    N = len(nodes)
    # 直線チェーンのエッジ
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    prev = 3
    for i in range(4, N):
        edges.append((prev, i))
        prev = i

    adj = torch.zeros(N, N, dtype=torch.float32)
    for s, d in edges:
        adj[s][d] = adj[d][s] = 1.0

    # 超エッジ: root-answer直接 + 前半 + 後半 + root-中間
    mid_point = N // 2
    hyperedges = [
        [0, N - 1],                      # root -> answer
        list(range(0, mid_point + 1)),    # 前半
        list(range(mid_point, N)),        # 後半
        [0, mid_point],                   # root -> 中間
    ]

    prob = MathProblemGraph(
        name=f"一次方程式({N}ノード)",
        problem_str=f"{a}x + {b} = {c}",
        x=torch.stack(nodes).float(), adj=adj,
        answer_node=N - 1, answer_value=true_answer,
        node_labels=labels,
    )
    return prob, hyperedges


# ─────────────────────────────────────────────
#  カスタム超エッジ対応の訓練・測定
# ─────────────────────────────────────────────

def train_hgnn_custom(train_problems, hyperedges_fn, n_epochs=500, lr=1e-3, seed=42,
                      task_name=""):
    """超エッジを外部から渡せる版の train_hgnn。"""
    from solvers.math_problem_gnn import (
        path_contrastive_loss, dependency_loss,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    solver = HGNNMathSolver()
    optimizer = torch.optim.Adam(solver.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-5,
    )

    log_interval = max(1, n_epochs // 6)
    solver.train()
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        for prob in train_problems:
            optimizer.zero_grad()
            hyperedges = hyperedges_fn(prob)
            out = solver.hgnn(prob.x, hyperedges)
            pred = solver.decoder(out[prob.answer_node])
            true_val = torch.tensor(prob.answer_value / SCALE, dtype=torch.float32)
            reg = F.mse_loss(pred, true_val)
            con = path_contrastive_loss(out, prob)
            dep = dependency_loss(out, prob)
            loss = reg + 0.5 * con + 1.0 * dep
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        if task_name and epoch % log_interval == 0:
            avg_loss = epoch_loss / len(train_problems)
            logger.info(f"  {task_name}: training epoch {epoch}/{n_epochs} "
                        f"loss={avg_loss:.4f}")

    solver.eval()
    return solver


def measure_single(solver, test_prob, hyperedges, seed=42):
    """単一問題の回帰誤差・leaf_norm・OS開始層を測定する。"""
    # 回帰誤差
    solver.eval()
    with torch.no_grad():
        out = solver.hgnn(test_prob.x, hyperedges)
    pred = float(solver.decoder(out[test_prob.answer_node]).item()) * SCALE
    error = abs(pred - test_prob.answer_value)

    # leaf_norm
    def fwd(x, prob, _s=solver, _he=hyperedges):
        return _s.hgnn(x, _he)[prob.answer_node]
    leaf_raw, _ = measure_leaf_jacobian(fwd, test_prob)

    # OS開始層 (カスタム超エッジ対応)
    for n_layers in range(1, 9):
        torch.manual_seed(seed)
        model = HGNNMathSolver(num_layers=n_layers)
        model.eval()
        with torch.no_grad():
            model.hgnn(test_prob.x, hyperedges, record_history=True)
        final_h = model.hgnn.signal_history[-1]
        N = final_h.size(0)
        norms = final_h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normed = final_h / norms
        sim_matrix = normed @ normed.T
        mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
        score = sim_matrix[mask].mean().item()
        if score > 0.9:
            os_start = n_layers
            break
    else:
        os_start = 9  # >8

    return error, leaf_raw, os_start


# ─────────────────────────────────────────────
#  軸1: 複雑な問題タイプ
# ─────────────────────────────────────────────

def gen_three_var_data(n, seed=42):
    """三元連立方程式の訓練データを生成する。"""
    rng = random.Random(seed)
    problems, he_list = [], []
    count = 0
    while count < n:
        coeffs = [rng.randint(-3, 3) for _ in range(12)]
        A = np.array([[coeffs[0],coeffs[1],coeffs[2]],
                       [coeffs[4],coeffs[5],coeffs[6]],
                       [coeffs[8],coeffs[9],coeffs[10]]], dtype=float)
        if abs(np.linalg.det(A)) < 1e-8:
            continue
        prob, he = build_three_variable_equations(*coeffs)
        if math.isnan(prob.answer_value):
            continue
        problems.append(prob)
        he_list.append(he)
        count += 1
    return problems, he_list


def gen_word_problem_data(n, seed=42):
    """文章題の訓練データを生成する。"""
    rng = random.Random(seed)
    problems, he_list = [], []
    for _ in range(n):
        mult = rng.randint(2, 9)
        total = rng.randint(10, 100)
        prob, he = build_word_problem(float(mult), float(total))
        problems.append(prob)
        he_list.append(he)
    return problems, he_list


def run_axis1(n_samples=1000, n_epochs=300, seed=42):
    """軸1: 複雑な問題タイプの測定。"""
    logger.info("=" * 50)
    logger.info("Axis1 start: complex problem types")
    logger.info("=" * 50)

    results = {}
    threshold = 0.1

    # existing tasks (reference)
    for task_name, test_prob in [
        ('linear', build_linear_equation(2, 3, 7)),
        ('quadratic', build_quadratic_equation(1, -5, 6)),
        ('simultaneous', build_simultaneous_equations(1, 1, 5, 1, -1, 1)),
    ]:
        logger.info(f"  [{task_name}] Training...")
        data = _gen_task_data(task_name, n_samples, seed)
        split = int(len(data) * 0.8)
        solver = train_hgnn(data[:split], n_epochs, 1e-3, seed, silent=True)
        he = get_hyperedges(test_prob)
        err, leaf, os_s = measure_single(solver, test_prob, he, seed)
        results[task_name] = (err, leaf, os_s)
        logger.info(f"  {task_name}: done "
                    f"error={err:.4f} leaf={leaf:.3e} OS={os_s}")
        if err > threshold:
            logger.info(f"  *** LIMIT DETECTED: {task_name} error>{threshold} ***")

    # three-variable simultaneous equations
    logger.info("  [three-variable] Training...")
    tv_data, tv_he = gen_three_var_data(n_samples, seed)
    split = int(len(tv_data) * 0.8)
    tv_train = tv_data[:split]

    def tv_he_fn(prob):
        idx = next((i for i, p in enumerate(tv_data) if p is prob), -1)
        return tv_he[idx] if idx >= 0 else tv_he[0]

    solver_tv = train_hgnn_custom(tv_train, tv_he_fn, n_epochs, 1e-3, seed,
                                  task_name="three-variable")
    test_tv, test_tv_he = build_three_variable_equations(
        1, 1, 1, 6,  2, -1, 1, 3,  1, 2, -1, 1)
    err, leaf, os_s = measure_single(solver_tv, test_tv, test_tv_he, seed)
    results['three-variable'] = (err, leaf, os_s)
    logger.info(f"  three-variable: done "
                f"error={err:.4f} leaf={leaf:.3e} OS={os_s}")
    if err > threshold:
        logger.info(f"  *** LIMIT DETECTED: three-variable error>{threshold} ***")

    # word problem
    logger.info("  [word-problem] Training...")
    wp_data, wp_he = gen_word_problem_data(n_samples, seed)
    split = int(len(wp_data) * 0.8)
    wp_train = wp_data[:split]

    def wp_he_fn(prob):
        idx = next((i for i, p in enumerate(wp_data) if p is prob), -1)
        return wp_he[idx] if idx >= 0 else wp_he[0]

    solver_wp = train_hgnn_custom(wp_train, wp_he_fn, n_epochs, 1e-3, seed,
                                  task_name="word-problem")
    test_wp, test_wp_he = build_word_problem(3.0, 20.0)
    err, leaf, os_s = measure_single(solver_wp, test_wp, test_wp_he, seed)
    results['word-problem'] = (err, leaf, os_s)
    logger.info(f"  word-problem: done "
                f"error={err:.4f} leaf={leaf:.3e} OS={os_s}")
    if err > threshold:
        logger.info(f"  *** LIMIT DETECTED: word-problem error>{threshold} ***")

    # results
    logger.info("")
    logger.info(f"  {'problem_type':<20} {'error':>10} {'leaf_norm':>10} {'OS_start':>8} {'result':>6}")
    logger.info("  " + "-" * 58)
    for name, (err, leaf, os_s) in results.items():
        os_str = str(os_s) if os_s <= 8 else ">8"
        ok = err < threshold
        verdict = "PASS" if ok else "FAIL"
        logger.info(f"  {name:<20} {err:>10.4f} {leaf:>10.4f} {os_str:>8} {verdict:>6}")
    logger.info("  " + "-" * 58)
    logger.info("  criterion: error < 0.1")

    return results


# ─────────────────────────────────────────────
#  軸2: スケーラビリティ
# ─────────────────────────────────────────────

def run_axis2(n_samples=1000, n_epochs=300, seed=42):
    """軸2: ノード数スケーラビリティ。"""
    logger.info("=" * 50)
    logger.info("Axis2 start: graph scale scalability")
    logger.info("=" * 50)

    node_counts = [7, 11, 15, 21]
    results = {}
    threshold = 0.1

    for n_nodes in node_counts:
        task_name = f"N={n_nodes}"
        logger.info(f"  [{task_name}] Training...")

        rng = random.Random(seed)
        train_data = []
        he_map = {}
        for _ in range(n_samples):
            a = rng.randint(1, 9)
            b = rng.randint(-9, 9)
            c = rng.randint(-20, 20)
            prob, he = build_extended_linear(float(a), float(b), float(c), n_nodes)
            train_data.append(prob)
            he_map[id(prob)] = he

        def he_fn(prob, _m=he_map):
            return _m.get(id(prob), [[0, prob.x.size(0) - 1]])

        split = int(len(train_data) * 0.8)
        t0 = time.time()
        solver = train_hgnn_custom(train_data[:split], he_fn, n_epochs, 1e-3, seed,
                                   task_name=task_name)
        train_time = time.time() - t0

        test_prob, test_he = build_extended_linear(2.0, 3.0, 7.0, n_nodes)
        err, leaf, os_s = measure_single(solver, test_prob, test_he, seed)
        results[n_nodes] = (err, leaf, os_s, train_time)
        logger.info(f"  {task_name}: done "
                    f"error={err:.4f} leaf={leaf:.3e} OS={os_s}")
        if err > threshold:
            logger.info(f"  *** LIMIT DETECTED: {task_name} error>{threshold} ***")

    logger.info("")
    logger.info(f"  {'n_nodes':>8} {'error':>10} {'leaf_norm':>10} {'OS_start':>8} {'time':>10}")
    logger.info("  " + "-" * 50)
    for n, (err, leaf, os_s, t) in results.items():
        os_str = str(os_s) if os_s <= 8 else ">8"
        logger.info(f"  {n:>8} {err:>10.4f} {leaf:>10.4f} {os_str:>8} {t:>9.1f}s")
    logger.info("  " + "-" * 50)
    logger.info("  limit: error > 0.1 or OS < 4")

    return results


# ─────────────────────────────────────────────
#  軸3: ノイズ耐性・汎化
# ─────────────────────────────────────────────

def run_axis3(n_samples=1000, n_epochs=300, seed=42):
    """軸3: OOD 汎化性能。"""
    logger.info("=" * 50)
    logger.info("Axis3 start: noise robustness / generalization")
    logger.info("=" * 50)

    test_ranges = [
        ('in-distribution', (1, 9),   (-9, 9),    (-20, 20)),
        ('mild-ood',        (1, 15),  (-15, 15),  (-30, 30)),
        ('severe-ood',      (1, 50),  (-50, 50),  (-100, 100)),
        ('extreme-ood',     (1, 100), (-100, 100), (-500, 500)),
    ]

    tasks = {
        'linear': ('linear', build_linear_equation),
        'quadratic': ('quadratic', build_quadratic_equation),
        'simultaneous': ('simultaneous', build_simultaneous_equations),
    }

    # 訓練（in-distribution）
    solvers = {}
    for task_name, (gen_name, _) in tasks.items():
        logger.info(f"  [{task_name}] Training...")
        data = _gen_task_data(gen_name, n_samples, seed)
        split = int(len(data) * 0.8)
        solvers[task_name] = train_hgnn(data[:split], n_epochs, 1e-3, seed, silent=True)

    # OOD テスト
    results = {}
    rng = random.Random(seed + 1)
    n_test = 50

    for range_name, a_range, b_range, c_range in test_ranges:
        logger.info(f"  OOD test: {range_name}")
        row = {}
        for task_name in tasks:
            errors = []
            for _ in range(n_test):
                a = rng.randint(*a_range)
                b = rng.randint(*b_range)
                c = rng.randint(*c_range)
                if a == 0:
                    a = 1

                if task_name == 'linear':
                    prob = build_linear_equation(float(a), float(b), float(c))
                elif task_name == 'quadratic':
                    D = b ** 2 - 4 * c
                    if D < 0:
                        continue
                    prob = build_quadratic_equation(1.0, float(b), float(c))
                else:
                    a2 = rng.randint(*a_range)
                    b2 = rng.randint(*b_range)
                    c2 = rng.randint(*c_range)
                    if a2 == 0:
                        a2 = 1
                    det = a * b2 - a2 * b
                    if abs(det) < 1e-8:
                        continue
                    prob = build_simultaneous_equations(
                        float(a), float(b), float(c),
                        float(a2), float(b2), float(c2),
                    )
                    if math.isnan(prob.answer_value):
                        continue

                solver = solvers[task_name]
                he = get_hyperedges(prob)
                solver.eval()
                with torch.no_grad():
                    out = solver.hgnn(prob.x, he)
                pred = float(solver.decoder(out[prob.answer_node]).item()) * SCALE
                errors.append(abs(pred - prob.answer_value))

            row[task_name] = np.mean(errors) if errors else float('nan')
        results[range_name] = row

    logger.info("")
    logger.info(f"  {'test_range':<16} {'linear':>10} {'quadratic':>10} {'simultaneous':>12}")
    logger.info("  " + "-" * 52)

    in_dist = results.get('in-distribution', {})
    for range_name, row in results.items():
        vals = []
        for t in ['linear', 'quadratic', 'simultaneous']:
            v = row.get(t, float('nan'))
            vals.append(f"{v:>10.4f}")
        logger.info(f"  {range_name:<16} {vals[0]} {vals[1]} {vals[2]}")

        # 限界検出
        if range_name != 'in-distribution':
            for t in ['linear', 'quadratic', 'simultaneous']:
                base = in_dist.get(t, 0.001)
                if base < 1e-8:
                    base = 0.001
                if row.get(t, 0) > base * 3:
                    logger.info(f"  *** LIMIT DETECTED: {range_name}/{t} "
                                f"error>{base * 3:.4f} ***")

    logger.info("  " + "-" * 52)
    logger.info("  limit: range exceeding 3x in-distribution")

    return results


# ─────────────────────────────────────────────
#  軸4: Anchor Score スケーラビリティ
# ─────────────────────────────────────────────

def run_axis4(n_samples=1000, n_epochs=300, seed=42):
    """軸4: ノード数増加に対する Anchor Score の変化。"""
    logger.info("=" * 50)
    logger.info("Axis4 start: Anchor Score scalability")
    logger.info("=" * 50)

    node_counts = [7, 11, 15, 21]
    results = {}
    threshold = 0.9

    for n_nodes in node_counts:
        task_name = f"N={n_nodes}"
        logger.info(f"  [{task_name}] Training...")

        rng = random.Random(seed)
        train_data = []
        he_map = {}
        for _ in range(n_samples):
            a = rng.randint(1, 9)
            b = rng.randint(-9, 9)
            c = rng.randint(-20, 20)
            prob, he = build_extended_linear(float(a), float(b), float(c), n_nodes)
            train_data.append(prob)
            he_map[id(prob)] = he

        def he_fn(prob, _m=he_map):
            return _m.get(id(prob), [[0, prob.x.size(0) - 1]])

        split = int(len(train_data) * 0.8)

        test_prob, test_he = build_extended_linear(2.0, 3.0, 7.0, n_nodes)
        G = nx.from_numpy_array(test_prob.adj.numpy())

        # leaf_norm (before training)
        torch.manual_seed(seed)
        model_pre = HGNNMathSolver()
        model_pre.eval()

        def fwd_pre(x, prob, _m=model_pre, _he=test_he):
            return _m.hgnn(x, _he)[prob.answer_node]
        leaf_pre, _ = measure_leaf_jacobian(fwd_pre, test_prob)

        # training
        solver = train_hgnn_custom(train_data[:split], he_fn, n_epochs, 1e-3, seed,
                                   task_name=task_name)

        # leaf_norm (after training)
        def fwd_post(x, prob, _s=solver, _he=test_he):
            return _s.hgnn(x, _he)[prob.answer_node]
        leaf_post, _ = measure_leaf_jacobian(fwd_post, test_prob)

        # Anchor Score
        solver.eval()
        with torch.no_grad():
            out = solver.hgnn(test_prob.x, test_he)
        anchor = compute_anchor_effect_score(out, G, source_node=0)

        results[n_nodes] = (anchor, leaf_pre, leaf_post)
        logger.info(f"  {task_name}: done "
                    f"Anchor={anchor:.3f} leaf_pre={leaf_pre:.3e} leaf_post={leaf_post:.3e}")
        if anchor > threshold:
            logger.info(f"  *** LIMIT DETECTED: {task_name} Anchor>{threshold} ***")

    logger.info("")
    logger.info(f"  {'n_nodes':>8} {'Anchor Score':>13} {'leaf(pre)':>12} {'leaf(post)':>12}")
    logger.info("  " + "-" * 50)
    for n, (anc, lp, la) in results.items():
        logger.info(f"  {n:>8} {anc:>13.3f} {lp:>12.4f} {la:>12.4f}")
    logger.info("  " + "-" * 50)
    logger.info("  limit: Anchor Score > 0.9")

    return results


# ─────────────────────────────────────────────
#  メイン
# ─────────────────────────────────────────────

if __name__ == '__main__':
    setup_logger()

    logger.info("=" * 50)
    logger.info("HGNN bottleneck analysis start")
    logger.info("=" * 50)

    # 軸の小規模設定（訓練量を抑えて実行時間を短縮）
    N_SAMPLES = 1000
    N_EPOCHS = 300
    SEED = 42

    r1 = run_axis1(N_SAMPLES, N_EPOCHS, SEED)
    r2 = run_axis2(N_SAMPLES, N_EPOCHS, SEED)
    r3 = run_axis3(N_SAMPLES, N_EPOCHS, SEED)
    r4 = run_axis4(N_SAMPLES, N_EPOCHS, SEED)

    # 限界点まとめ
    logger.info("")
    logger.info("=" * 50)
    logger.info("All axes complete")
    logger.info("=" * 50)

    # 軸1
    fail1 = [k for k, (e, _, _) in r1.items() if e > 0.1]
    logger.info(f"  Axis1 limit: {fail1 if fail1 else 'none'}")

    # 軸2
    fail2 = [k for k, (e, _, os_s, _) in r2.items() if e > 0.1 or os_s < 4]
    logger.info(f"  Axis2 limit: {fail2 if fail2 else 'none'}")

    # 軸3
    in_d = r3.get('in-distribution', {})
    fail3 = []
    for rname, row in r3.items():
        if rname == 'in-distribution':
            continue
        for t in ['linear', 'quadratic', 'simultaneous']:
            base = in_d.get(t, 0.001)
            if base < 1e-8:
                base = 0.001
            if row.get(t, 0) > base * 3:
                fail3.append(f"{rname}/{t}")
    logger.info(f"  Axis3 limit: {fail3 if fail3 else 'none'}")

    # 軸4
    fail4 = [k for k, (a, _, _) in r4.items() if a > 0.9]
    logger.info(f"  Axis4 limit: {fail4 if fail4 else 'none'}")

    logger.info("")
    logger.info("Analysis complete")
    logger.info("=" * 50)
