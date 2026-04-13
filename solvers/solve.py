"""
コマンドラインから数学問題を入力して GNN に解かせるツール。

使用例:
  python solve.py --eq "2x+3=7"
  python solve.py --eq "x^2-5x+6=0"
  python solve.py --sim "x+y=5,x-y=1"
  python solve.py --interactive
  python solve.py --eq "2x+3=7" --verbose

機能:
  - 問題文字列のパース（簡易パーサー、SymPy 利用可）
  - 問題グラフの自動構築
  - GNN 推論の実行
  - 結果の表示（予測値・正解・誤差・Anchor Score）
  - --verbose オプションで Message Passing 過程を表示
"""

import argparse
import os
import re
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers.math_problem_gnn import (
    build_linear_equation,
    build_quadratic_equation,
    build_simultaneous_equations,
    MathSolverGNN,
    MathProblemGraph,
    evaluate_information_flow,
    NODE_DIM,
    SCALE,
)
from experiments.comparison_experiment import compute_anchor_effect_score
import networkx as nx


# ─────────────────────────────────────────────
#  パーサー
# ─────────────────────────────────────────────

def parse_linear(eq_str: str) -> MathProblemGraph:
    """
    一次方程式 "ax+b=c" をパースする。
    対応形式: "2x+3=7", "-3x-1=5", "x=4", "5x=10"
    """
    eq_str = eq_str.replace(' ', '').replace('*', '')
    lhs, rhs = eq_str.split('=')
    c = float(rhs)

    # ax+b のパース
    # x の係数と定数を分離
    pattern = r'([+-]?\d*\.?\d*)x([+-]\d+\.?\d*)?'
    m = re.match(pattern, lhs)
    if not m:
        # 定数なし: "5x=10" → a=5, b=0
        pattern2 = r'([+-]?\d*\.?\d*)x$'
        m2 = re.match(pattern2, lhs)
        if m2:
            a_str = m2.group(1)
            a = float(a_str) if a_str and a_str not in ('+', '-') else (
                1.0 if a_str != '-' else -1.0
            )
            return build_linear_equation(a, 0.0, c)
        raise ValueError(f"Cannot parse linear equation: {eq_str}")

    a_str = m.group(1)
    b_str = m.group(2)

    a = float(a_str) if a_str and a_str not in ('+', '-') else (
        1.0 if a_str != '-' else -1.0
    )
    b = float(b_str) if b_str else 0.0

    return build_linear_equation(a, b, c)


def parse_quadratic(eq_str: str) -> MathProblemGraph:
    """
    二次方程式 "x^2+bx+c=0" をパースする。
    対応形式: "x^2-5x+6=0", "x^2+3x=0", "x^2-4=0"
    """
    eq_str = eq_str.replace(' ', '').replace('*', '')
    lhs, rhs = eq_str.split('=')
    rhs_val = float(rhs)

    # x^2 の係数
    lhs = lhs.replace('^2', '**2')
    pattern = r'([+-]?\d*\.?\d*)x\*\*2([+-]?\d*\.?\d*)x([+-]\d+\.?\d*)?'
    m = re.match(pattern, lhs)
    if m:
        a_str = m.group(1)
        b_str = m.group(2)
        c_str = m.group(3)
        a = float(a_str) if a_str and a_str not in ('+', '-') else (
            1.0 if a_str != '-' else -1.0
        )
        b = float(b_str) if b_str and b_str not in ('+', '-') else (
            1.0 if b_str != '-' else -1.0
        )
        c_val = float(c_str) if c_str else 0.0
        c_val -= rhs_val
        return build_quadratic_equation(a, b, c_val)

    # x^2+c=0 形式
    pattern2 = r'([+-]?\d*\.?\d*)x\*\*2([+-]\d+\.?\d*)?'
    m2 = re.match(pattern2, lhs)
    if m2:
        a_str = m2.group(1)
        c_str = m2.group(2)
        a = float(a_str) if a_str and a_str not in ('+', '-') else (
            1.0 if a_str != '-' else -1.0
        )
        c_val = float(c_str) if c_str else 0.0
        c_val -= rhs_val
        return build_quadratic_equation(a, 0.0, c_val)

    raise ValueError(f"Cannot parse quadratic equation: {eq_str}")


def parse_simultaneous(eq_str: str) -> MathProblemGraph:
    """
    連立方程式 "a1x+b1y=c1,a2x+b2y=c2" をパースする。
    """
    parts = eq_str.replace(' ', '').split(',')
    if len(parts) != 2:
        raise ValueError(f"Expected 2 equations separated by comma: {eq_str}")

    def parse_one(s):
        lhs, rhs = s.split('=')
        c = float(rhs)
        # ax+by のパース
        pattern = r'([+-]?\d*\.?\d*)x([+-]\d*\.?\d*)y'
        m = re.match(pattern, lhs)
        if not m:
            raise ValueError(f"Cannot parse equation: {s}")
        a_str = m.group(1)
        b_str = m.group(2)
        a = float(a_str) if a_str and a_str not in ('+', '-') else (
            1.0 if a_str != '-' else -1.0
        )
        b = float(b_str) if b_str and b_str not in ('+', '-') else (
            1.0 if b_str != '-' else -1.0
        )
        return a, b, c

    a1, b1, c1 = parse_one(parts[0])
    a2, b2, c2 = parse_one(parts[1])
    return build_simultaneous_equations(a1, b1, c1, a2, b2, c2)


def parse_equation(eq_str: str) -> tuple[MathProblemGraph, str]:
    """
    入力文字列を自動判別してパースする。

    Returns: (problem_graph, equation_type)
    """
    eq_clean = eq_str.replace(' ', '')

    if ',' in eq_clean and 'y' in eq_clean:
        return parse_simultaneous(eq_clean), 'simultaneous'

    if 'x^2' in eq_clean or 'x**2' in eq_clean:
        return parse_quadratic(eq_clean), 'quadratic'

    return parse_linear(eq_clean), 'linear'


# ─────────────────────────────────────────────
#  推論実行
# ─────────────────────────────────────────────

def load_solver(model_path: str = 'trained_model.pt') -> MathSolverGNN:
    """訓練済みモデルをロードする。なければ未訓練モデルを返す。"""
    solver = MathSolverGNN(node_dim=NODE_DIM, hidden_dim=64, num_layers=4)

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, weights_only=False)
        solver.load_state_dict(checkpoint['model_state_dict'])
        print(f"  [info] Loaded trained model from {model_path}")
    else:
        print(f"  [warn] No trained model found at {model_path}. Using untrained model.")
        print(f"         Run: python math_problem_gnn.py --train")

    solver.eval()
    return solver


def solve_and_display(
    solver: MathSolverGNN,
    problem: MathProblemGraph,
    eq_type: str,
    verbose: bool = False,
):
    """問題を解いて結果を表示する。"""
    print()
    print(f"  Problem: {problem.problem_str}")
    print(f"  Type:    {eq_type}")
    print(f"  Graph:   {problem.x.shape[0]} nodes, "
          f"{int(problem.adj.sum().item() // 2)} edges")
    print()

    # GNN 推論
    result = solver(problem)

    print(f"  True answer:      {result['true_answer']:.4f}")
    print(f"  Predicted (reg):  {result['predicted_value']:.4f}")
    print(f"  Direct read:      {result['direct_read']:.4f}")
    print(f"  Error (direct):   {result['error']:.4f}")

    # Anchor Score
    G = nx.from_numpy_array(problem.adj.numpy())
    anchor_score = compute_anchor_effect_score(result['gnn_output'], G)
    print(f"  Anchor Score:     {anchor_score:.4f}")

    # 評価3軸
    ev = evaluate_information_flow(solver, problem)
    dep_marker = "pass" if ev['dependency_ok'] else "fail"
    route_marker = "pass" if ev['route_ok'] else "fail"
    print(f"  Input Sim:        {ev['input_sim']:.4f}")
    print(f"  Dependency:       {dep_marker} "
          f"(parent={ev['max_parent_sim']:.3f} vs root={ev['root_sim']:.3f})")
    print(f"  Route:            {route_marker} "
          f"(path={ev['path_mean']:.3f} vs non-path={ev['nonpath_mean']:.3f})")

    if verbose:
        print()
        print("  --- Message Passing detail ---")
        solver.analyze_message_passing(problem)


def interactive_mode(solver: MathSolverGNN, verbose: bool = False):
    """対話モードで問題を入力して解く。"""
    print()
    print("=" * 50)
    print("  GNN Math Solver - Interactive Mode")
    print("=" * 50)
    print()
    print("  Enter an equation to solve (or 'quit' to exit):")
    print("  Examples:")
    print("    2x+3=7")
    print("    x^2-5x+6=0")
    print("    x+y=5,x-y=1")
    print()

    while True:
        try:
            eq_str = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if eq_str.lower() in ('quit', 'exit', 'q', ''):
            break

        try:
            problem, eq_type = parse_equation(eq_str)
            solve_and_display(solver, problem, eq_type, verbose=verbose)
            print()
        except (ValueError, ZeroDivisionError) as e:
            print(f"  Error: {e}")
            print()


# ─────────────────────────────────────────────
#  メイン
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GNN Stigmergy Math Solver CLI"
    )
    parser.add_argument(
        '--eq', type=str,
        help='Linear or quadratic equation (e.g., "2x+3=7", "x^2-5x+6=0")'
    )
    parser.add_argument(
        '--sim', type=str,
        help='Simultaneous equations (e.g., "x+y=5,x-y=1")'
    )
    parser.add_argument(
        '--interactive', action='store_true',
        help='Enter interactive mode'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Show Message Passing detail'
    )
    parser.add_argument(
        '--model', type=str, default='trained_model.pt',
        help='Path to trained model (default: trained_model.pt)'
    )

    args = parser.parse_args()

    if not args.eq and not args.sim and not args.interactive:
        parser.print_help()
        print()
        print("  Quick start:")
        print('    python solve.py --eq "2x+3=7"')
        print('    python solve.py --interactive')
        return

    solver = load_solver(args.model)

    if args.interactive:
        interactive_mode(solver, verbose=args.verbose)
        return

    eq_str = args.eq or args.sim
    if not eq_str:
        parser.error("No equation provided.")
        return

    # 連立方程式は --sim 経由
    if args.sim:
        eq_str = args.sim

    try:
        problem, eq_type = parse_equation(eq_str)
        solve_and_display(solver, problem, eq_type, verbose=args.verbose)
    except (ValueError, ZeroDivisionError) as e:
        print(f"  Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
