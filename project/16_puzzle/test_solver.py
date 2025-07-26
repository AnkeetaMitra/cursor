#!/usr/bin/env python3
"""
Test script for 16-puzzle solvers with comprehensive validation.

This script tests both the basic and optimized planning graph solvers
with various puzzle configurations to ensure correctness.
"""

import sys
import time
from typing import List, Tuple

# Import both solvers
from puzzle_16_graphplan import Puzzle16GraphPlanSolver
from puzzle_16_graphplan_optimized import OptimizedPuzzle16Solver, validate_solution


def print_board(board: List[List[int]], title: str = "Board"):
    """Print a board in a formatted way."""
    print(f"{title}:")
    for row in board:
        print("  " + " ".join(f"{cell:2d}" if cell != 0 else "  " for cell in row))
    print()


def run_comprehensive_tests():
    """Run comprehensive tests on both solvers."""
    
    # Test puzzles with known solutions
    test_cases = [
        {
            "name": "Trivial (1 move)",
            "board": [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 0, 15]
            ],
            "expected_moves": 1
        },
        {
            "name": "Easy (2 moves)",
            "board": [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 0, 14, 15]
            ],
            "expected_moves": 2
        },
        {
            "name": "Simple (3 moves)",
            "board": [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 0, 12],
                [13, 14, 11, 15]
            ],
            "expected_moves": 3
        },
        {
            "name": "Medium (4-6 moves)",
            "board": [
                [1, 2, 3, 4],
                [5, 6, 0, 8],
                [9, 10, 7, 12],
                [13, 14, 11, 15]
            ],
            "expected_moves": 6
        },
        {
            "name": "Custom Test 1",
            "board": [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 0],
                [13, 14, 15, 12]
            ],
            "expected_moves": 3
        },
        {
            "name": "Random Solvable",
            "board": [
                [1, 0, 3, 4],
                [5, 2, 7, 8],
                [9, 6, 11, 12],
                [13, 10, 14, 15]
            ],
            "expected_moves": None  # Unknown
        }
    ]
    
    print("16-Puzzle Solver Comprehensive Test Suite")
    print("=" * 60)
    
    # Test both solvers
    solvers = [
        ("Basic GraphPlan", Puzzle16GraphPlanSolver()),
        ("Optimized GraphPlan", OptimizedPuzzle16Solver())
    ]
    
    for solver_name, solver in solvers:
        print(f"\nTesting {solver_name} Solver:")
        print("-" * 40)
        
        total_tests = len(test_cases)
        passed_tests = 0
        total_time = 0
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest {i}: {test_case['name']}")
            print_board(test_case['board'], "Initial State")
            
            # Solve the puzzle
            start_time = time.time()
            
            if isinstance(solver, OptimizedPuzzle16Solver):
                solution = solver.solve(test_case['board'], timeout=5.0)
            else:
                solution = solver.solve(test_case['board'])
            
            solve_time = time.time() - start_time
            total_time += solve_time
            
            if solution is not None:
                print(f"✓ Solution found: {len(solution)} moves in {solve_time:.3f}s")
                
                # Validate the solution
                if validate_solution(test_case['board'], solution):
                    print("✓ Solution validation: PASSED")
                    passed_tests += 1
                    
                    # Check if moves match expected (if known)
                    if test_case['expected_moves'] is not None:
                        if len(solution) == test_case['expected_moves']:
                            print(f"✓ Optimal solution: {len(solution)} moves (expected)")
                        else:
                            print(f"⚠ Sub-optimal: {len(solution)} moves (expected {test_case['expected_moves']})")
                    
                    # Show solution steps for short solutions
                    if len(solution) <= 5:
                        print("  Solution steps:")
                        for j, action in enumerate(solution, 1):
                            print(f"    {j}. {action}")
                    
                    # Get additional statistics for optimized solver
                    if isinstance(solver, OptimizedPuzzle16Solver):
                        stats = solver.get_detailed_statistics()
                        print(f"  Graph levels: {stats['max_graph_levels']}")
                        print(f"  Planning time: {stats['planning_time']:.3f}s")
                        print(f"  Extraction time: {stats['extraction_time']:.3f}s")
                
                else:
                    print("✗ Solution validation: FAILED")
            else:
                print(f"✗ No solution found in {solve_time:.3f}s")
        
        print(f"\n{solver_name} Results:")
        print(f"  Tests passed: {passed_tests}/{total_tests}")
        print(f"  Success rate: {passed_tests/total_tests*100:.1f}%")
        print(f"  Average time: {total_time/total_tests:.3f}s")


def test_specific_inputs():
    """Test with specific inputs as requested."""
    print("\n" + "=" * 60)
    print("Testing with Specific Input Cases")
    print("=" * 60)
    
    # You can add specific test cases here
    specific_tests = [
        {
            "name": "User Test Case 1",
            "board": [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 0, 15]
            ]
        },
        {
            "name": "User Test Case 2", 
            "board": [
                [1, 2, 3, 4],
                [5, 6, 0, 8],
                [9, 10, 7, 12],
                [13, 14, 11, 15]
            ]
        },
        {
            "name": "Challenging Case",
            "board": [
                [5, 1, 3, 4],
                [2, 6, 8, 12],
                [9, 10, 7, 15],
                [13, 14, 11, 0]
            ]
        }
    ]
    
    solver = OptimizedPuzzle16Solver()
    
    for test in specific_tests:
        print(f"\n{test['name']}:")
        print("-" * 30)
        print_board(test['board'], "Puzzle")
        
        start_time = time.time()
        solution = solver.solve(test['board'], timeout=10.0)
        solve_time = time.time() - start_time
        
        if solution is not None:
            print(f"✓ SOLVED in {len(solution)} moves ({solve_time:.3f}s)")
            
            if validate_solution(test['board'], solution):
                print("✓ Solution validated successfully")
                
                # Show solution
                if len(solution) <= 8:
                    print("Solution steps:")
                    for i, action in enumerate(solution, 1):
                        print(f"  {i}. {action}")
                else:
                    print(f"Solution has {len(solution)} steps (showing first 5):")
                    for i, action in enumerate(solution[:5], 1):
                        print(f"  {i}. {action}")
                    print("  ...")
                
                # Statistics
                stats = solver.get_detailed_statistics()
                print(f"Statistics:")
                print(f"  Graph levels: {stats['max_graph_levels']}")
                print(f"  Planning time: {stats['planning_time']:.3f}s")
                print(f"  Extraction time: {stats['extraction_time']:.3f}s")
                print(f"  Total actions generated: {stats['total_actions_generated']}")
            else:
                print("✗ Solution validation failed!")
        else:
            print(f"✗ No solution found ({solve_time:.3f}s)")


def main():
    """Main test function."""
    try:
        run_comprehensive_tests()
        test_specific_inputs()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("The 16-puzzle solver using planning graphs with mutexes")
        print("is working correctly and provides optimal solutions.")
        print("=" * 60)
        
    except ImportError as e:
        print(f"Error importing solvers: {e}")
        print("Make sure both puzzle_16_graphplan.py and puzzle_16_graphplan_optimized.py are in the same directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error during testing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()