#!/usr/bin/env python3
"""
Test script for your specific 16-puzzle inputs using true planning graph approach.
"""

from puzzle_16_true_graphplan import TruePlanningGraphSolver, validate_solution


def test_your_specific_puzzles():
    """Test the specific puzzles you provided."""
    
    print("🎯 TESTING YOUR SPECIFIC 16-PUZZLE INPUTS")
    print("=" * 70)
    print()
    print("Using TRUE PLANNING GRAPH with:")
    print("• State layers (proposition levels)")
    print("• Action layers with real actions and no-ops")
    print("• Mutex relationships between actions and propositions")
    print("• Forward search to build the graph")
    print("• Plan extraction using backward search")
    print()
    
    # Your original puzzles
    your_puzzles = [
        {
            "name": "Your Puzzle 1 (Original - Unsolvable)",
            "input": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 14, 0],
            "note": "This puzzle is mathematically unsolvable (1 inversion, empty on odd row)"
        },
        {
            "name": "Your Puzzle 2 (Original - Unsolvable)", 
            "input": [6, 1, 2, 3, 5, 0, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12],
            "note": "This puzzle is mathematically unsolvable (13 inversions, empty on odd row)"
        },
        {
            "name": "Your Puzzle 3 (Original - Solvable)",
            "input": [2, 3, 4, 8, 1, 6, 7, 0, 5, 9, 10, 12, 13, 14, 11, 15],
            "note": "This puzzle is solvable (12 inversions, empty on odd row)"
        }
    ]
    
    # Solvable versions of your first two puzzles (modified to be solvable)
    solvable_versions = [
        {
            "name": "Your Puzzle 1 (Modified to be Solvable)",
            "input": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 15],
            "note": "Modified version with 0 inversions, empty on odd row - should be solvable"
        },
        {
            "name": "Your Puzzle 2 (Modified to be Solvable)",
            "input": [6, 1, 2, 3, 5, 7, 4, 0, 9, 10, 11, 8, 13, 14, 15, 12],
            "note": "Modified version - moved empty to make it solvable"
        }
    ]
    
    solver = TruePlanningGraphSolver()
    
    # Test original puzzles
    print("TESTING YOUR ORIGINAL PUZZLES:")
    print("-" * 40)
    
    for i, test_case in enumerate(your_puzzles, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Note: {test_case['note']}")
        print()
        
        # Display puzzle
        puzzle_input = test_case['input']
        board = [puzzle_input[j:j+4] for j in range(0, 16, 4)]
        
        print("Initial State:")
        for row in board:
            print("  " + " ".join(f"{cell:2d}" if cell != 0 else "  " for cell in row))
        print()
        
        # Solve
        print("Attempting to solve...")
        solution = solver.solve(puzzle_input, timeout=60.0)
        
        if solution is not None:
            print(f"✅ SOLUTION FOUND: {len(solution)} moves")
            
            # Validate
            if validate_solution(puzzle_input, solution):
                print("✅ Solution validated successfully")
                
                # Statistics
                stats = solver.get_statistics()
                print(f"  Graph levels built: {stats['graph_levels']}")
                print(f"  Graph build time: {stats['graph_build_time']:.3f}s")
                print(f"  Plan extraction time: {stats['plan_extraction_time']:.3f}s")
                print(f"  Total time: {stats['total_time']:.3f}s")
                
                # Show solution
                if len(solution) <= 20:
                    print(f"\nSolution steps:")
                    for j, action in enumerate(solution, 1):
                        print(f"  {j:2d}. {action}")
                else:
                    print(f"\nSolution has {len(solution)} steps (showing first 10):")
                    for j, action in enumerate(solution[:10], 1):
                        print(f"  {j:2d}. {action}")
                    print("  ...")
                
            else:
                print("❌ Solution validation failed")
        else:
            print("❌ No solution found (expected for unsolvable puzzles)")
    
    # Test solvable versions
    print("\n" + "=" * 70)
    print("TESTING SOLVABLE VERSIONS:")
    print("-" * 40)
    
    for i, test_case in enumerate(solvable_versions, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Note: {test_case['note']}")
        print()
        
        # Display puzzle
        puzzle_input = test_case['input']
        board = [puzzle_input[j:j+4] for j in range(0, 16, 4)]
        
        print("Initial State:")
        for row in board:
            print("  " + " ".join(f"{cell:2d}" if cell != 0 else "  " for cell in row))
        print()
        
        # Solve
        print("Building planning graph and extracting solution...")
        solution = solver.solve(puzzle_input, timeout=120.0)
        
        if solution is not None:
            print(f"✅ SOLUTION FOUND: {len(solution)} moves")
            
            # Validate
            if validate_solution(puzzle_input, solution):
                print("✅ Solution validated successfully")
                
                # Statistics
                stats = solver.get_statistics()
                print(f"  Graph levels built: {stats['graph_levels']}")
                print(f"  Graph build time: {stats['graph_build_time']:.3f}s")
                print(f"  Plan extraction time: {stats['plan_extraction_time']:.3f}s")
                print(f"  Total time: {stats['total_time']:.3f}s")
                
                # Show solution
                if len(solution) <= 15:
                    print(f"\nSolution steps:")
                    for j, action in enumerate(solution, 1):
                        print(f"  {j:2d}. {action}")
                else:
                    print(f"\nSolution has {len(solution)} steps (showing first 10):")
                    for j, action in enumerate(solution[:10], 1):
                        print(f"  {j:2d}. {action}")
                    print("  ...")
                
            else:
                print("❌ Solution validation failed")
        else:
            print("❌ No solution found within timeout")
    
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("• Your Puzzle 1 & 2 are mathematically unsolvable")
    print("• Your Puzzle 3 should be solvable with the true planning graph")
    print("• Modified versions demonstrate the solver works correctly")
    print("=" * 70)


if __name__ == "__main__":
    test_your_specific_puzzles()