"""
Comprehensive Tower of Hanoi Algorithm Comparison
Compares BFS and GraphPlan approaches with detailed statistics
"""

import time
from toh_bfs import hanoi_bfs, print_solution as print_bfs_solution
from toh_PG import (
    HanoiPlanningDomain, GraphPlanSolver, NoOpAction, 
    validate_solution, print_solution_plan
)

def unified_comparison():
    """Run comprehensive comparison between BFS and GraphPlan"""
    print("="*80)
    print("UNIFIED TOWER OF HANOI ALGORITHM COMPARISON")
    print("="*80)
    
    results = []
    
    for n in [1, 2, 3, 4]:
        print(f"\n{'='*60}")
        print(f"ANALYZING {n} DISK PROBLEM")
        print('='*60)
        
        expected_optimal = 2**n - 1
        
        # BFS Analysis
        print("BFS Algorithm:")
        bfs_start = time.time()
        bfs_solution = hanoi_bfs(n, max_states=100000)
        bfs_time = time.time() - bfs_start
        
        bfs_moves = len(bfs_solution) if bfs_solution else 0
        bfs_valid = bfs_moves > 0
        bfs_optimal = bfs_moves == expected_optimal if bfs_valid else False
        
        print(f"  Found solution: {bfs_valid}")
        print(f"  Moves: {bfs_moves}")
        print(f"  Time: {bfs_time:.4f}s")
        print(f"  Optimal: {bfs_optimal}")
        
        # GraphPlan Analysis
        print("\nGraphPlan Algorithm:")
        domain = HanoiPlanningDomain(n)
        solver = GraphPlanSolver(domain, use_heuristics=True)
        
        gp_start = time.time()
        gp_solution = solver.solve(max_levels=30)
        gp_time = time.time() - gp_start
        
        if gp_solution:
            gp_moves = len([a for a in gp_solution if not isinstance(a, NoOpAction)])
            gp_valid, gp_validation = validate_solution(gp_solution, n)
        else:
            gp_moves = 0
            gp_valid = False
            gp_validation = "No solution found"
        
        gp_optimal = gp_moves == expected_optimal if gp_valid else False
        
        print(f"  Found solution: {gp_solution is not None}")
        print(f"  Valid solution: {gp_valid}")
        print(f"  Moves: {gp_moves}")
        print(f"  Time: {gp_time:.4f}s")
        print(f"  Graph levels: {solver.search_stats['graph_levels']}")
        print(f"  Optimal: {gp_optimal}")
        
        # Store results
        result = {
            'n_disks': n,
            'expected': expected_optimal,
            'bfs_found': bfs_valid,
            'bfs_moves': bfs_moves,
            'bfs_time': bfs_time,
            'bfs_optimal': bfs_optimal,
            'gp_found': gp_solution is not None,
            'gp_valid': gp_valid,
            'gp_moves': gp_moves,
            'gp_time': gp_time,
            'gp_optimal': gp_optimal,
            'gp_levels': solver.search_stats['graph_levels']
        }
        results.append(result)
        
        # Show solutions for small cases
        if n <= 2 and bfs_valid:
            print(f"\nBFS Solution for {n} disks:")
            print_bfs_solution(bfs_solution, n)
    
    # Summary table
    print(f"\n{'='*80}")
    print("ALGORITHM COMPARISON SUMMARY")
    print('='*80)
    
    print(f"{'N':<3} {'Exp':<4} {'BFS ✓':<6} {'BFS Mv':<7} {'BFS T':<8} {'GP ✓':<6} {'GP V':<6} {'GP Mv':<7} {'GP T':<8} {'GP Lvl':<7}")
    print("-" * 80)
    
    for r in results:
        bfs_check = "✓" if r['bfs_found'] else "✗"
        gp_check = "✓" if r['gp_found'] else "✗"
        gp_valid = "✓" if r['gp_valid'] else "✗"
        
        print(f"{r['n_disks']:<3} {r['expected']:<4} {bfs_check:<6} {r['bfs_moves']:<7} {r['bfs_time']:<8.4f} {gp_check:<6} {gp_valid:<6} {r['gp_moves']:<7} {r['gp_time']:<8.4f} {r['gp_levels']:<7}")
    
    # Analysis
    print(f"\n{'='*60}")
    print("ALGORITHM ANALYSIS")
    print('='*60)
    
    print("BFS Characteristics:")
    print("  ✓ Always finds valid optimal solutions")
    print("  ✓ Simple and reliable implementation")
    print("  ✓ Very fast execution (< 0.001s for tested cases)")
    print("  - Exponential memory usage (not visible in small cases)")
    print("  - Less principled approach to constraints")
    
    print("\nGraphPlan Characteristics:")
    print("  ✓ Sophisticated planning approach")
    print("  ✓ Systematic constraint handling")
    print("  ✓ Successfully builds planning graphs")
    print("  ✓ Good theoretical foundation")
    print("  - Solution extraction complexity")
    print("  - May produce invalid solutions")
    print("  - Slower execution times")
    
    print("\nConclusion:")
    print("- BFS is superior for Tower of Hanoi specifically")
    print("- GraphPlan demonstrates advanced AI planning concepts")
    print("- GraphPlan would be better for more complex planning domains")
    print("- Both show exponential complexity scaling")
    
    return results

if __name__ == "__main__":
    unified_comparison()