from collections import deque

def hanoi_bfs(n, max_states=None):
    """
    Solve Tower of Hanoi using BFS (Breadth-First Search)
    
    Args:
        n: number of disks
        max_states: maximum number of states to explore (to prevent memory issues)
    
    Returns: 
        list of moves to solve the puzzle, where each move is (from_rod, to_rod, disk)
        Returns empty list if no solution found within max_states limit
    """
    if n <= 0:
        return []
    
    if n == 1:
        return [(0, 2, 0)]  # Move single disk from rod A to rod C
    
    # Initial state: all disks on rod 0 (disk 0 is smallest, disk n-1 is largest)
    # Each rod is represented as a tuple, with disks stored from bottom to top
    initial_state = (tuple(range(n-1, -1, -1)), (), ())
    
    # Goal state: all disks on rod 2
    goal_state = ((), (), tuple(range(n-1, -1, -1)))
    
    # BFS setup
    queue = deque([(initial_state, [])])
    visited = {initial_state}
    states_explored = 0
    
    print(f"Solving Tower of Hanoi with {n} disks using BFS")
    print(f"Initial state: Rods A={initial_state[0]}, B={initial_state[1]}, C={initial_state[2]}")
    print(f"Goal state: Rods A={goal_state[0]}, B={goal_state[1]}, C={goal_state[2]}")
    
    while queue:
        if max_states and states_explored >= max_states:
            print(f"Reached maximum state limit of {max_states}")
            return []
        
        state, path = queue.popleft()
        states_explored += 1
        
        # Progress indicator for larger problems
        if states_explored % 10000 == 0:
            print(f"Explored {states_explored} states, queue size: {len(queue)}")
        
        # Check if we reached the goal
        if state == goal_state:
            print(f"Solution found! Explored {states_explored} states total.")
            return path
        
        # Generate all possible moves
        for from_rod in range(3):
            if state[from_rod]:  # Rod has disks
                # Get the top disk (last element in tuple)
                top_disk = state[from_rod][-1]
                
                # Try moving to other rods
                for to_rod in range(3):
                    if from_rod != to_rod:
                        # Check if move is valid (can only place smaller disk on larger disk)
                        if not state[to_rod] or top_disk < state[to_rod][-1]:
                            # Create new state
                            new_state = list(state)
                            new_state[from_rod] = state[from_rod][:-1]  # Remove top disk
                            new_state[to_rod] = state[to_rod] + (top_disk,)  # Add disk to top
                            new_state = tuple(new_state)
                            
                            # If not visited, add to queue
                            if new_state not in visited:
                                visited.add(new_state)
                                new_path = path + [(from_rod, to_rod, top_disk)]
                                queue.append((new_state, new_path))
    
    print(f"No solution found after exploring {states_explored} states")
    return []  # No solution found

def print_solution(moves, n):
    """
    Print the solution moves in a readable format
    
    Args:
        moves: list of moves where each move is (from_rod, to_rod, disk)
        n: number of disks
    """
    rod_names = ['A', 'B', 'C']
    
    print(f"\nSolution for {n} disks:")
    print("-" * 40)
    
    if not moves:
        print("No solution found!")
        return
    
    for i, (from_rod, to_rod, disk) in enumerate(moves):
        print(f"Move {i+1:2d}: Move disk {disk} from rod {rod_names[from_rod]} to rod {rod_names[to_rod]}")
    
    print(f"\nTotal moves: {len(moves)}")
    expected_moves = 2**n - 1
    print(f"Expected optimal moves: {expected_moves}")
    print(f"Is optimal: {len(moves) == expected_moves}")

def visualize_move(moves, n):
    """
    Visualize the Tower of Hanoi solution step by step
    
    Args:
        moves: list of moves
        n: number of disks
    """
    # Initialize the rods
    rods = [list(range(n-1, -1, -1)), [], []]
    rod_names = ['A', 'B', 'C']
    
    def print_rods():
        print("\nCurrent state:")
        for i in range(2, -1, -1):  # Print from top to bottom
            line = ""
            for rod in rods:
                if len(rod) > i:
                    disk = rod[i]
                    line += f" {disk} "
                else:
                    line += "   "
                line += "  "
            print(line)
        print("-" * (3 * 6))
        print(" A    B    C ")
        print()
    
    print("Initial configuration:")
    print_rods()
    
    for i, (from_rod, to_rod, disk) in enumerate(moves):
        # Make the move
        moved_disk = rods[from_rod].pop()
        rods[to_rod].append(moved_disk)
        
        print(f"Move {i+1}: Move disk {disk} from rod {rod_names[from_rod]} to rod {rod_names[to_rod]}")
        print_rods()

def test_hanoi_bfs():
    """Test the BFS implementation with different numbers of disks"""
    print("Testing Tower of Hanoi BFS Implementation")
    print("=" * 60)
    
    # Test with small values first
    test_cases = [1, 2, 3]
    
    for n in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing with {n} disk(s)")
        print('='*50)
        
        solution = hanoi_bfs(n)
        print_solution(solution, n)
        
        # Visualize solution for small cases
        if n <= 3 and solution:
            print(f"\nVisualization for {n} disk(s):")
            visualize_move(solution, n)
    
    # Test with larger values (with state limits)
    print(f"\n{'='*50}")
    print("Testing larger cases (with state limits)")
    print('='*50)
    
    # n=4 with reasonable limit
    print(f"\nTesting n=4 (limited to 50,000 states):")
    solution = hanoi_bfs(4, max_states=50000)
    print_solution(solution, 4)
    
    # Warning for even larger cases
    print(f"\n{'='*50}")
    print("Memory and Time Complexity Warning:")
    print('='*50)
    print("BFS for Tower of Hanoi has exponential space complexity!")
    print("Estimated memory usage:")
    print("n=5: ~3-5 million states (~500MB-1GB RAM)")
    print("n=6: ~700 million states (~100GB+ RAM)")
    print("n=7: ~1.5 billion states (~200GB+ RAM)")
    print("\nFor n>=5, consider using recursive or iterative solutions instead.")

# Example usage and testing
if __name__ == "__main__":
    test_hanoi_bfs()
    
    # Uncomment to test n=5 (WARNING: high memory usage!)
    # print(f"\n{'='*50}")
    # print("Testing n=5 (WARNING: High memory usage!)")
    # print('='*50)
    # solution = hanoi_bfs(5, max_states=1000000)
    # print_solution(solution, 5)