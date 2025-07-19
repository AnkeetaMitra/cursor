from collections import deque

def hanoi_bfs(n):
    """
    Solve Tower of Hanoi using BFS
    n: number of disks
    Returns: list of moves to solve the puzzle
    """
    # Initial state: all disks on rod 0 (disk 0 is smallest, disk n-1 is largest)
    # Store from bottom to top: largest to smallest
    initial_state = (tuple(range(n-1, -1, -1)), (), ())
    # Goal state: all disks on rod 2
    goal_state = ((), (), tuple(range(n-1, -1, -1)))
    
    # BFS setup
    queue = deque([(initial_state, [])])
    visited = {initial_state}
    
    print(f"Initial state: {initial_state}")
    print(f"Goal state: {goal_state}")
    print(f"Are they equal? {initial_state == goal_state}")
    
    while queue:
        state, path = queue.popleft()
        
        # Check if we reached the goal
        if state == goal_state:
            return path
        
        # Generate all possible moves
        for from_rod in range(3):
            if state[from_rod]:  # Rod has disks
                # Get the top disk (last element)
                top_disk = state[from_rod][-1]
                
                # Try moving to other rods
                for to_rod in range(3):
                    if from_rod != to_rod:
                        # Check if move is valid (smaller disk on top)
                        if not state[to_rod] or top_disk < state[to_rod][-1]:
                            # Create new state
                            new_state = list(state)
                            new_state[from_rod] = state[from_rod][:-1]  # Remove from end
                            new_state[to_rod] = state[to_rod] + (top_disk,)  # Add to end
                            new_state = tuple(new_state)
                            
                            # If not visited, add to queue
                            if new_state not in visited:
                                visited.add(new_state)
                                new_path = path + [(from_rod, to_rod, top_disk)]
                                queue.append((new_state, new_path))
    
    return []  # No solution found

def print_solution(moves, n):
    """Print the solution moves"""
    rod_names = ['A', 'B', 'C']
    print(f"Solution for {n} disks:")
    for i, (from_rod, to_rod) in enumerate(moves):
        print(f"Move {i+1}: {rod_names[from_rod]} -> {rod_names[to_rod]}")
    print(f"Total moves: {len(moves)}")

# Example usage
if __name__ == "__main__":
    # Test with smaller values first
    for n in [3, 4]:
        print(f"\n{'='*50}")
        solution = hanoi_bfs(n, max_states=100000)
        if solution:
            print_solution(solution, n)
            print(f"Expected moves: {2**n - 1}")
            print(f"Actual moves: {len(solution)}")
            print(f"Optimal: {len(solution) == 2**n - 1}")
    
    # For n=5 and above, use higher limits but warn about memory usage
    print(f"\n{'='*50}")
    print("WARNING: n=5 and above require significant memory and time!")
    print("Estimated states for n=5: ~3 million")
    print("Estimated states for n=6: ~700 million")
    
    # Uncomment to test n=5 (will take time and memory)
    # n = 5
    # print(f"\nTesting n={n}...")
    # solution = hanoi_bfs(n, max_states=5000000)
    # if solution:
    #     print_solution(solution, n)
    #     print(f"Expected moves: {2**n - 1}")
    #     print(f"Actual moves: {len(solution)}")
    #     print(f"Optimal: {len(solution) == 2**n - 1}")