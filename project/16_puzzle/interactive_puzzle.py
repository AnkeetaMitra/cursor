"""
Interactive 16-Puzzle Solver
Allows users to input custom puzzles and see step-by-step solutions.
"""

from puzzle_16_bfs import Puzzle16State, Puzzle16BFS
import time


def parse_puzzle_input(input_str: str):
    """
    Parse puzzle input from string format.
    
    Args:
        input_str: String containing 16 numbers (0-15) separated by spaces or commas
        
    Returns:
        4x4 board representation
    """
    # Remove extra whitespace and split by whitespace or commas
    numbers = input_str.replace(',', ' ').split()
    
    if len(numbers) != 16:
        raise ValueError(f"Expected 16 numbers, got {len(numbers)}")
    
    # Convert to integers
    try:
        numbers = [int(n) for n in numbers]
    except ValueError:
        raise ValueError("All inputs must be integers")
    
    # Check for valid range and uniqueness
    if set(numbers) != set(range(16)):
        raise ValueError("Must contain exactly the numbers 0-15, each appearing once")
    
    # Convert to 4x4 board
    board = []
    for i in range(4):
        row = numbers[i*4:(i+1)*4]
        board.append(row)
    
    return board


def display_solution_step_by_step(initial_state: Puzzle16State, solution):
    """Display the solution step by step."""
    print("\nStep-by-step solution:")
    print("=" * 30)
    
    current_state = initial_state
    print(f"Initial state:")
    print(current_state)
    print()
    
    for step, (direction, tile) in enumerate(solution, 1):
        # Find the state after this move
        possible_moves = current_state.get_possible_moves()
        for move in possible_moves:
            if (hasattr(move, 'move_direction') and 
                hasattr(move, 'moved_tile') and
                move.move_direction == direction and 
                move.moved_tile == tile):
                current_state = move
                break
        
        print(f"Step {step}: Move tile {tile} {direction}")
        print(current_state)
        print()


def get_puzzle_from_user():
    """Get puzzle input from user."""
    print("Enter the 16-puzzle configuration:")
    print("Use numbers 0-15 where 0 represents the empty space.")
    print("You can enter them in one line separated by spaces or commas,")
    print("or in 4 lines of 4 numbers each.")
    print()
    print("Example: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 0 15")
    print("Or:")
    print("1 2 3 4")
    print("5 6 7 8") 
    print("9 10 11 12")
    print("13 14 0 15")
    print()
    
    # Try to get input
    lines = []
    print("Enter your puzzle (press Enter twice when done):")
    
    while True:
        line = input().strip()
        if not line:
            break
        lines.append(line)
    
    # Join all lines and parse
    input_str = " ".join(lines)
    return parse_puzzle_input(input_str)


def create_random_puzzle():
    """Create a random solvable puzzle."""
    import random
    
    # Start with solved state
    board = [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 10, 11, 12],
             [13, 14, 15, 0]]
    
    # Perform random moves to shuffle
    state = Puzzle16State(board)
    
    for _ in range(100):  # 100 random moves
        moves = state.get_possible_moves()
        if moves:
            state = random.choice(moves)
    
    return state.board


def main():
    """Main interactive function."""
    print("16-Puzzle Interactive Solver")
    print("=" * 40)
    
    solver = Puzzle16BFS()
    
    while True:
        print("\nOptions:")
        print("1. Enter your own puzzle")
        print("2. Generate random puzzle")
        print("3. Use sample puzzles")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            try:
                board = get_puzzle_from_user()
                initial_state = Puzzle16State(board)
            except ValueError as e:
                print(f"Error: {e}")
                continue
                
        elif choice == '2':
            print("Generating random puzzle...")
            board = create_random_puzzle()
            initial_state = Puzzle16State(board)
            print("Generated puzzle:")
            print(initial_state)
            
        elif choice == '3':
            print("\nSample puzzles:")
            print("1. Easy (2 moves)")
            print("2. Medium (4 moves)")
            print("3. Hard (10+ moves)")
            
            sample_choice = input("Choose sample (1-3): ").strip()
            
            if sample_choice == '1':
                board = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 0, 15]]
            elif sample_choice == '2':
                board = [[1, 2, 3, 4], [5, 6, 0, 8], [9, 10, 7, 12], [13, 14, 11, 15]]
            elif sample_choice == '3':
                board = [[5, 1, 3, 4], [2, 6, 8, 12], [9, 10, 7, 15], [13, 14, 11, 0]]
            else:
                print("Invalid choice!")
                continue
                
            initial_state = Puzzle16State(board)
            
        elif choice == '4':
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice! Please enter 1-4.")
            continue
        
        # Show the puzzle
        print("\nPuzzle to solve:")
        print(initial_state)
        
        # Check if already solved
        if initial_state.is_goal():
            print("\nThis puzzle is already solved!")
            continue
        
        # Check if solvable
        if not solver.is_solvable(initial_state):
            print("\nThis puzzle is not solvable!")
            print("The 16-puzzle has configurations that cannot be solved.")
            continue
        
        # Solve the puzzle
        print("\nSolving...")
        start_time = time.time()
        
        solution = solver.solve(initial_state)
        
        end_time = time.time()
        
        if solution is not None:
            print(f"\nSolution found in {len(solution)} moves!")
            print(f"Nodes explored: {solver.nodes_explored}")
            print(f"Maximum queue size: {solver.max_queue_size}")
            print(f"Time taken: {end_time - start_time:.4f} seconds")
            
            # Ask if user wants to see step-by-step solution
            if len(solution) <= 30:  # Only offer for reasonable length solutions
                show_steps = input("\nShow step-by-step solution? (y/n): ").strip().lower()
                if show_steps in ['y', 'yes']:
                    display_solution_step_by_step(initial_state, solution)
            else:
                print(f"\nSolution has {len(solution)} moves (too long to display step-by-step)")
                
            # Show just the moves
            print(f"\nSolution moves:")
            for i, (direction, tile) in enumerate(solution, 1):
                print(f"{i}. Move tile {tile} {direction}")
                
        else:
            print("\nNo solution found! This shouldn't happen for a solvable puzzle.")


if __name__ == "__main__":
    main()