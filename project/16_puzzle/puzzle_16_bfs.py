"""
16-Puzzle Solver using Breadth-First Search (BFS)

The 16-puzzle is a sliding puzzle game with numbered tiles from 1-15 and one empty space
arranged in a 4x4 grid. The goal is to arrange the tiles in numerical order with the 
empty space in the bottom-right corner.

This implementation uses BFS to find the optimal solution (minimum number of moves).
"""

from collections import deque
import time
from typing import List, Tuple, Optional, Set


class Puzzle16State:
    """Represents a state of the 16-puzzle."""
    
    def __init__(self, board: List[List[int]]):
        """
        Initialize a puzzle state.
        
        Args:
            board: 4x4 grid where 0 represents the empty space
        """
        self.board = [row[:] for row in board]  # Deep copy
        self.size = 4
        self.empty_pos = self._find_empty_position()
        
    def _find_empty_position(self) -> Tuple[int, int]:
        """Find the position of the empty space (0)."""
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    return (i, j)
        raise ValueError("No empty space found in the board")
    
    def __eq__(self, other) -> bool:
        """Check if two states are equal."""
        if not isinstance(other, Puzzle16State):
            return False
        return self.board == other.board
    
    def __hash__(self) -> int:
        """Make the state hashable for use in sets."""
        return hash(tuple(tuple(row) for row in self.board))
    
    def __str__(self) -> str:
        """String representation of the puzzle state."""
        result = []
        for row in self.board:
            row_str = []
            for cell in row:
                if cell == 0:
                    row_str.append("  ")
                else:
                    row_str.append(f"{cell:2d}")
            result.append(" ".join(row_str))
        return "\n".join(result)
    
    def is_goal(self) -> bool:
        """Check if current state is the goal state."""
        goal = [[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 0]]
        return self.board == goal
    
    def get_possible_moves(self) -> List['Puzzle16State']:
        """Get all possible moves from current state."""
        moves = []
        row, col = self.empty_pos
        
        # Define possible directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        direction_names = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        
        for i, (dr, dc) in enumerate(directions):
            new_row, new_col = row + dr, col + dc
            
            # Check if the new position is within bounds
            if 0 <= new_row < self.size and 0 <= new_col < self.size:
                # Create new state by swapping empty space with the tile
                new_board = [row[:] for row in self.board]
                new_board[row][col], new_board[new_row][new_col] = \
                    new_board[new_row][new_col], new_board[row][col]
                
                new_state = Puzzle16State(new_board)
                new_state.move_direction = direction_names[i]
                new_state.moved_tile = self.board[new_row][new_col]
                moves.append(new_state)
        
        return moves


class Puzzle16BFS:
    """BFS solver for the 16-puzzle."""
    
    def __init__(self):
        self.nodes_explored = 0
        self.max_queue_size = 0
    
    def solve(self, initial_state: Puzzle16State) -> Optional[List[Tuple[str, int]]]:
        """
        Solve the 16-puzzle using BFS.
        
        Args:
            initial_state: Starting state of the puzzle
            
        Returns:
            List of (direction, tile_number) tuples representing the solution path,
            or None if no solution exists
        """
        if initial_state.is_goal():
            return []
        
        # BFS setup
        queue = deque([(initial_state, [])])
        visited: Set[Puzzle16State] = {initial_state}
        
        self.nodes_explored = 0
        self.max_queue_size = 0
        
        while queue:
            self.max_queue_size = max(self.max_queue_size, len(queue))
            current_state, path = queue.popleft()
            self.nodes_explored += 1
            
            # Get all possible moves from current state
            for next_state in current_state.get_possible_moves():
                if next_state not in visited:
                    new_path = path + [(next_state.move_direction, next_state.moved_tile)]
                    
                    if next_state.is_goal():
                        return new_path
                    
                    visited.add(next_state)
                    queue.append((next_state, new_path))
        
        return None  # No solution found
    
    def is_solvable(self, state: Puzzle16State) -> bool:
        """
        Check if the puzzle is solvable using the inversion count method.
        
        For a 4x4 puzzle, it's solvable if:
        - If blank is on an even row counting from bottom (2nd or 4th row from bottom),
          then number of inversions must be odd
        - If blank is on an odd row counting from bottom (1st or 3rd row from bottom),
          then number of inversions must be even
        """
        # Flatten the board and remove the empty space
        flat_board = []
        empty_row_from_bottom = 0
        
        for i in range(4):
            for j in range(4):
                if state.board[i][j] == 0:
                    empty_row_from_bottom = 4 - i  # Row counting from bottom (1-indexed)
                else:
                    flat_board.append(state.board[i][j])
        
        # Count inversions
        inversions = 0
        for i in range(len(flat_board)):
            for j in range(i + 1, len(flat_board)):
                if flat_board[i] > flat_board[j]:
                    inversions += 1
        
        # Apply solvability rules
        if empty_row_from_bottom % 2 == 0:  # Even row from bottom
            return inversions % 2 == 1  # Inversions must be odd
        else:  # Odd row from bottom
            return inversions % 2 == 0  # Inversions must be even


def create_sample_puzzles():
    """Create some sample puzzles for testing."""
    
    # Easy puzzle (2 moves to solve)
    easy_puzzle = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 0, 15]
    ]
    
    # Medium puzzle
    medium_puzzle = [
        [1, 2, 3, 4],
        [5, 6, 0, 8],
        [9, 10, 7, 12],
        [13, 14, 11, 15]
    ]
    
    # Hard puzzle
    hard_puzzle = [
        [5, 1, 3, 4],
        [2, 6, 8, 12],
        [9, 10, 7, 15],
        [13, 14, 11, 0]
    ]
    
    return easy_puzzle, medium_puzzle, hard_puzzle


def main():
    """Main function to demonstrate the 16-puzzle solver."""
    
    print("16-Puzzle Solver using Breadth-First Search")
    print("=" * 50)
    
    solver = Puzzle16BFS()
    
    # Create sample puzzles
    easy, medium, hard = create_sample_puzzles()
    puzzles = [
        ("Easy", easy),
        ("Medium", medium),
        ("Hard", hard)
    ]
    
    for name, puzzle_board in puzzles:
        print(f"\n{name} Puzzle:")
        print("-" * 20)
        
        initial_state = Puzzle16State(puzzle_board)
        print("Initial state:")
        print(initial_state)
        print()
        
        # Check if solvable
        if not solver.is_solvable(initial_state):
            print("This puzzle is not solvable!")
            continue
        
        print("Solving...")
        start_time = time.time()
        
        solution = solver.solve(initial_state)
        
        end_time = time.time()
        
        if solution is not None:
            print(f"Solution found in {len(solution)} moves!")
            print(f"Nodes explored: {solver.nodes_explored}")
            print(f"Maximum queue size: {solver.max_queue_size}")
            print(f"Time taken: {end_time - start_time:.4f} seconds")
            
            if len(solution) <= 20:  # Only show moves for shorter solutions
                print("\nSolution moves:")
                for i, (direction, tile) in enumerate(solution, 1):
                    print(f"{i}. Move tile {tile} {direction}")
            else:
                print(f"\nSolution has {len(solution)} moves (too long to display)")
        else:
            print("No solution found!")
        
        print()


if __name__ == "__main__":
    main()