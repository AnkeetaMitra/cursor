# 16-Puzzle Solver using BFS

This project implements a 16-puzzle (also known as the 15-puzzle) solver using the Breadth-First Search (BFS) algorithm. The 16-puzzle is a sliding puzzle game with numbered tiles from 1-15 and one empty space arranged in a 4x4 grid.

## Goal

Arrange the tiles in numerical order with the empty space in the bottom-right corner:

```
 1  2  3  4
 5  6  7  8
 9 10 11 12
13 14 15   
```

## Features

- **BFS Algorithm**: Guarantees finding the optimal solution (minimum number of moves)
- **Solvability Check**: Determines if a puzzle configuration is solvable before attempting to solve
- **Performance Metrics**: Tracks nodes explored, maximum queue size, and solving time
- **Interactive Interface**: Allows users to input custom puzzles or use predefined samples
- **Step-by-step Solution**: Shows the complete solution path with visual representation

## Files

1. **`puzzle_16_bfs.py`** - Main implementation of the BFS solver
2. **`interactive_puzzle.py`** - Interactive interface for solving custom puzzles
3. **`README.md`** - This documentation file

## Usage

### Running the Basic Solver

```bash
python puzzle_16_bfs.py
```

This will run the solver on three predefined sample puzzles (easy, medium, hard) and display the results.

### Running the Interactive Solver

```bash
python interactive_puzzle.py
```

The interactive version provides options to:
- Enter your own puzzle configuration
- Generate a random solvable puzzle
- Choose from sample puzzles
- See step-by-step solutions

### Input Format

When entering a custom puzzle, provide 16 numbers (0-15) where 0 represents the empty space. You can enter them:

**Single line format:**
```
1 2 3 4 5 6 7 8 9 10 11 12 13 14 0 15
```

**Multi-line format:**
```
1 2 3 4
5 6 7 8
9 10 11 12
13 14 0 15
```

## Algorithm Details

### BFS Implementation

The solver uses Breadth-First Search to explore all possible moves level by level:

1. Start with the initial puzzle state
2. Generate all possible moves (up, down, left, right for the empty space)
3. Add new states to a queue if not already visited
4. Continue until the goal state is found
5. Return the path of moves that leads to the solution

### Solvability

Not all 16-puzzle configurations are solvable. The solver uses the inversion count method:

- For a 4x4 puzzle, count the number of inversions (pairs of tiles where the first appears before the second but has a higher number)
- Check the position of the empty space
- Apply solvability rules based on the empty space position and inversion count

### Time Complexity

- **Time Complexity**: O(16!) in the worst case, but typically much better with pruning
- **Space Complexity**: O(16!) for storing visited states

## Example Output

```
16-Puzzle Solver using Breadth-First Search
==================================================

Easy Puzzle:
--------------------
Initial state:
 1  2  3  4
 5  6  7  8
 9 10 11 12
13 14     15

Solving...
Solution found in 2 moves!
Nodes explored: 8
Maximum queue size: 6
Time taken: 0.0012 seconds

Solution moves:
1. Move tile 15 LEFT
2. Move tile 15 UP
```

## Performance Notes

- Easy puzzles (1-5 moves): Solve instantly
- Medium puzzles (6-15 moves): Solve in seconds
- Hard puzzles (16+ moves): May take longer due to the exponential search space
- Very hard puzzles (25+ moves): May require significant time and memory

The BFS algorithm guarantees the optimal solution but can be memory-intensive for complex puzzles. For puzzles requiring more than 25-30 moves, consider using A* algorithm with a heuristic function for better performance.

## Dependencies

- Python 3.6+
- No external libraries required (uses only built-in modules)

## Educational Value

This implementation demonstrates:
- Graph search algorithms (BFS)
- State space representation
- Queue-based algorithms
- Combinatorial problem solving
- Solvability analysis using mathematical properties