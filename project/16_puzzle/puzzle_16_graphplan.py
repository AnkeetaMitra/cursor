"""
16-Puzzle Solver using Planning Graphs with Forward Search and Mutexes

This implementation uses a planning graph approach with forward search to solve
the 16-puzzle optimally. It incorporates mutexes to reduce the search space
and ensure correctness.

The 16-puzzle is a sliding puzzle with numbered tiles 1-15 and one empty space
in a 4x4 grid. The goal is to arrange tiles in numerical order.
"""

import time
from collections import deque, defaultdict
from typing import List, Tuple, Set, Dict, Optional, FrozenSet
import copy


class PuzzleState:
    """Represents a state of the 16-puzzle as a set of propositions."""
    
    def __init__(self, board: List[List[int]]):
        self.board = [row[:] for row in board]
        self.size = 4
        self.empty_pos = self._find_empty_position()
        self.propositions = self._board_to_propositions()
    
    def _find_empty_position(self) -> Tuple[int, int]:
        """Find the position of the empty space (0)."""
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    return (i, j)
        raise ValueError("No empty space found")
    
    def _board_to_propositions(self) -> Set[str]:
        """Convert board state to a set of propositions."""
        props = set()
        for i in range(self.size):
            for j in range(self.size):
                tile = self.board[i][j]
                if tile == 0:
                    props.add(f"empty_at_{i}_{j}")
                else:
                    props.add(f"tile_{tile}_at_{i}_{j}")
        return props
    
    def __str__(self):
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


class PuzzleAction:
    """Represents an action in the 16-puzzle (moving a tile into empty space)."""
    
    def __init__(self, name: str, preconditions: Set[str], effects: Set[str]):
        self.name = name
        self.preconditions = set(preconditions)
        self.effects = set(effects)
        self.add_effects = set(e for e in effects if not e.startswith('not_'))
        self.del_effects = set(e[4:] for e in effects if e.startswith('not_'))
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    def is_applicable(self, state: Set[str]) -> bool:
        """Check if action can be applied in given state."""
        return self.preconditions.issubset(state)
    
    def apply(self, state: Set[str]) -> Set[str]:
        """Apply action to state and return new state."""
        new_state = state.copy()
        new_state -= self.del_effects
        new_state |= self.add_effects
        return new_state


class PuzzlePlanningGraph:
    """Planning graph implementation for 16-puzzle with mutexes."""
    
    def __init__(self, initial_state: PuzzleState, goal_state: PuzzleState):
        self.initial_propositions = initial_state.propositions
        self.goal_propositions = goal_state.propositions
        
        # Graph levels
        self.proposition_levels = []
        self.action_levels = []
        self.proposition_mutexes = []
        self.action_mutexes = []
        
        # Generate all possible actions
        self.actions = self._generate_all_actions()
        
        # Maximum levels to prevent infinite loops
        self.max_levels = 50
        
        # Memoization for mutex calculations
        self.mutex_cache = {}
    
    def _generate_all_actions(self) -> List[PuzzleAction]:
        """Generate all possible move actions for the 16-puzzle."""
        actions = []
        
        # For each position, generate actions to move tiles into that position
        for i in range(4):
            for j in range(4):
                # Generate actions to move tiles from adjacent positions
                adjacent_positions = [
                    (i-1, j), (i+1, j), (i, j-1), (i, j+1)
                ]
                
                for adj_i, adj_j in adjacent_positions:
                    if 0 <= adj_i < 4 and 0 <= adj_j < 4:
                        # For each possible tile value (1-15)
                        for tile in range(1, 16):
                            action_name = f"move_tile_{tile}_from_{adj_i}_{adj_j}_to_{i}_{j}"
                            
                            preconditions = {
                                f"tile_{tile}_at_{adj_i}_{adj_j}",
                                f"empty_at_{i}_{j}"
                            }
                            
                            effects = {
                                f"tile_{tile}_at_{i}_{j}",
                                f"empty_at_{adj_i}_{adj_j}",
                                f"not_tile_{tile}_at_{adj_i}_{adj_j}",
                                f"not_empty_at_{i}_{j}"
                            }
                            
                            action = PuzzleAction(action_name, preconditions, effects)
                            actions.append(action)
        
        return actions
    
    def build_graph(self) -> Optional[int]:
        """Build the planning graph until goal is achievable or max levels reached."""
        # Level 0: Initial state
        self.proposition_levels.append(self.initial_propositions.copy())
        self.proposition_mutexes.append(set())
        
        level = 0
        while level < self.max_levels:
            # Check if goal is reachable and non-mutex
            if self.goal_propositions.issubset(self.proposition_levels[level]):
                if not self._goals_are_mutex(level):
                    return level
            
            # Build next action level
            applicable_actions = self._get_applicable_actions(level)
            
            # Add no-op actions for persistence
            noop_actions = self._generate_noop_actions(level)
            applicable_actions.extend(noop_actions)
            
            self.action_levels.append(applicable_actions)
            
            # Compute action mutexes
            action_mutexes = self._compute_action_mutexes(applicable_actions, level)
            self.action_mutexes.append(action_mutexes)
            
            # Build next proposition level
            next_props = self._compute_next_propositions(applicable_actions, level)
            self.proposition_levels.append(next_props)
            
            # Compute proposition mutexes
            prop_mutexes = self._compute_proposition_mutexes(next_props, level + 1)
            self.proposition_mutexes.append(prop_mutexes)
            
            level += 1
            
            # Check for leveling off (optimization)
            if level > 1 and self._graph_leveled_off(level):
                break
        
        return None
    
    def _get_applicable_actions(self, level: int) -> List[PuzzleAction]:
        """Get all actions applicable at the given level."""
        applicable = []
        current_props = self.proposition_levels[level]
        
        for action in self.actions:
            if action.is_applicable(current_props):
                applicable.append(action)
        
        return applicable
    
    def _generate_noop_actions(self, level: int) -> List[PuzzleAction]:
        """Generate no-op actions for proposition persistence."""
        noop_actions = []
        for prop in self.proposition_levels[level]:
            noop_name = f"noop_{prop}"
            noop_action = PuzzleAction(noop_name, {prop}, {prop})
            noop_actions.append(noop_action)
        return noop_actions
    
    def _compute_action_mutexes(self, actions: List[PuzzleAction], level: int) -> Set[Tuple[str, str]]:
        """Compute mutex relationships between actions."""
        mutexes = set()
        
        for i, action1 in enumerate(actions):
            for j, action2 in enumerate(actions):
                if i < j and self._actions_are_mutex(action1, action2, level):
                    mutexes.add((action1.name, action2.name))
                    mutexes.add((action2.name, action1.name))
        
        return mutexes
    
    def _actions_are_mutex(self, action1: PuzzleAction, action2: PuzzleAction, level: int) -> bool:
        """Check if two actions are mutex."""
        # Cache key for memoization
        cache_key = (action1.name, action2.name, level, 'action_mutex')
        if cache_key in self.mutex_cache:
            return self.mutex_cache[cache_key]
        
        # Inconsistent effects: one deletes what the other adds
        if (action1.add_effects & action2.del_effects or 
            action2.add_effects & action1.del_effects):
            self.mutex_cache[cache_key] = True
            return True
        
        # Competing needs: preconditions are mutex
        for p1 in action1.preconditions:
            for p2 in action2.preconditions:
                if self._propositions_are_mutex(p1, p2, level):
                    self.mutex_cache[cache_key] = True
                    return True
        
        # Interference: one deletes precondition of the other
        if (action1.del_effects & action2.preconditions or
            action2.del_effects & action1.preconditions):
            self.mutex_cache[cache_key] = True
            return True
        
        self.mutex_cache[cache_key] = False
        return False
    
    def _compute_next_propositions(self, actions: List[PuzzleAction], level: int) -> Set[str]:
        """Compute propositions at the next level."""
        next_props = set()
        
        for action in actions:
            next_props.update(action.add_effects)
        
        return next_props
    
    def _compute_proposition_mutexes(self, propositions: Set[str], level: int) -> Set[Tuple[str, str]]:
        """Compute mutex relationships between propositions."""
        mutexes = set()
        
        props_list = list(propositions)
        for i, prop1 in enumerate(props_list):
            for j, prop2 in enumerate(props_list):
                if i < j and self._propositions_are_mutex(prop1, prop2, level):
                    mutexes.add((prop1, prop2))
                    mutexes.add((prop2, prop1))
        
        return mutexes
    
    def _propositions_are_mutex(self, prop1: str, prop2: str, level: int) -> bool:
        """Check if two propositions are mutex at given level."""
        if level == 0:
            return False
        
        # Cache key for memoization
        cache_key = (prop1, prop2, level, 'prop_mutex')
        if cache_key in self.mutex_cache:
            return self.mutex_cache[cache_key]
        
        # Check if all ways of achieving these propositions are mutex
        actions_for_prop1 = self._get_actions_achieving_proposition(prop1, level - 1)
        actions_for_prop2 = self._get_actions_achieving_proposition(prop2, level - 1)
        
        # If either proposition has no achieving actions, they're not mutex
        if not actions_for_prop1 or not actions_for_prop2:
            self.mutex_cache[cache_key] = False
            return False
        
        # Check if all pairs of achieving actions are mutex
        for action1 in actions_for_prop1:
            for action2 in actions_for_prop2:
                if not self._actions_are_mutex(action1, action2, level - 1):
                    self.mutex_cache[cache_key] = False
                    return False
        
        self.mutex_cache[cache_key] = True
        return True
    
    def _get_actions_achieving_proposition(self, prop: str, level: int) -> List[PuzzleAction]:
        """Get all actions that achieve a given proposition at a level."""
        if level < 0 or level >= len(self.action_levels):
            return []
        
        achieving_actions = []
        for action in self.action_levels[level]:
            if prop in action.add_effects:
                achieving_actions.append(action)
        
        return achieving_actions
    
    def _goals_are_mutex(self, level: int) -> bool:
        """Check if goal propositions are mutex at given level."""
        goal_list = list(self.goal_propositions)
        
        for i, goal1 in enumerate(goal_list):
            for j, goal2 in enumerate(goal_list):
                if i < j and self._propositions_are_mutex(goal1, goal2, level):
                    return True
        
        return False
    
    def _graph_leveled_off(self, level: int) -> bool:
        """Check if the graph has leveled off (no new propositions or mutexes)."""
        if level < 2:
            return False
        
        # Check if propositions are the same
        if self.proposition_levels[level] != self.proposition_levels[level - 1]:
            return False
        
        # Check if mutexes are the same
        if self.proposition_mutexes[level] != self.proposition_mutexes[level - 1]:
            return False
        
        return True
    
    def extract_plan(self, goal_level: int) -> Optional[List[str]]:
        """Extract a plan using backward search from the goal level."""
        return self._backward_search(self.goal_propositions, goal_level)
    
    def _backward_search(self, goals: Set[str], level: int) -> Optional[List[str]]:
        """Perform backward search to extract a plan."""
        if level == 0:
            if goals.issubset(self.initial_propositions):
                return []
            else:
                return None
        
        # Try to find a set of non-mutex actions that achieve all goals
        action_combinations = self._find_action_combinations(goals, level - 1)
        
        for actions in action_combinations:
            # Compute preconditions for this action combination
            preconditions = set()
            for action in actions:
                preconditions.update(action.preconditions)
            
            # Recursively search for plan to achieve preconditions
            subplan = self._backward_search(preconditions, level - 1)
            if subplan is not None:
                current_actions = [action.name for action in actions if not action.name.startswith('noop_')]
                return subplan + current_actions
        
        return None
    
    def _find_action_combinations(self, goals: Set[str], level: int) -> List[List[PuzzleAction]]:
        """Find combinations of non-mutex actions that achieve all goals."""
        if level < 0 or level >= len(self.action_levels):
            return []
        
        # Get actions that achieve each goal
        goal_actions = {}
        for goal in goals:
            goal_actions[goal] = self._get_actions_achieving_proposition(goal, level)
        
        # Generate all combinations
        combinations = []
        self._generate_combinations(list(goals), goal_actions, [], combinations, level)
        
        return combinations
    
    def _generate_combinations(self, remaining_goals: List[str], goal_actions: Dict[str, List[PuzzleAction]], 
                             current_combination: List[PuzzleAction], all_combinations: List[List[PuzzleAction]], 
                             level: int):
        """Recursively generate all valid action combinations."""
        if not remaining_goals:
            # Check if current combination is mutex-free
            if self._combination_is_mutex_free(current_combination, level):
                all_combinations.append(current_combination[:])
            return
        
        goal = remaining_goals[0]
        remaining = remaining_goals[1:]
        
        for action in goal_actions[goal]:
            # Check if this action is mutex with any action in current combination
            mutex_with_current = False
            for existing_action in current_combination:
                if self._actions_are_mutex(action, existing_action, level):
                    mutex_with_current = True
                    break
            
            if not mutex_with_current:
                current_combination.append(action)
                self._generate_combinations(remaining, goal_actions, current_combination, all_combinations, level)
                current_combination.pop()
    
    def _combination_is_mutex_free(self, actions: List[PuzzleAction], level: int) -> bool:
        """Check if a combination of actions is mutex-free."""
        for i, action1 in enumerate(actions):
            for j, action2 in enumerate(actions):
                if i < j and self._actions_are_mutex(action1, action2, level):
                    return False
        return True


class Puzzle16GraphPlanSolver:
    """Main solver class for 16-puzzle using planning graphs."""
    
    def __init__(self):
        self.nodes_explored = 0
        self.max_graph_levels = 0
        self.planning_time = 0
        self.extraction_time = 0
    
    def solve(self, initial_board: List[List[int]], goal_board: List[List[int]] = None) -> Optional[List[str]]:
        """
        Solve the 16-puzzle using planning graphs.
        
        Args:
            initial_board: Starting configuration
            goal_board: Goal configuration (default is standard solved state)
        
        Returns:
            List of action names representing the solution, or None if unsolvable
        """
        if goal_board is None:
            goal_board = [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 0]
            ]
        
        initial_state = PuzzleState(initial_board)
        goal_state = PuzzleState(goal_board)
        
        # Check if already solved
        if initial_state.propositions == goal_state.propositions:
            return []
        
        # Check solvability
        if not self._is_solvable(initial_board):
            return None
        
        # Build planning graph
        planning_graph = PuzzlePlanningGraph(initial_state, goal_state)
        
        start_time = time.time()
        goal_level = planning_graph.build_graph()
        self.planning_time = time.time() - start_time
        
        if goal_level is None:
            return None
        
        self.max_graph_levels = goal_level
        
        # Extract plan
        start_time = time.time()
        plan = planning_graph.extract_plan(goal_level)
        self.extraction_time = time.time() - start_time
        
        return plan
    
    def _is_solvable(self, board: List[List[int]]) -> bool:
        """Check if the puzzle configuration is solvable."""
        # Flatten board and find empty position
        flat_board = []
        empty_row_from_bottom = 0
        
        for i in range(4):
            for j in range(4):
                if board[i][j] == 0:
                    empty_row_from_bottom = 4 - i
                else:
                    flat_board.append(board[i][j])
        
        # Count inversions
        inversions = 0
        for i in range(len(flat_board)):
            for j in range(i + 1, len(flat_board)):
                if flat_board[i] > flat_board[j]:
                    inversions += 1
        
        # Solvability rules for 4x4 puzzle
        if empty_row_from_bottom % 2 == 0:  # Even row from bottom
            return inversions % 2 == 1
        else:  # Odd row from bottom
            return inversions % 2 == 0
    
    def get_statistics(self) -> Dict[str, float]:
        """Get solver statistics."""
        return {
            'nodes_explored': self.nodes_explored,
            'max_graph_levels': self.max_graph_levels,
            'planning_time': self.planning_time,
            'extraction_time': self.extraction_time,
            'total_time': self.planning_time + self.extraction_time
        }


def create_test_puzzles():
    """Create test puzzles for validation."""
    
    # Easy puzzle (few moves)
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
    
    # Very hard puzzle
    very_hard_puzzle = [
        [15, 14, 1, 6],
        [9, 11, 4, 12],
        [2, 5, 7, 3],
        [13, 8, 10, 0]
    ]
    
    return [
        ("Easy", easy_puzzle),
        ("Medium", medium_puzzle),
        ("Hard", hard_puzzle),
        ("Very Hard", very_hard_puzzle)
    ]


def print_board(board: List[List[int]], title: str = "Board"):
    """Print a board in a nice format."""
    print(f"{title}:")
    for row in board:
        row_str = []
        for cell in row:
            if cell == 0:
                row_str.append("  ")
            else:
                row_str.append(f"{cell:2d}")
        print(" ".join(row_str))
    print()


def main():
    """Main function to test the planning graph solver."""
    print("16-Puzzle Solver using Planning Graphs with Forward Search and Mutexes")
    print("=" * 80)
    
    solver = Puzzle16GraphPlanSolver()
    test_puzzles = create_test_puzzles()
    
    for name, puzzle in test_puzzles:
        print(f"\n{name} Puzzle:")
        print("-" * 40)
        
        print_board(puzzle, "Initial State")
        
        # Solve the puzzle
        start_time = time.time()
        solution = solver.solve(puzzle)
        total_time = time.time() - start_time
        
        if solution is not None:
            print(f"✓ Solution found with {len(solution)} actions!")
            
            # Get statistics
            stats = solver.get_statistics()
            print(f"Graph levels built: {stats['max_graph_levels']}")
            print(f"Planning time: {stats['planning_time']:.4f}s")
            print(f"Extraction time: {stats['extraction_time']:.4f}s")
            print(f"Total time: {total_time:.4f}s")
            
            # Show first few actions if solution is reasonable length
            if len(solution) <= 10:
                print("\nSolution actions:")
                for i, action in enumerate(solution, 1):
                    print(f"{i}. {action}")
            else:
                print(f"\nSolution has {len(solution)} actions (showing first 5):")
                for i, action in enumerate(solution[:5], 1):
                    print(f"{i}. {action}")
                print("...")
        else:
            print("✗ No solution found or puzzle is unsolvable")
        
        print()


if __name__ == "__main__":
    main()