"""
True Planning Graph 16-Puzzle Solver

This implements a proper planning graph with:
- State layers (proposition levels)
- Action layers with real actions and no-ops
- Mutex relationships between actions and propositions
- Forward search to build the graph
- Plan extraction using backward search

Following the classical GRAPHPLAN algorithm structure.
"""

import time
from collections import defaultdict
from typing import List, Tuple, Set, Dict, Optional, FrozenSet
import itertools


class Proposition:
    """Represents a proposition in the planning graph."""
    
    def __init__(self, name: str):
        self.name = name
        self._hash = hash(name)
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"Prop({self.name})"
    
    def __hash__(self):
        return self._hash
    
    def __eq__(self, other):
        return isinstance(other, Proposition) and self.name == other.name


class Action:
    """Represents an action in the planning graph."""
    
    def __init__(self, name: str, preconditions: Set[Proposition], 
                 add_effects: Set[Proposition], del_effects: Set[Proposition]):
        self.name = name
        self.preconditions = frozenset(preconditions)
        self.add_effects = frozenset(add_effects)
        self.del_effects = frozenset(del_effects)
        self.effects = self.add_effects | self.del_effects
        self._hash = hash((name, self.preconditions, self.add_effects, self.del_effects))
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"Action({self.name})"
    
    def __hash__(self):
        return self._hash
    
    def __eq__(self, other):
        return (isinstance(other, Action) and 
                self.name == other.name and
                self.preconditions == other.preconditions and
                self.add_effects == other.add_effects and
                self.del_effects == other.del_effects)
    
    def is_applicable(self, state: Set[Proposition]) -> bool:
        """Check if action is applicable in given state."""
        return self.preconditions.issubset(state)


class NoOpAction(Action):
    """Special no-op action for persistence."""
    
    def __init__(self, proposition: Proposition):
        super().__init__(f"noop_{proposition.name}", {proposition}, {proposition}, set())
        self.proposition = proposition


class PlanningGraph:
    """True planning graph implementation for 16-puzzle."""
    
    def __init__(self, initial_state: Set[Proposition], goal_state: Set[Proposition]):
        self.initial_state = frozenset(initial_state)
        self.goal_state = frozenset(goal_state)
        
        # Graph structure
        self.proposition_levels = []  # List of sets of propositions
        self.action_levels = []       # List of sets of actions
        
        # Mutex relationships
        self.proposition_mutexes = []  # List of sets of mutex proposition pairs
        self.action_mutexes = []       # List of sets of mutex action pairs
        
        # Generate all possible actions
        self.all_actions = self._generate_all_actions()
        
        # Memoization
        self.mutex_cache = {}
        
        # Termination conditions
        self.max_levels = 50
        self.leveled_off = False
    
    def _generate_all_actions(self) -> List[Action]:
        """Generate all possible move actions for 16-puzzle."""
        actions = []
        
        # For each position (i,j), generate actions to move tiles into that position
        for i in range(4):
            for j in range(4):
                # Adjacent positions that can move into (i,j)
                adjacent_positions = []
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < 4 and 0 <= nj < 4:
                        adjacent_positions.append((ni, nj))
                
                # For each tile value (1-15)
                for tile in range(1, 16):
                    for from_i, from_j in adjacent_positions:
                        action_name = f"move_tile_{tile}_from_{from_i}_{from_j}_to_{i}_{j}"
                        
                        # Preconditions: tile at source, empty at destination
                        preconditions = {
                            Proposition(f"tile_{tile}_at_{from_i}_{from_j}"),
                            Proposition(f"empty_at_{i}_{j}")
                        }
                        
                        # Effects: tile moves to destination, source becomes empty
                        add_effects = {
                            Proposition(f"tile_{tile}_at_{i}_{j}"),
                            Proposition(f"empty_at_{from_i}_{from_j}")
                        }
                        
                        del_effects = {
                            Proposition(f"tile_{tile}_at_{from_i}_{from_j}"),
                            Proposition(f"empty_at_{i}_{j}")
                        }
                        
                        action = Action(action_name, preconditions, add_effects, del_effects)
                        actions.append(action)
        
        return actions
    
    def build_graph(self) -> Optional[int]:
        """Build the planning graph using forward search."""
        print("Building planning graph...")
        
        # Level 0: Initial state
        self.proposition_levels.append(self.initial_state)
        self.proposition_mutexes.append(set())
        
        level = 0
        while level < self.max_levels:
            print(f"  Building level {level}...")
            
            # Check if goals are reachable and non-mutex
            current_props = self.proposition_levels[level]
            if self.goal_state.issubset(current_props):
                if not self._goals_are_mutex(level):
                    print(f"  Goals achievable at level {level}")
                    return level
            
            # Build action level
            applicable_actions = self._get_applicable_actions(level)
            noop_actions = self._generate_noop_actions(level)
            all_level_actions = applicable_actions + noop_actions
            
            if not applicable_actions:  # No progress possible
                print(f"  No applicable actions at level {level}")
                break
            
            self.action_levels.append(set(all_level_actions))
            
            # Compute action mutexes
            action_mutexes = self._compute_action_mutexes(all_level_actions, level)
            self.action_mutexes.append(action_mutexes)
            
            # Build next proposition level
            next_props = self._compute_next_propositions(all_level_actions)
            self.proposition_levels.append(next_props)
            
            # Compute proposition mutexes
            prop_mutexes = self._compute_proposition_mutexes(next_props, level + 1)
            self.proposition_mutexes.append(prop_mutexes)
            
            level += 1
            
            # Check for leveling off
            if self._check_leveling_off(level):
                print(f"  Graph leveled off at level {level}")
                self.leveled_off = True
                break
        
        print(f"  Graph building completed at level {level}")
        return None
    
    def _get_applicable_actions(self, level: int) -> List[Action]:
        """Get all actions applicable at the given level."""
        applicable = []
        current_props = self.proposition_levels[level]
        
        for action in self.all_actions:
            if action.is_applicable(current_props):
                applicable.append(action)
        
        return applicable
    
    def _generate_noop_actions(self, level: int) -> List[NoOpAction]:
        """Generate no-op actions for all propositions at current level."""
        noop_actions = []
        for prop in self.proposition_levels[level]:
            noop_actions.append(NoOpAction(prop))
        return noop_actions
    
    def _compute_action_mutexes(self, actions: List[Action], level: int) -> Set[Tuple[Action, Action]]:
        """Compute mutex relationships between actions."""
        mutexes = set()
        
        for i, action1 in enumerate(actions):
            for action2 in actions[i+1:]:
                if self._actions_are_mutex(action1, action2, level):
                    mutexes.add((action1, action2))
                    mutexes.add((action2, action1))
        
        return mutexes
    
    def _actions_are_mutex(self, action1: Action, action2: Action, level: int) -> bool:
        """Check if two actions are mutex."""
        if action1 == action2:
            return False
        
        # Cache key
        cache_key = (action1, action2, level, 'action_mutex')
        if cache_key in self.mutex_cache:
            return self.mutex_cache[cache_key]
        
        result = False
        
        # Inconsistent effects: one deletes what the other adds
        if (action1.add_effects & action2.del_effects or 
            action2.add_effects & action1.del_effects):
            result = True
        
        # Interference: one deletes precondition of the other
        elif (action1.del_effects & action2.preconditions or
              action2.del_effects & action1.preconditions):
            result = True
        
        # Competing needs: preconditions are mutex
        elif self._preconditions_are_mutex(action1, action2, level):
            result = True
        
        self.mutex_cache[cache_key] = result
        return result
    
    def _preconditions_are_mutex(self, action1: Action, action2: Action, level: int) -> bool:
        """Check if preconditions of two actions are mutex."""
        for p1 in action1.preconditions:
            for p2 in action2.preconditions:
                if self._propositions_are_mutex(p1, p2, level):
                    return True
        return False
    
    def _compute_next_propositions(self, actions: List[Action]) -> FrozenSet[Proposition]:
        """Compute propositions at next level."""
        next_props = set()
        for action in actions:
            next_props.update(action.add_effects)
        return frozenset(next_props)
    
    def _compute_proposition_mutexes(self, propositions: FrozenSet[Proposition], 
                                   level: int) -> Set[Tuple[Proposition, Proposition]]:
        """Compute mutex relationships between propositions."""
        mutexes = set()
        props_list = list(propositions)
        
        for i, prop1 in enumerate(props_list):
            for prop2 in props_list[i+1:]:
                if self._propositions_are_mutex(prop1, prop2, level):
                    mutexes.add((prop1, prop2))
                    mutexes.add((prop2, prop1))
        
        return mutexes
    
    def _propositions_are_mutex(self, prop1: Proposition, prop2: Proposition, level: int) -> bool:
        """Check if two propositions are mutex at given level."""
        if level == 0 or prop1 == prop2:
            return False
        
        # Cache key
        cache_key = (prop1, prop2, level, 'prop_mutex')
        if cache_key in self.mutex_cache:
            return self.mutex_cache[cache_key]
        
        # Get all actions that achieve these propositions
        actions1 = self._get_actions_achieving_proposition(prop1, level - 1)
        actions2 = self._get_actions_achieving_proposition(prop2, level - 1)
        
        if not actions1 or not actions2:
            self.mutex_cache[cache_key] = False
            return False
        
        # Check if all pairs of achieving actions are mutex
        result = True
        for a1 in actions1:
            for a2 in actions2:
                if not self._actions_are_mutex(a1, a2, level - 1):
                    result = False
                    break
            if not result:
                break
        
        self.mutex_cache[cache_key] = result
        return result
    
    def _get_actions_achieving_proposition(self, prop: Proposition, level: int) -> List[Action]:
        """Get all actions that achieve a proposition at given level."""
        if level < 0 or level >= len(self.action_levels):
            return []
        
        achieving_actions = []
        for action in self.action_levels[level]:
            if prop in action.add_effects:
                achieving_actions.append(action)
        
        return achieving_actions
    
    def _goals_are_mutex(self, level: int) -> bool:
        """Check if goal propositions are mutex at given level."""
        goal_list = list(self.goal_state)
        for i, goal1 in enumerate(goal_list):
            for goal2 in goal_list[i+1:]:
                if self._propositions_are_mutex(goal1, goal2, level):
                    return True
        return False
    
    def _check_leveling_off(self, level: int) -> bool:
        """Check if the graph has leveled off."""
        if level < 2:
            return False
        
        # Compare current level with previous
        current_props = self.proposition_levels[level]
        prev_props = self.proposition_levels[level - 1]
        
        current_mutexes = self.proposition_mutexes[level]
        prev_mutexes = self.proposition_mutexes[level - 1]
        
        return (current_props == prev_props and current_mutexes == prev_mutexes)
    
    def extract_solution(self, goal_level: int) -> Optional[List[str]]:
        """Extract solution using backward search from goal level."""
        print(f"Extracting solution from level {goal_level}...")
        
        solution = self._backward_search(self.goal_state, goal_level)
        
        if solution is not None:
            # Filter out no-op actions and remove duplicates while preserving order
            filtered_solution = []
            seen_actions = set()
            for action_name in solution:
                if not action_name.startswith('noop_') and action_name not in seen_actions:
                    filtered_solution.append(action_name)
                    seen_actions.add(action_name)
            return filtered_solution
        
        return None
    
    def _backward_search(self, goals: FrozenSet[Proposition], level: int) -> Optional[List[str]]:
        """Backward search to extract plan."""
        if level == 0:
            if goals.issubset(self.initial_state):
                return []
            else:
                return None
        
        # Find all possible action combinations that achieve the goals
        action_combinations = self._find_achieving_action_combinations(goals, level - 1)
        
        for actions in action_combinations:
            # Compute preconditions for this combination
            preconditions = set()
            for action in actions:
                preconditions.update(action.preconditions)
            
            # Recursively solve for preconditions
            subplan = self._backward_search(frozenset(preconditions), level - 1)
            if subplan is not None:
                # Get unique action names, avoiding duplicates
                action_names = []
                seen_names = set()
                for action in actions:
                    if not isinstance(action, NoOpAction) and action.name not in seen_names:
                        action_names.append(action.name)
                        seen_names.add(action.name)
                return subplan + action_names
        
        return None
    
    def _find_achieving_action_combinations(self, goals: FrozenSet[Proposition], 
                                          level: int) -> List[List[Action]]:
        """Find all combinations of non-mutex actions that achieve the goals."""
        if level < 0 or level >= len(self.action_levels):
            return []
        
        # Group actions by the goals they achieve (each action can achieve multiple goals)
        goal_to_actions = defaultdict(list)
        for action in self.action_levels[level]:
            for goal in goals:
                if goal in action.add_effects:
                    goal_to_actions[goal].append(action)
        
        # Check if all goals can be achieved
        for goal in goals:
            if not goal_to_actions[goal]:
                return []
        
        # Find minimal combinations that cover all goals
        combinations = []
        self._find_minimal_combinations(goals, goal_to_actions, combinations, level)
        
        return combinations
    
    def _find_minimal_combinations(self, goals: FrozenSet[Proposition],
                                 goal_to_actions: Dict[Proposition, List[Action]],
                                 combinations: List[List[Action]],
                                 level: int):
        """Find minimal action combinations that achieve all goals."""
        # Try to find a single action that achieves multiple goals first
        for action in self.action_levels[level]:
            goals_achieved = action.add_effects & goals
            if goals_achieved:
                remaining_goals = goals - goals_achieved
                if not remaining_goals:
                    # Single action achieves all goals
                    combinations.append([action])
                else:
                    # Recursively find actions for remaining goals
                    self._find_combinations_recursive(
                        remaining_goals, goal_to_actions, [action], combinations, level, max_actions=3
                    )
        
        # If no combinations found, try traditional approach
        if not combinations:
            self._generate_action_combinations(
                list(goals), goal_to_actions, [], combinations, level
            )
    
    def _find_combinations_recursive(self, remaining_goals: FrozenSet[Proposition],
                                   goal_to_actions: Dict[Proposition, List[Action]],
                                   current_combination: List[Action],
                                   all_combinations: List[List[Action]],
                                   level: int, max_actions: int = 5):
        """Recursively find action combinations with limits."""
        if not remaining_goals:
            if self._is_mutex_free_combination(current_combination, level):
                all_combinations.append(current_combination[:])
            return
        
        if len(current_combination) >= max_actions or len(all_combinations) >= 5:
            return
        
        # Pick a goal and try actions for it
        goal = next(iter(remaining_goals))
        for action in goal_to_actions[goal]:
            # Check mutex with current combination
            if not any(self._actions_are_mutex(action, existing, level) 
                      for existing in current_combination):
                
                goals_achieved = action.add_effects & remaining_goals
                new_remaining = remaining_goals - goals_achieved
                
                current_combination.append(action)
                self._find_combinations_recursive(
                    new_remaining, goal_to_actions, current_combination, 
                    all_combinations, level, max_actions
                )
                current_combination.pop()

    def _generate_action_combinations(self, remaining_goals: List[Proposition],
                                    goal_to_actions: Dict[Proposition, List[Action]],
                                    current_combination: List[Action],
                                    all_combinations: List[List[Action]],
                                    level: int):
        """Recursively generate action combinations (fallback method)."""
        if not remaining_goals:
            # Check if current combination is mutex-free
            if self._is_mutex_free_combination(current_combination, level):
                all_combinations.append(current_combination[:])
            return
        
        # Limit search to prevent explosion
        if len(all_combinations) >= 3:
            return
        
        goal = remaining_goals[0]
        remaining = remaining_goals[1:]
        
        for action in goal_to_actions[goal]:
            # Check if this action is mutex with any in current combination
            if not any(self._actions_are_mutex(action, existing, level) 
                      for existing in current_combination):
                current_combination.append(action)
                self._generate_action_combinations(
                    remaining, goal_to_actions, current_combination, all_combinations, level
                )
                current_combination.pop()
    
    def _is_mutex_free_combination(self, actions: List[Action], level: int) -> bool:
        """Check if a combination of actions is mutex-free."""
        for i, action1 in enumerate(actions):
            for action2 in actions[i+1:]:
                if self._actions_are_mutex(action1, action2, level):
                    return False
        return True


class TruePlanningGraphSolver:
    """True planning graph solver for 16-puzzle."""
    
    def __init__(self):
        self.reset_statistics()
    
    def reset_statistics(self):
        """Reset solver statistics."""
        self.graph_levels = 0
        self.graph_build_time = 0
        self.plan_extraction_time = 0
        self.total_time = 0
    
    def solve(self, puzzle_input: List[int], timeout: float = 120.0) -> Optional[List[str]]:
        """
        Solve 16-puzzle using true planning graph approach.
        
        Args:
            puzzle_input: Flat list of 16 integers (0 = empty)
            timeout: Maximum solving time
            
        Returns:
            List of action names or None if unsolvable
        """
        self.reset_statistics()
        
        # Convert input to 4x4 board
        if len(puzzle_input) != 16:
            print("Error: Input must be exactly 16 numbers")
            return None
        
        board = [puzzle_input[i:i+4] for i in range(0, 16, 4)]
        
        # Validate input
        if not self._validate_board(board):
            print("Error: Invalid board configuration")
            return None
        
        # Check solvability
        if not self._is_solvable(board):
            print("Error: Puzzle is not solvable")
            return None
        
        # Convert to propositions
        initial_state = self._board_to_propositions(board)
        goal_state = self._board_to_propositions([
            [1, 2, 3, 4],
            [5, 6, 7, 8], 
            [9, 10, 11, 12],
            [13, 14, 15, 0]
        ])
        
        start_time = time.time()
        
        # Build planning graph
        planning_graph = PlanningGraph(initial_state, goal_state)
        
        graph_start = time.time()
        goal_level = planning_graph.build_graph()
        self.graph_build_time = time.time() - graph_start
        
        if goal_level is None:
            print("Could not find solution within graph limits")
            return None
        
        self.graph_levels = goal_level
        
        # Extract solution
        extract_start = time.time()
        solution = planning_graph.extract_solution(goal_level)
        self.plan_extraction_time = time.time() - extract_start
        
        self.total_time = time.time() - start_time
        
        return solution
    
    def _validate_board(self, board: List[List[int]]) -> bool:
        """Validate board format."""
        if len(board) != 4:
            return False
        
        flat = []
        for row in board:
            if len(row) != 4:
                return False
            flat.extend(row)
        
        return sorted(flat) == list(range(16))
    
    def _is_solvable(self, board: List[List[int]]) -> bool:
        """Check if puzzle is solvable using inversion count."""
        flat = []
        empty_row_from_bottom = 0
        
        for i, row in enumerate(board):
            for j, val in enumerate(row):
                if val == 0:
                    empty_row_from_bottom = 4 - i
                else:
                    flat.append(val)
        
        # Count inversions
        inversions = 0
        for i in range(len(flat)):
            for j in range(i + 1, len(flat)):
                if flat[i] > flat[j]:
                    inversions += 1
        
        # Solvability rule for 4x4 puzzle
        if empty_row_from_bottom % 2 == 0:  # Even row from bottom
            return inversions % 2 == 1  # Odd inversions
        else:  # Odd row from bottom  
            return inversions % 2 == 0  # Even inversions
    
    def _board_to_propositions(self, board: List[List[int]]) -> Set[Proposition]:
        """Convert board to set of propositions."""
        propositions = set()
        
        for i in range(4):
            for j in range(4):
                tile = board[i][j]
                if tile == 0:
                    propositions.add(Proposition(f"empty_at_{i}_{j}"))
                else:
                    propositions.add(Proposition(f"tile_{tile}_at_{i}_{j}"))
        
        return propositions
    
    def get_statistics(self) -> Dict[str, float]:
        """Get detailed solving statistics."""
        return {
            'graph_levels': self.graph_levels,
            'graph_build_time': self.graph_build_time,
            'plan_extraction_time': self.plan_extraction_time,
            'total_time': self.total_time
        }


def validate_solution(puzzle_input: List[int], solution: List[str]) -> bool:
    """Validate that solution actually solves the puzzle."""
    # Convert to board
    board = [puzzle_input[i:i+4] for i in range(0, 16, 4)]
    current_board = [row[:] for row in board]
    
    goal_board = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 0]
    ]
    
    for step, action in enumerate(solution):
        try:
            # Parse action: "move_tile_15_from_2_3_to_3_3"
            parts = action.split('_')
            if len(parts) >= 8 and parts[0] == 'move' and parts[1] == 'tile':
                tile = int(parts[2])
                from_i, from_j = int(parts[4]), int(parts[5])
                to_i, to_j = int(parts[7]), int(parts[8])
                
                # Validate and execute move
                if (current_board[from_i][from_j] == tile and 
                    current_board[to_i][to_j] == 0):
                    current_board[from_i][from_j] = 0
                    current_board[to_i][to_j] = tile
                else:
                    print(f"Invalid move at step {step + 1}: {action}")
                    print(f"Expected tile {tile} at ({from_i},{from_j}), found {current_board[from_i][from_j]}")
                    print(f"Expected empty at ({to_i},{to_j}), found {current_board[to_i][to_j]}")
                    return False
        except (ValueError, IndexError) as e:
            print(f"Error parsing action at step {step + 1}: {action} - {e}")
            return False
    
    return current_board == goal_board


def test_true_planning_graph():
    """Test the true planning graph solver on the provided hard puzzles."""
    
    print("🎯 TRUE PLANNING GRAPH 16-PUZZLE SOLVER")
    print("=" * 80)
    print()
    print("This implements a proper planning graph with:")
    print("• State layers (proposition levels)")
    print("• Action layers with real actions and no-ops")  
    print("• Mutex relationships between actions and propositions")
    print("• Forward search to build the graph")
    print("• Plan extraction using backward search")
    print()
    
    # Test cases
    test_puzzles = [
        {
            "name": "Hard Puzzle 1",
            "input": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 14, 0]
        },
        {
            "name": "Hard Puzzle 2", 
            "input": [6, 1, 2, 3, 5, 0, 7, 4, 9, 10, 11, 8, 13, 14, 15, 12]
        },
        {
            "name": "Hard Puzzle 3",
            "input": [2, 3, 4, 8, 1, 6, 7, 0, 5, 9, 10, 12, 13, 14, 11, 15]
        }
    ]
    
    solver = TruePlanningGraphSolver()
    
    for i, test_case in enumerate(test_puzzles, 1):
        print(f"Test {i}: {test_case['name']}")
        print("-" * 50)
        
        # Display puzzle
        puzzle_input = test_case['input']
        board = [puzzle_input[j:j+4] for j in range(0, 16, 4)]
        
        print("Initial State:")
        for row in board:
            print("  " + " ".join(f"{cell:2d}" if cell != 0 else "  " for cell in row))
        print()
        
        # Solve
        print("Building planning graph and extracting solution...")
        solution = solver.solve(puzzle_input, timeout=180.0)
        
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
        
        print()
    
    print("=" * 80)
    print("TRUE PLANNING GRAPH SOLVER TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_true_planning_graph()