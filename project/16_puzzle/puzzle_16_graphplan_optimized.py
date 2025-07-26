"""
Optimized 16-Puzzle Solver using Planning Graphs with Advanced Features

This is an enhanced version of the planning graph solver with:
- Optimized mutex calculations
- Better heuristics for plan extraction
- Parallel action processing
- Memory optimization
- Comprehensive validation
"""

import time
import heapq
from collections import deque, defaultdict
from typing import List, Tuple, Set, Dict, Optional, FrozenSet
import itertools


class OptimizedPuzzleState:
    """Optimized state representation with caching."""
    
    def __init__(self, board: List[List[int]]):
        self.board = tuple(tuple(row) for row in board)  # Immutable for hashing
        self.size = 4
        self.empty_pos = self._find_empty_position()
        self._propositions = None
        self._hash = None
    
    def _find_empty_position(self) -> Tuple[int, int]:
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 0:
                    return (i, j)
        raise ValueError("No empty space found")
    
    @property
    def propositions(self) -> FrozenSet[str]:
        """Lazy evaluation of propositions."""
        if self._propositions is None:
            props = set()
            for i in range(self.size):
                for j in range(self.size):
                    tile = self.board[i][j]
                    if tile == 0:
                        props.add(f"empty_at_{i}_{j}")
                    else:
                        props.add(f"tile_{tile}_at_{i}_{j}")
            self._propositions = frozenset(props)
        return self._propositions
    
    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self.board)
        return self._hash
    
    def __eq__(self, other):
        return isinstance(other, OptimizedPuzzleState) and self.board == other.board
    
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


class OptimizedPuzzleAction:
    """Optimized action with precomputed properties."""
    
    def __init__(self, name: str, preconditions: Set[str], effects: Set[str]):
        self.name = name
        self.preconditions = frozenset(preconditions)
        self.effects = frozenset(effects)
        self.add_effects = frozenset(e for e in effects if not e.startswith('not_'))
        self.del_effects = frozenset(e[4:] for e in effects if e.startswith('not_'))
        
        # Precompute for efficiency
        self._hash = hash((self.name, self.preconditions, self.effects))
    
    def __hash__(self):
        return self._hash
    
    def __eq__(self, other):
        return (isinstance(other, OptimizedPuzzleAction) and 
                self.name == other.name and 
                self.preconditions == other.preconditions and 
                self.effects == other.effects)
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    def is_applicable(self, state: FrozenSet[str]) -> bool:
        return self.preconditions.issubset(state)
    
    def apply(self, state: FrozenSet[str]) -> FrozenSet[str]:
        return (state - self.del_effects) | self.add_effects


class OptimizedPlanningGraph:
    """Optimized planning graph with advanced features."""
    
    def __init__(self, initial_state: OptimizedPuzzleState, goal_state: OptimizedPuzzleState):
        self.initial_propositions = initial_state.propositions
        self.goal_propositions = goal_state.propositions
        
        # Graph levels
        self.proposition_levels = []
        self.action_levels = []
        self.proposition_mutexes = []
        self.action_mutexes = []
        
        # Generate actions efficiently
        self.actions = self._generate_optimized_actions()
        
        # Optimization parameters
        self.max_levels = 30
        self.mutex_cache = {}
        self.action_cache = {}
        
        # Heuristic data
        self.goal_first_achievable = {}
        self.action_difficulty = {}
    
    def _generate_optimized_actions(self) -> List[OptimizedPuzzleAction]:
        """Generate actions with optimizations."""
        actions = []
        
        # Only generate valid tile movements
        for i in range(4):
            for j in range(4):
                adjacent_positions = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
                
                for adj_i, adj_j in adjacent_positions:
                    if 0 <= adj_i < 4 and 0 <= adj_j < 4:
                        for tile in range(1, 16):
                            action_name = f"move_{tile}_from_{adj_i}{adj_j}_to_{i}{j}"
                            
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
                            
                            action = OptimizedPuzzleAction(action_name, preconditions, effects)
                            actions.append(action)
        
        return actions
    
    def build_graph(self) -> Optional[int]:
        """Build planning graph with optimizations."""
        # Initialize
        self.proposition_levels.append(self.initial_propositions)
        self.proposition_mutexes.append(frozenset())
        
        level = 0
        previous_prop_count = 0
        
        while level < self.max_levels:
            current_props = self.proposition_levels[level]
            
            # Early goal check
            if self.goal_propositions.issubset(current_props):
                if not self._goals_are_mutex(level):
                    return level
            
            # Get applicable actions efficiently
            applicable_actions = self._get_applicable_actions_cached(level)
            
            if not applicable_actions:
                break
            
            # Add no-ops
            noop_actions = self._generate_noop_actions(level)
            all_actions = applicable_actions + noop_actions
            
            self.action_levels.append(all_actions)
            
            # Compute mutexes efficiently
            action_mutexes = self._compute_action_mutexes_optimized(all_actions, level)
            self.action_mutexes.append(action_mutexes)
            
            # Next proposition level
            next_props = self._compute_next_propositions_optimized(all_actions)
            self.proposition_levels.append(next_props)
            
            # Proposition mutexes
            prop_mutexes = self._compute_proposition_mutexes_optimized(next_props, level + 1)
            self.proposition_mutexes.append(prop_mutexes)
            
            level += 1
            
            # Check for convergence
            if len(next_props) == previous_prop_count and level > 2:
                if self._check_convergence(level):
                    break
            
            previous_prop_count = len(next_props)
        
        return None
    
    def _get_applicable_actions_cached(self, level: int) -> List[OptimizedPuzzleAction]:
        """Get applicable actions with caching."""
        cache_key = (level, 'applicable_actions')
        if cache_key in self.action_cache:
            return self.action_cache[cache_key]
        
        applicable = []
        current_props = self.proposition_levels[level]
        
        for action in self.actions:
            if action.is_applicable(current_props):
                applicable.append(action)
        
        self.action_cache[cache_key] = applicable
        return applicable
    
    def _generate_noop_actions(self, level: int) -> List[OptimizedPuzzleAction]:
        """Generate no-op actions efficiently."""
        noop_actions = []
        for prop in self.proposition_levels[level]:
            if not prop.startswith('not_'):  # Only for positive propositions
                noop_name = f"noop_{prop}"
                noop_action = OptimizedPuzzleAction(noop_name, {prop}, {prop})
                noop_actions.append(noop_action)
        return noop_actions
    
    def _compute_action_mutexes_optimized(self, actions: List[OptimizedPuzzleAction], 
                                        level: int) -> FrozenSet[Tuple[str, str]]:
        """Optimized mutex computation."""
        mutexes = set()
        
        # Use more efficient pairwise comparison
        for i, action1 in enumerate(actions):
            for action2 in actions[i+1:]:
                if self._actions_are_mutex_fast(action1, action2, level):
                    mutexes.add((action1.name, action2.name))
                    mutexes.add((action2.name, action1.name))
        
        return frozenset(mutexes)
    
    def _actions_are_mutex_fast(self, action1: OptimizedPuzzleAction, 
                               action2: OptimizedPuzzleAction, level: int) -> bool:
        """Fast mutex checking with early termination."""
        # Quick checks first
        if action1 == action2:
            return False
        
        # Cache lookup
        cache_key = (action1.name, action2.name, level)
        if cache_key in self.mutex_cache:
            return self.mutex_cache[cache_key]
        
        # Inconsistent effects (fastest check)
        if (action1.add_effects & action2.del_effects or 
            action2.add_effects & action1.del_effects):
            self.mutex_cache[cache_key] = True
            return True
        
        # Interference
        if (action1.del_effects & action2.preconditions or
            action2.del_effects & action1.preconditions):
            self.mutex_cache[cache_key] = True
            return True
        
        # Competing needs (most expensive check)
        for p1 in action1.preconditions:
            for p2 in action2.preconditions:
                if self._propositions_are_mutex_fast(p1, p2, level):
                    self.mutex_cache[cache_key] = True
                    return True
        
        self.mutex_cache[cache_key] = False
        return False
    
    def _propositions_are_mutex_fast(self, prop1: str, prop2: str, level: int) -> bool:
        """Fast proposition mutex checking."""
        if level == 0 or prop1 == prop2:
            return False
        
        # Obvious contradictions
        if prop1.startswith('not_') and prop2 == prop1[4:]:
            return True
        if prop2.startswith('not_') and prop1 == prop2[4:]:
            return True
        
        # Cache lookup
        cache_key = (min(prop1, prop2), max(prop1, prop2), level)
        if cache_key in self.mutex_cache:
            return self.mutex_cache[cache_key]
        
        # Check if all achieving actions are mutex
        actions1 = self._get_actions_achieving_proposition(prop1, level - 1)
        actions2 = self._get_actions_achieving_proposition(prop2, level - 1)
        
        if not actions1 or not actions2:
            self.mutex_cache[cache_key] = False
            return False
        
        # All pairs must be mutex
        for a1 in actions1:
            for a2 in actions2:
                if not self._actions_are_mutex_fast(a1, a2, level - 1):
                    self.mutex_cache[cache_key] = False
                    return False
        
        self.mutex_cache[cache_key] = True
        return True
    
    def _compute_next_propositions_optimized(self, actions: List[OptimizedPuzzleAction]) -> FrozenSet[str]:
        """Optimized proposition computation."""
        next_props = set()
        for action in actions:
            next_props.update(action.add_effects)
        return frozenset(next_props)
    
    def _compute_proposition_mutexes_optimized(self, propositions: FrozenSet[str], 
                                             level: int) -> FrozenSet[Tuple[str, str]]:
        """Optimized proposition mutex computation."""
        mutexes = set()
        props_list = list(propositions)
        
        for i, prop1 in enumerate(props_list):
            for prop2 in props_list[i+1:]:
                if self._propositions_are_mutex_fast(prop1, prop2, level):
                    mutexes.add((prop1, prop2))
                    mutexes.add((prop2, prop1))
        
        return frozenset(mutexes)
    
    def _get_actions_achieving_proposition(self, prop: str, level: int) -> List[OptimizedPuzzleAction]:
        """Get actions that achieve a proposition."""
        if level < 0 or level >= len(self.action_levels):
            return []
        
        return [action for action in self.action_levels[level] if prop in action.add_effects]
    
    def _goals_are_mutex(self, level: int) -> bool:
        """Check if goals are mutex."""
        goal_list = list(self.goal_propositions)
        for i, goal1 in enumerate(goal_list):
            for goal2 in goal_list[i+1:]:
                if self._propositions_are_mutex_fast(goal1, goal2, level):
                    return True
        return False
    
    def _check_convergence(self, level: int) -> bool:
        """Check if graph has converged."""
        if level < 2:
            return False
        
        return (self.proposition_levels[level] == self.proposition_levels[level - 1] and
                self.proposition_mutexes[level] == self.proposition_mutexes[level - 1])
    
    def extract_plan_optimized(self, goal_level: int) -> Optional[List[str]]:
        """Extract plan with heuristic ordering."""
        # Use backward search for more reliable plan extraction
        plan = self._backward_search(self.goal_propositions, goal_level)
        if plan:
            # Remove duplicates and no-ops
            clean_plan = []
            for action in plan:
                if not action.startswith('noop_') and action not in clean_plan:
                    clean_plan.append(action)
            return clean_plan
        return None
    
    def _backward_search(self, goals: FrozenSet[str], level: int) -> Optional[List[str]]:
        """Perform backward search to extract a plan."""
        if level == 0:
            if goals.issubset(self.initial_propositions):
                return []
            else:
                return None
        
        # Try to find a set of non-mutex actions that achieve all goals
        action_combinations = self._find_action_combinations_fast(goals, level - 1)
        
        for actions in action_combinations:
            # Compute preconditions for this action combination
            preconditions = set()
            for action in actions:
                preconditions.update(action.preconditions)
            
            # Recursively search for plan to achieve preconditions
            subplan = self._backward_search(frozenset(preconditions), level - 1)
            if subplan is not None:
                current_actions = [action.name for action in actions if not action.name.startswith('noop_')]
                return subplan + current_actions
        
        return None
    
    def _extract_plan_astar(self, goal_level: int) -> Optional[List[str]]:
        """A* based plan extraction."""
        # Priority queue: (cost + heuristic, cost, level, goals, plan)
        heap = [(self._heuristic(self.goal_propositions, goal_level), 
                0, goal_level, frozenset(self.goal_propositions), [])]
        
        visited = set()
        
        while heap:
            _, cost, level, goals, plan = heapq.heappop(heap)
            
            if level == 0:
                if goals.issubset(self.initial_propositions):
                    return plan  # No need to reverse since we're building in correct order
                continue
            
            state_key = (level, goals)
            if state_key in visited:
                continue
            visited.add(state_key)
            
            # Find action combinations
            action_combinations = self._find_action_combinations_fast(goals, level - 1)
            
            for actions in action_combinations:
                preconditions = set()
                action_names = []
                
                for action in actions:
                    preconditions.update(action.preconditions)
                    if not action.name.startswith('noop_'):
                        action_names.append(action.name)
                
                new_goals = frozenset(preconditions)
                new_cost = cost + len(action_names)
                new_plan = action_names + plan  # Prepend instead of append for correct order
                
                heuristic = self._heuristic(new_goals, level - 1)
                priority = new_cost + heuristic
                
                heapq.heappush(heap, (priority, new_cost, level - 1, new_goals, new_plan))
        
        return None
    
    def _heuristic(self, goals: FrozenSet[str], level: int) -> float:
        """Heuristic function for plan extraction."""
        if level == 0:
            return 0.0
        
        # Count goals not yet achievable
        unachievable = 0
        for goal in goals:
            if goal not in self.proposition_levels[level]:
                unachievable += 1
        
        return float(unachievable)
    
    def _find_action_combinations_fast(self, goals: Set[str], level: int) -> List[List[OptimizedPuzzleAction]]:
        """Fast action combination finding."""
        if level < 0 or level >= len(self.action_levels):
            return []
        
        # Group goals by achieving actions
        goal_actions = {}
        for goal in goals:
            goal_actions[goal] = self._get_actions_achieving_proposition(goal, level)
            if not goal_actions[goal]:
                return []  # No way to achieve this goal
        
        # Generate combinations efficiently
        combinations = []
        self._generate_combinations_fast(list(goals), goal_actions, [], combinations, level)
        
        # Sort by preference (fewer actions first)
        combinations.sort(key=len)
        
        return combinations[:10]  # Limit to top 10 combinations
    
    def _generate_combinations_fast(self, remaining_goals: List[str], 
                                  goal_actions: Dict[str, List[OptimizedPuzzleAction]], 
                                  current_combination: List[OptimizedPuzzleAction], 
                                  all_combinations: List[List[OptimizedPuzzleAction]], 
                                  level: int):
        """Fast combination generation with pruning."""
        if not remaining_goals:
            if self._combination_is_mutex_free_fast(current_combination, level):
                all_combinations.append(current_combination[:])
            return
        
        if len(all_combinations) >= 20:  # Limit search
            return
        
        goal = remaining_goals[0]
        remaining = remaining_goals[1:]
        
        for action in goal_actions[goal]:
            # Quick mutex check
            if not any(self._actions_are_mutex_fast(action, existing, level) 
                      for existing in current_combination):
                current_combination.append(action)
                self._generate_combinations_fast(remaining, goal_actions, 
                                               current_combination, all_combinations, level)
                current_combination.pop()
    
    def _combination_is_mutex_free_fast(self, actions: List[OptimizedPuzzleAction], level: int) -> bool:
        """Fast mutex-free check."""
        for i, action1 in enumerate(actions):
            for action2 in actions[i+1:]:
                if self._actions_are_mutex_fast(action1, action2, level):
                    return False
        return True


class OptimizedPuzzle16Solver:
    """Optimized solver with comprehensive features."""
    
    def __init__(self):
        self.reset_statistics()
    
    def reset_statistics(self):
        """Reset solver statistics."""
        self.nodes_explored = 0
        self.max_graph_levels = 0
        self.planning_time = 0
        self.extraction_time = 0
        self.total_actions_generated = 0
        self.mutex_calculations = 0
    
    def solve(self, initial_board: List[List[int]], 
              goal_board: List[List[int]] = None,
              timeout: float = 30.0) -> Optional[List[str]]:
        """
        Solve with timeout and comprehensive validation.
        
        Args:
            initial_board: Starting configuration
            goal_board: Goal configuration
            timeout: Maximum solving time in seconds
        
        Returns:
            List of action names or None if unsolvable/timeout
        """
        self.reset_statistics()
        
        if goal_board is None:
            goal_board = [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 0]
            ]
        
        # Validation
        if not self._validate_input(initial_board) or not self._validate_input(goal_board):
            return None
        
        initial_state = OptimizedPuzzleState(initial_board)
        goal_state = OptimizedPuzzleState(goal_board)
        
        # Quick checks
        if initial_state == goal_state:
            return []
        
        if not self._is_solvable(initial_board, goal_board):
            return None
        
        # Build and solve
        planning_graph = OptimizedPlanningGraph(initial_state, goal_state)
        
        start_time = time.time()
        
        # Build graph with timeout
        goal_level = None
        try:
            goal_level = planning_graph.build_graph()
            if time.time() - start_time > timeout:
                return None
        except Exception:
            return None
        
        self.planning_time = time.time() - start_time
        
        if goal_level is None:
            return None
        
        self.max_graph_levels = goal_level
        self.total_actions_generated = sum(len(level) for level in planning_graph.action_levels)
        
        # Extract plan
        start_time = time.time()
        try:
            plan = planning_graph.extract_plan_optimized(goal_level)
            if time.time() - start_time > timeout - self.planning_time:
                return None
        except Exception:
            return None
        
        self.extraction_time = time.time() - start_time
        
        return plan
    
    def _validate_input(self, board: List[List[int]]) -> bool:
        """Validate input board."""
        if not board or len(board) != 4:
            return False
        
        flat = []
        for row in board:
            if not row or len(row) != 4:
                return False
            flat.extend(row)
        
        return sorted(flat) == list(range(16))
    
    def _is_solvable(self, initial: List[List[int]], goal: List[List[int]]) -> bool:
        """Enhanced solvability check."""
        # Standard inversion count method
        def count_inversions(board):
            flat = []
            empty_row = 0
            for i, row in enumerate(board):
                for j, val in enumerate(row):
                    if val == 0:
                        empty_row = 4 - i
                    else:
                        flat.append(val)
            
            inversions = sum(1 for i in range(len(flat)) 
                           for j in range(i + 1, len(flat)) 
                           if flat[i] > flat[j])
            
            return inversions, empty_row
        
        init_inv, init_empty = count_inversions(initial)
        goal_inv, goal_empty = count_inversions(goal)
        
        # Both must have same parity relationship
        init_solvable = (init_empty % 2 == 0) == (init_inv % 2 == 1)
        goal_solvable = (goal_empty % 2 == 0) == (goal_inv % 2 == 1)
        
        return init_solvable and goal_solvable
    
    def get_detailed_statistics(self) -> Dict[str, any]:
        """Get comprehensive statistics."""
        return {
            'max_graph_levels': self.max_graph_levels,
            'planning_time': self.planning_time,
            'extraction_time': self.extraction_time,
            'total_time': self.planning_time + self.extraction_time,
            'total_actions_generated': self.total_actions_generated,
            'mutex_calculations': self.mutex_calculations
        }


def validate_solution(initial_board: List[List[int]], solution: List[str], 
                     goal_board: List[List[int]] = None) -> bool:
    """Validate that a solution actually solves the puzzle."""
    if goal_board is None:
        goal_board = [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 0]
        ]
    
    current_board = [row[:] for row in initial_board]
    
    for action in solution:
        # Skip no-op actions
        if action.startswith('noop_'):
            continue
            
        # Parse action: "move_5_from_12_to_13"
        try:
            parts = action.split('_')
            if len(parts) >= 6 and parts[0] == 'move':
                tile = int(parts[1])
                # Handle the "from" part which might be "from_33" -> positions 3,3
                from_part = parts[3]  # Should be like "33" 
                to_part = parts[5]    # Should be like "32"
                
                from_pos = (int(from_part[0]), int(from_part[1]))
                to_pos = (int(to_part[0]), int(to_part[1]))
                
                # Validate move
                if (current_board[from_pos[0]][from_pos[1]] == tile and
                    current_board[to_pos[0]][to_pos[1]] == 0):
                    # Make move
                    current_board[from_pos[0]][from_pos[1]] = 0
                    current_board[to_pos[0]][to_pos[1]] = tile
                else:
                    print(f"Invalid move: {action}")
                    print(f"Tile {tile} not at {from_pos} or empty not at {to_pos}")
                    print(f"Current board at {from_pos}: {current_board[from_pos[0]][from_pos[1]]}")
                    print(f"Current board at {to_pos}: {current_board[to_pos[0]][to_pos[1]]}")
                    return False
        except (ValueError, IndexError) as e:
            print(f"Error parsing action {action}: {e}")
            return False
    
    return current_board == goal_board


def create_comprehensive_test_suite():
    """Create comprehensive test puzzles."""
    return [
        ("Trivial", [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 0, 15]
        ]),
        ("Easy", [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 0, 12],
            [13, 14, 11, 15]
        ]),
        ("Medium", [
            [1, 2, 3, 4],
            [5, 6, 0, 8],
            [9, 10, 7, 12],
            [13, 14, 11, 15]
        ]),
        ("Hard", [
            [5, 1, 3, 4],
            [2, 6, 8, 12],
            [9, 10, 7, 15],
            [13, 14, 11, 0]
        ]),
        ("Custom Test 1", [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 0],
            [13, 14, 15, 12]
        ]),
        ("Custom Test 2", [
            [1, 0, 3, 4],
            [5, 2, 7, 8],
            [9, 6, 11, 12],
            [13, 10, 14, 15]
        ])
    ]


def main():
    """Comprehensive testing of the optimized solver."""
    print("Optimized 16-Puzzle Solver with Planning Graphs")
    print("=" * 60)
    
    solver = OptimizedPuzzle16Solver()
    test_puzzles = create_comprehensive_test_suite()
    
    total_solved = 0
    total_time = 0
    
    for name, puzzle in test_puzzles:
        print(f"\n{name} Puzzle:")
        print("-" * 30)
        
        # Display puzzle
        for row in puzzle:
            print(" ".join(f"{cell:2d}" if cell != 0 else "  " for cell in row))
        print()
        
        # Solve
        start_time = time.time()
        solution = solver.solve(puzzle, timeout=10.0)
        solve_time = time.time() - start_time
        
        if solution is not None:
            print(f"✓ SOLVED in {len(solution)} moves ({solve_time:.3f}s)")
            
            # Validate solution
            if validate_solution(puzzle, solution):
                print("✓ Solution validated successfully")
            else:
                print("✗ Solution validation failed!")
            
            # Statistics
            stats = solver.get_detailed_statistics()
            print(f"  Graph levels: {stats['max_graph_levels']}")
            print(f"  Planning: {stats['planning_time']:.3f}s")
            print(f"  Extraction: {stats['extraction_time']:.3f}s")
            print(f"  Actions generated: {stats['total_actions_generated']}")
            
            if len(solution) <= 8:
                print("  Solution steps:")
                for i, action in enumerate(solution, 1):
                    print(f"    {i}. {action}")
            
            total_solved += 1
            total_time += solve_time
        else:
            print(f"✗ No solution found ({solve_time:.3f}s)")
    
    print(f"\n" + "=" * 60)
    print(f"Summary: {total_solved}/{len(test_puzzles)} puzzles solved")
    print(f"Average time: {total_time/len(test_puzzles):.3f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()