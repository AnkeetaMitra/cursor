import heapq
import time
from collections import defaultdict, deque
from typing import List, Set, Dict, Tuple, Optional
import itertools

class Action:
    def __init__(self, name, preconditions, effects):
        self.name = name
        self.preconditions = set(preconditions)
        self.effects = set(effects)
        self.add_effects = set(e for e in effects if not e.startswith('not_'))
        self.del_effects = set(e[4:] for e in effects if e.startswith('not_'))
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        return isinstance(other, Action) and self.name == other.name
    
    def __lt__(self, other):
        """Add comparison operator for heap operations"""
        return self.name < other.name if isinstance(other, Action) else False
    
    def is_applicable(self, state):
        return self.preconditions.issubset(state)

class HeuristicEstimator:
    """Base class for heuristic estimators"""
    def __init__(self, name):
        self.name = name
        self.computation_time = 0
    
    def estimate(self, state, goals, planning_graph):
        """Return heuristic estimate (lower bound on steps to goal)"""
        raise NotImplementedError

class LevelCostHeuristic(HeuristicEstimator):
    """Level cost heuristic - sum of levels where goals first appear"""
    def __init__(self):
        super().__init__("Level Cost")
    
    def estimate(self, state, goals, planning_graph):
        start_time = time.time()
        
        cost = 0
        for goal in goals:
            # Find first level where goal appears
            level = 0
            found = False
            for level, prop_level in enumerate(planning_graph.proposition_levels):
                if goal in prop_level:
                    cost += level
                    found = True
                    break
            
            if not found:
                cost += len(planning_graph.proposition_levels) * 2  # Higher penalty for unreachable goals
        
        self.computation_time += time.time() - start_time
        return cost

class MaxLevelHeuristic(HeuristicEstimator):
    """Max level heuristic - maximum level where any goal first appears"""
    def __init__(self):
        super().__init__("Max Level")
    
    def estimate(self, state, goals, planning_graph):
        start_time = time.time()
        
        max_level = 0
        for goal in goals:
            for level, prop_level in enumerate(planning_graph.proposition_levels):
                if goal in prop_level:
                    max_level = max(max_level, level)
                    break
        
        self.computation_time += time.time() - start_time
        return max_level

class LevelSumHeuristic(HeuristicEstimator):
    """Level sum heuristic - sum of levels considering goal interactions"""
    def __init__(self):
        super().__init__("Level Sum")
    
    def estimate(self, state, goals, planning_graph):
        start_time = time.time()
        
        # Build a relaxed planning graph ignoring delete effects
        relaxed_levels = [set(state)]
        current_level = set(state)
        
        for level_num in range(20):  # Limit iterations to prevent infinite loops
            next_level = set(current_level)
            
            # Apply all applicable actions (ignoring delete effects)
            for action in planning_graph.actions:
                if action.preconditions.issubset(current_level):
                    next_level.update(action.add_effects)
            
            relaxed_levels.append(next_level)
            
            # Check if all goals are satisfied
            if goals.issubset(next_level):
                break
                
            # Check for convergence
            if next_level == current_level:
                break
                
            current_level = next_level
        
        # Sum the levels where goals first appear in relaxed graph
        cost = 0
        for goal in goals:
            for level, prop_level in enumerate(relaxed_levels):
                if goal in prop_level:
                    cost += level
                    break
            else:
                cost += len(relaxed_levels) * 2  # Goal not reachable
        
        self.computation_time += time.time() - start_time
        return cost

class SetLevelHeuristic(HeuristicEstimator):
    """Set level heuristic - level where all goals can be achieved simultaneously"""
    def __init__(self):
        super().__init__("Set Level")
    
    def estimate(self, state, goals, planning_graph):
        start_time = time.time()
        
        # Find the earliest level where all goals appear and are non-mutex
        for level, prop_level in enumerate(planning_graph.proposition_levels):
            if goals.issubset(prop_level):
                # Check if goals are mutex at this level
                goals_mutex = False
                goals_list = list(goals)
                for i in range(len(goals_list)):
                    for j in range(i + 1, len(goals_list)):
                        if planning_graph.are_mutex_at_level(goals_list[i], goals_list[j], level):
                            goals_mutex = True
                            break
                    if goals_mutex:
                        break
                
                if not goals_mutex:
                    self.computation_time += time.time() - start_time
                    return level
        
        self.computation_time += time.time() - start_time
        return len(planning_graph.proposition_levels)

class SerialPlanningHeuristic(HeuristicEstimator):
    """Serial planning heuristic - plan for each goal independently"""
    def __init__(self):
        super().__init__("Serial Planning")
    
    def estimate(self, state, goals, planning_graph):
        start_time = time.time()
        
        total_cost = 0
        current_state = set(state)
        
        # Plan for each goal independently
        for goal in goals:
            if goal in current_state:
                continue
                
            # Find shortest path to this goal using BFS
            queue = deque([(current_state, 0)])
            visited = set()
            goal_cost = 0
            
            for _ in range(100):  # Limit search depth
                if not queue:
                    break
                    
                curr_state, cost = queue.popleft()
                
                if goal in curr_state:
                    goal_cost = cost
                    break
                
                state_key = frozenset(curr_state)
                if state_key in visited:
                    continue
                visited.add(state_key)
                
                # Try all applicable actions
                for action in planning_graph.actions:
                    if action.preconditions.issubset(curr_state):
                        new_state = (curr_state | action.add_effects) - action.del_effects
                        if len(visited) < 1000:  # Limit memory usage
                            queue.append((new_state, cost + 1))
            
            total_cost += goal_cost
        
        self.computation_time += time.time() - start_time
        return total_cost

class FFHeuristic(HeuristicEstimator):
    """Fast Forward (FF) heuristic - relaxed plan length"""
    def __init__(self):
        super().__init__("Fast Forward")
    
    def estimate(self, state, goals, planning_graph):
        start_time = time.time()
        
        # Build relaxed plan by working backwards from goals
        current_state = set(state)
        plan_length = 0
        remaining_goals = set(goals)
        
        # Iteratively achieve goals
        for iteration in range(20):  # Limit iterations
            if not remaining_goals:
                break
                
            # Find actions that achieve remaining goals
            helpful_actions = []
            for action in planning_graph.actions:
                if action.add_effects & remaining_goals:
                    if action.preconditions.issubset(current_state):
                        helpful_actions.append(action)
            
            if not helpful_actions:
                plan_length += len(remaining_goals) * 2  # Penalty for unreachable goals
                break
            
            # Choose action that achieves the most goals
            best_action = max(helpful_actions, key=lambda a: len(a.add_effects & remaining_goals))
            
            # Apply action (ignore delete effects for relaxed planning)
            current_state.update(best_action.add_effects)
            remaining_goals -= best_action.add_effects
            plan_length += 1
        
        self.computation_time += time.time() - start_time
        return plan_length

class EnhancedGraphPlan:
    def __init__(self, initial_state, goal_state, actions):
        self.initial_state = set(initial_state)
        self.goal_state = set(goal_state)
        self.actions = actions
        self.proposition_levels = []
        self.action_levels = []
        self.proposition_mutexes = []
        self.action_mutexes = []
        self.max_levels = 25  # Increased for more complex problems
        self.counter = 0  # Counter for tie-breaking in heap
        
        # Heuristic estimators
        self.heuristics = {
            'level_cost': LevelCostHeuristic(),
            'max_level': MaxLevelHeuristic(),
            'level_sum': LevelSumHeuristic(),
            'set_level': SetLevelHeuristic(),
            'serial_planning': SerialPlanningHeuristic(),
            'fast_forward': FFHeuristic()
        }
        
        # Performance tracking
        self.performance_data = {}
        
    def build_graph(self):
        """Build the planning graph"""
        start_time = time.time()
        
        # Level 0: Initial state
        self.proposition_levels.append(self.initial_state.copy())
        self.proposition_mutexes.append(set())
        
        level = 0
        while level < self.max_levels:
            # Check if goal is reachable and non-mutex
            if self.goal_state.issubset(self.proposition_levels[level]):
                if not self.goals_are_mutex(level):
                    build_time = time.time() - start_time
                    return level, build_time
            
            # Find applicable actions
            applicable_actions = []
            for action in self.actions:
                if action.is_applicable(self.proposition_levels[level]):
                    applicable_actions.append(action)
            
            # Add no-op actions for persistence
            for prop in self.proposition_levels[level]:
                noop = Action(f"noop_{prop}", [prop], [prop])
                applicable_actions.append(noop)
            
            self.action_levels.append(applicable_actions)
            
            # Compute action mutexes
            action_mutex_pairs = set()
            for i, a1 in enumerate(applicable_actions):
                for j, a2 in enumerate(applicable_actions):
                    if i < j and self.actions_are_mutex(a1, a2, level):
                        action_mutex_pairs.add((a1, a2))
            self.action_mutexes.append(action_mutex_pairs)
            
            # Generate next proposition level
            next_props = set()
            for action in applicable_actions:
                next_props.update(action.add_effects)
            
            self.proposition_levels.append(next_props)
            
            # Compute proposition mutexes
            prop_mutex_pairs = set()
            for prop1 in next_props:
                for prop2 in next_props:
                    if prop1 != prop2 and self.propositions_are_mutex(prop1, prop2, level + 1):
                        prop_mutex_pairs.add((prop1, prop2))
            self.proposition_mutexes.append(prop_mutex_pairs)
            
            level += 1
            
            # Check for fixed point
            if level > 1:
                if (self.proposition_levels[level] == self.proposition_levels[level-1] and 
                    self.proposition_mutexes[level] == self.proposition_mutexes[level-1]):
                    break
        
        build_time = time.time() - start_time
        return -1, build_time
    
    def actions_are_mutex(self, a1, a2, level):
        if a1.name == a2.name:
            return False
            
        # Inconsistent effects
        if a1.add_effects & a2.del_effects or a2.add_effects & a1.del_effects:
            return True
        
        # Interference
        if a1.del_effects & a2.add_effects or a2.del_effects & a1.add_effects:
            return True
        
        # Competing needs
        for p1 in a1.preconditions:
            for p2 in a2.preconditions:
                if self.are_mutex_at_level(p1, p2, level):
                    return True
        
        return False
    
    def propositions_are_mutex(self, p1, p2, level):
        if level == 0:
            return False
            
        p1_actions = [a for a in self.action_levels[level-1] if p1 in a.add_effects]
        p2_actions = [a for a in self.action_levels[level-1] if p2 in a.add_effects]
        
        if not p1_actions or not p2_actions:
            return False
        
        # All pairs of actions must be mutex
        for a1 in p1_actions:
            for a2 in p2_actions:
                if not self.are_mutex_at_level(a1, a2, level-1):
                    return False
        
        return True
    
    def are_mutex_at_level(self, item1, item2, level):
        if level >= len(self.action_mutexes) or level >= len(self.proposition_mutexes):
            return False
            
        if hasattr(item1, 'name'):  # Actions
            return (item1, item2) in self.action_mutexes[level] or (item2, item1) in self.action_mutexes[level]
        else:  # Propositions
            return (item1, item2) in self.proposition_mutexes[level] or (item2, item1) in self.proposition_mutexes[level]
    
    def goals_are_mutex(self, level):
        goals = list(self.goal_state)
        for i in range(len(goals)):
            for j in range(i+1, len(goals)):
                if self.are_mutex_at_level(goals[i], goals[j], level):
                    return True
        return False
    
    def display_plan(self, solution):
        """Display the plan in a readable format"""
        if not solution:
            return "No solution found"
        
        plan_str = []
        plan_str.append(f"Plan ({len(solution)} steps):")
        for i, step in enumerate(solution):
            if isinstance(step, list):
                # Multiple actions in parallel
                if len(step) == 1:
                    plan_str.append(f"  {i+1}. {step[0]}")
                else:
                    plan_str.append(f"  {i+1}. [Parallel] {' | '.join(str(a) for a in step)}")
            else:
                plan_str.append(f"  {i+1}. {step}")
        
        return "\n".join(plan_str)
    
    def validate_plan(self, solution):
        """Validate that the plan actually achieves the goal"""
        if not solution:
            return False, "No solution to validate"
        
        current_state = set(self.initial_state)
        
        for i, step in enumerate(solution):
            if isinstance(step, list):
                # Multiple actions in parallel
                actions_to_apply = step
            else:
                actions_to_apply = [step]
            
            # Check if all actions are applicable
            for action in actions_to_apply:
                if not action.is_applicable(current_state):
                    return False, f"Action {action} not applicable at step {i+1}. Missing: {action.preconditions - current_state}"
            
            # Apply all actions
            for action in actions_to_apply:
                current_state.update(action.add_effects)
                current_state.difference_update(action.del_effects)
        
        # Check if goal is achieved
        if self.goal_state.issubset(current_state):
            return True, "Plan is valid"
        else:
            return False, f"Goal not achieved. Missing: {self.goal_state - current_state}"
    
    def solve_with_heuristic(self, heuristic_name, max_nodes=5000):
        """Solve using specified heuristic with node limit"""
        heuristic = self.heuristics[heuristic_name]
        
        # Reset heuristic computation time
        heuristic.computation_time = 0
        
        start_time = time.time()
        
        # Build planning graph
        goal_level, build_time = self.build_graph()
        
        if goal_level == -1:
            return None, {
                'heuristic': heuristic_name,
                'build_time': build_time,
                'search_time': 0,
                'total_time': time.time() - start_time,
                'heuristic_time': heuristic.computation_time,
                'nodes_expanded': 0,
                'solution_length': 0,
                'status': 'No solution found'
            }
        
        # Extract solution using A* with the specified heuristic
        solution, search_stats = self.extract_solution_with_heuristic(goal_level, heuristic, max_nodes)
        
        total_time = time.time() - start_time
        
        performance = {
            'heuristic': heuristic_name,
            'build_time': build_time,
            'search_time': search_stats['search_time'],
            'total_time': total_time,
            'heuristic_time': heuristic.computation_time,
            'nodes_expanded': search_stats['nodes_expanded'],
            'solution_length': len(solution) if solution else 0,
            'status': search_stats.get('status', 'Success' if solution else 'Failed')
        }
        
        return solution, performance
    
    def extract_solution_with_heuristic(self, goal_level, heuristic, max_nodes):
        """Extract solution using A* with specified heuristic"""
        start_time = time.time()
        nodes_expanded = 0
        
        # Priority queue: (f_score, counter, g_score, level, goals, partial_solution)
        # Using counter to break ties and avoid comparing incomparable objects
        pq = [(0, 0, 0, goal_level, frozenset(self.goal_state), [])]
        visited = set()
        
        while pq and nodes_expanded < max_nodes:
            f_score, counter, g_score, level, goals, partial_solution = heapq.heappop(pq)
            nodes_expanded += 1
            
            # Base case
            if level == 0:
                if goals.issubset(self.initial_state):
                    search_time = time.time() - start_time
                    return list(reversed(partial_solution)), {
                        'search_time': search_time,
                        'nodes_expanded': nodes_expanded,
                        'status': 'Success'
                    }
                continue
            
            # Skip if already visited
            state_key = (level, goals)
            if state_key in visited:
                continue
            visited.add(state_key)
            
            # Generate successor states
            action_sets = self.get_action_sets(goals, level)
            
            for action_set in action_sets[:10]:  # Limit branching factor
                if self.is_valid_action_set(action_set, level):
                    # Get preconditions
                    preconditions = set()
                    for action in action_set:
                        preconditions.update(action.preconditions)
                    
                    # Calculate scores
                    new_g_score = g_score + 1
                    h_score = heuristic.estimate(preconditions, preconditions, self)
                    new_f_score = new_g_score + h_score
                    
                    # Filter out no-ops
                    real_actions = [a for a in action_set if not a.name.startswith('noop_')]
                    
                    # Add to priority queue with unique counter
                    new_partial = [real_actions] + partial_solution if real_actions else partial_solution
                    self.counter += 1
                    heapq.heappush(pq, (new_f_score, self.counter, new_g_score, level - 1, 
                                      frozenset(preconditions), new_partial))
        
        search_time = time.time() - start_time
        status = 'Node limit reached' if nodes_expanded >= max_nodes else 'No solution found'
        return None, {
            'search_time': search_time,
            'nodes_expanded': nodes_expanded,
            'status': status
        }
    
    def get_action_sets(self, goals, level, max_sets=12):
        """Generate valid action sets for goals"""
        if level == 0:
            return []
        
        action_sets = []
        goals_list = list(goals)
        
        # Find all actions that can achieve each goal
        goal_actions = defaultdict(list)
        for goal in goals:
            for action in self.action_levels[level - 1]:
                if goal in action.add_effects:
                    goal_actions[goal].append(action)
        
        # Try to find minimal action sets
        for goal_combo in itertools.combinations(goals_list, min(len(goals_list), 3)):
            # Find actions that can achieve this combination
            possible_actions = []
            for goal in goal_combo:
                possible_actions.extend(goal_actions[goal])
            
            # Remove duplicates
            possible_actions = list(set(possible_actions))
            
            # Try combinations of actions
            for r in range(1, min(len(possible_actions), 4) + 1):
                for action_combo in itertools.combinations(possible_actions, r):
                    achieved = set()
                    for action in action_combo:
                        achieved.update(action.add_effects)
                    
                    # Check if this combination achieves the goals
                    if set(goal_combo).issubset(achieved):
                        action_sets.append(list(action_combo))
                        if len(action_sets) >= max_sets:
                            return action_sets
        
        return action_sets
    
    def is_valid_action_set(self, action_set, level):
        """Check if action set is valid (no mutex pairs)"""
        for i, a1 in enumerate(action_set):
            for j, a2 in enumerate(action_set):
                if i < j and self.are_mutex_at_level(a1, a2, level - 1):
                    return False
        return True
    
    def compare_heuristics(self):
        """Compare all heuristics on the current problem"""
        print(f"\n{'='*80}")
        print(f"HEURISTIC COMPARISON")
        print(f"{'='*80}")
        print(f"Problem: {len(self.initial_state)} initial facts, {len(self.goal_state)} goals")
        print(f"Actions: {len(self.actions)} available actions")
        print(f"Initial state: {sorted(self.initial_state)}")
        print(f"Goal state: {sorted(self.goal_state)}")
        
        results = {}
        
        for heuristic_name in self.heuristics.keys():
            print(f"\n{'-'*60}")
            print(f"Testing {heuristic_name.upper().replace('_', ' ')} Heuristic")
            print(f"{'-'*60}")
            
            # Reset the planner state
            self.proposition_levels = []
            self.action_levels = []
            self.proposition_mutexes = []
            self.action_mutexes = []
            self.counter = 0  # Reset counter
            
            solution, performance = self.solve_with_heuristic(heuristic_name)
            results[heuristic_name] = (solution, performance)
            
            if solution:
                print(f"✓ {performance['status']} in {performance['total_time']:.4f}s")
                print(f"  Solution length: {performance['solution_length']} steps")
                print(f"  Nodes expanded: {performance['nodes_expanded']}")
                print(f"  Graph build time: {performance['build_time']:.4f}s")
                print(f"  Search time: {performance['search_time']:.4f}s")
                print(f"  Heuristic computation time: {performance['heuristic_time']:.4f}s")
                
                # Display the plan
                print(f"\n{self.display_plan(solution)}")
                
                # Validate the plan
                is_valid, validation_msg = self.validate_plan(solution)
                if is_valid:
                    print(f"  ✓ Plan validation: {validation_msg}")
                else:
                    print(f"  ✗ Plan validation: {validation_msg}")
                
            else:
                print(f"✗ {performance['status']}")
                print(f"  Nodes expanded: {performance['nodes_expanded']}")
                print(f"  Time spent: {performance['total_time']:.4f}s")
        
        # Summary comparison
        print(f"\n{'='*80}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        
        successful_results = {k: v for k, v in results.items() if v[0] is not None}
        
        if successful_results:
            print(f"{'Heuristic':<15} {'Time':<8} {'Length':<8} {'Nodes':<8} {'H-Time':<8} {'Valid':<6}")
            print("-" * 65)
            
            for heuristic_name, (solution, perf) in successful_results.items():
                is_valid, _ = self.validate_plan(solution)
                valid_str = "✓" if is_valid else "✗"
                print(f"{heuristic_name:<15} {perf['total_time']:<8.3f} "
                      f"{perf['solution_length']:<8} {perf['nodes_expanded']:<8} "
                      f"{perf['heuristic_time']:<8.3f} {valid_str:<6}")
            
            # Find best performers
            fastest = min(successful_results.keys(), 
                         key=lambda k: successful_results[k][1]['total_time'])
            shortest = min(successful_results.keys(), 
                          key=lambda k: successful_results[k][1]['solution_length'])
            most_efficient = min(successful_results.keys(), 
                               key=lambda k: successful_results[k][1]['nodes_expanded'])
            
            print(f"\nBest performers:")
            print(f"  Fastest: {fastest} ({successful_results[fastest][1]['total_time']:.3f}s)")
            print(f"  Shortest solution: {shortest} ({successful_results[shortest][1]['solution_length']} steps)")
            print(f"  Most efficient search: {most_efficient} ({successful_results[most_efficient][1]['nodes_expanded']} nodes)")
        else:
            print("No heuristic found a solution within the node limit.")
        
        return results

def create_blocks_world_actions(num_blocks=6):
    """Create actions for blocks world domain with variable number of blocks"""
    blocks = [chr(ord('A') + i) for i in range(num_blocks)]
    actions = []
    
    # Stack(x, y): put block x on block y
    for x in blocks:
        for y in blocks:
            if x != y:
                preconditions = [f'clear({x})', f'clear({y})', f'ontable({x})']
                effects = [f'on({x},{y})', f'not_clear({y})', f'not_ontable({x})']
                actions.append(Action(f'stack({x},{y})', preconditions, effects))
    
    # Unstack(x, y): take block x off block y
    for x in blocks:
        for y in blocks:
            if x != y:
                preconditions = [f'clear({x})', f'on({x},{y})']
                effects = [f'not_on({x},{y})', f'clear({y})', f'ontable({x})']
                actions.append(Action(f'unstack({x},{y})', preconditions, effects))
    
    return actions

def run_comprehensive_tests():
    """Run comprehensive tests comparing all heuristics"""
    
    # Test Case 1: Simple 3-block tower building
    print("TEST CASE 1: Simple 3-Block Tower Building")
    print("=" * 50)
    actions1 = create_blocks_world_actions(3)
    initial_state1 = ['clear(A)', 'clear(B)', 'clear(C)', 'ontable(A)', 'ontable(B)', 'ontable(C)']
    goal_state1 = ['on(A,B)', 'on(B,C)', 'ontable(C)', 'clear(A)']
    
    planner1 = EnhancedGraphPlan(initial_state1, goal_state1, actions1)
    results1 = planner1.compare_heuristics()
    
    # Test Case 2: 4-block rearrangement
    print("\n\nTEST CASE 2: 4-Block Rearrangement")
    print("=" * 50)
    actions2 = create_blocks_world_actions(4)
    initial_state2 = ['clear(A)', 'on(A,B)', 'on(B,C)', 'on(C,D)', 'ontable(D)']
    goal_state2 = ['clear(D)', 'on(D,C)', 'on(C,B)', 'on(B,A)', 'ontable(A)']
    
    planner2 = EnhancedGraphPlan(initial_state2, goal_state2, actions2)
    results2 = planner2.compare_heuristics()
    
    # Test Case 3: Multiple towers
    print("\n\nTEST CASE 3: Multiple Towers")
    print("=" * 50)
    actions3 = create_blocks_world_actions(4)
    initial_state3 = ['clear(A)', 'clear(B)', 'clear(C)', 'clear(D)', 'ontable(A)', 'ontable(B)', 'ontable(C)', 'ontable(D)']
    goal_state3 = ['on(A,B)', 'on(C,D)', 'ontable(B)', 'ontable(D)', 'clear(A)', 'clear(C)']
    planner3 = EnhancedGraphPlan(initial_state3, goal_state3, actions3)
    results3 = planner3.compare_heuristics()

    # Overall Analysis
    print("\n\n" + "="*60)
    print("OVERALL ANALYSIS")
    print("="*60)
    
    # Analyze which heuristic works best for each test case
    test_cases = [
        ("Simple Tower", results1),
        ("Complex Rearrangement", results2),
        ("Parallel Goals", results3)
    ]
    
    for test_name, results in test_cases:
        print(f"\n{test_name}:")
        successful = {k: v for k, v in results.items() if v[0] is not None}
        
        if successful:
            best_time = min(successful.keys(), key=lambda k: successful[k][1]['total_time'])
            best_nodes = min(successful.keys(), key=lambda k: successful[k][1]['nodes_expanded'])
            best_length = min(successful.keys(), key=lambda k: successful[k][1]['solution_length'])
            
            print(f"  Best time: {best_time} ({successful[best_time][1]['total_time']:.3f}s)")
            print(f"  Best efficiency: {best_nodes} ({successful[best_nodes][1]['nodes_expanded']} nodes)")
            print(f"  Best solution: {best_length} ({successful[best_length][1]['solution_length']} steps)")

if __name__ == "__main__":
    run_comprehensive_tests()