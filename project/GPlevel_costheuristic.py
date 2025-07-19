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
        return self.name < other.name if isinstance(other, Action) else False
    
    def is_applicable(self, state):
        return self.preconditions.issubset(state)
    
    def apply(self, state):
        """Apply action to state, handling negative effects properly"""
        new_state = state.copy()
        new_state.update(self.add_effects)
        new_state.difference_update(self.del_effects)
        return new_state

class LevelCostHeuristic:
    """
    Level Cost Heuristic: Sum of levels where goals first appear
    """
    def __init__(self):
        self.name = "Level Cost"
        self.computation_time = 0
        self.evaluations = []  # Track heuristic evaluations
    
    def estimate(self, state, goals, planning_graph):
        """Calculate heuristic value for given state and goals"""
        start_time = time.time()
        
        cost = 0
        goal_details = {}
        
        for goal in goals:
            # Find first level where goal appears
            level = 0
            found = False
            for level, prop_level in enumerate(planning_graph.proposition_levels):
                if goal in prop_level:
                    cost += level
                    goal_details[goal] = level
                    found = True
                    break
            
            if not found:
                unreachable_cost = len(planning_graph.proposition_levels) * 2
                cost += unreachable_cost
                goal_details[goal] = f"unreachable (cost: {unreachable_cost})"
        
        # Record this evaluation
        evaluation = {
            'goals': goals,
            'cost': cost,
            'goal_details': goal_details,
            'time': time.time() - start_time
        }
        self.evaluations.append(evaluation)
        
        self.computation_time += time.time() - start_time
        return cost

class SearchNode:
    """Represents a node in the search tree for better tracking"""
    def __init__(self, level, goals, g_score, h_score, partial_solution, parent=None):
        self.level = level
        self.goals = goals
        self.g_score = g_score
        self.h_score = h_score
        self.f_score = g_score + h_score
        self.partial_solution = partial_solution
        self.parent = parent
        self.id = id(self)
    
    def __lt__(self, other):
        return self.f_score < other.f_score

class GraphPlanWithLevelCost:
    def __init__(self, initial_state, goal_state, actions):
        self.initial_state = set(initial_state)
        self.goal_state = set(goal_state)
        self.actions = actions
        self.proposition_levels = []
        self.action_levels = []
        self.proposition_mutexes = []
        self.action_mutexes = []
        self.max_levels = 25
        self.counter = 0
        
        # Heuristic and search tracking
        self.heuristic = LevelCostHeuristic()
        self.search_tree = []  # Track search nodes
        self.node_comparisons = []  # Track node comparisons
        self.solution_path = []  # Track solution path
        
        # Debug info
        print(f"Initial state: {sorted(self.initial_state)}")
        print(f"Goal state: {sorted(self.goal_state)}")
        print(f"Actions: {len(self.actions)}")
    
    def build_graph(self):
        """Build the planning graph"""
        print("Building planning graph...")
        
        # Level 0: Initial state
        self.proposition_levels.append(self.initial_state.copy())
        self.proposition_mutexes.append(set())
        
        level = 0
        while level < self.max_levels:
            print(f"Building level {level}...")
            print(f"  Propositions at level {level}: {len(self.proposition_levels[level])}")
            
            # Check if goal is reachable and non-mutex
            if self.goal_state.issubset(self.proposition_levels[level]):
                if not self.goals_are_mutex(level):
                    print(f"Goal found at level {level}")
                    return level
                else:
                    print(f"Goal present but mutex at level {level}")
            
            # Find applicable actions
            applicable_actions = []
            for action in self.actions:
                if action.is_applicable(self.proposition_levels[level]):
                    applicable_actions.append(action)
            
            print(f"  Applicable actions: {len(applicable_actions)}")
            
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
            next_props = self.proposition_levels[level].copy()
            for action in applicable_actions:
                next_props.update(action.add_effects)
                next_props.difference_update(action.del_effects)
            
            self.proposition_levels.append(next_props)
            print(f"  Next level will have {len(next_props)} propositions")
            
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
                    print(f"Fixed point reached at level {level}")
                    if self.goal_state.issubset(self.proposition_levels[level]):
                        if not self.goals_are_mutex(level):
                            print(f"Goal found at fixed point level {level}")
                            return level
                    break
        
        print("No solution found - goal not reachable")
        return -1
    
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
            
        if level-1 >= len(self.action_levels):
            return False
            
        p1_actions = [a for a in self.action_levels[level-1] if p1 in a.add_effects]
        p2_actions = [a for a in self.action_levels[level-1] if p2 in a.add_effects]
        
        if not p1_actions or not p2_actions:
            return False
        
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
    
    def solve(self, max_nodes=10000):
        """Solve using A* with Level Cost heuristic"""
        start_time = time.time()
        
        # Reset tracking
        self.heuristic.evaluations = []
        self.search_tree = []
        self.node_comparisons = []
        self.solution_path = []
        
        # Build planning graph
        goal_level = self.build_graph()
        build_time = time.time() - start_time
        
        if goal_level == -1:
            return None, {
                'build_time': build_time,
                'search_time': 0,
                'total_time': build_time,
                'heuristic_time': self.heuristic.computation_time,
                'nodes_expanded': 0,
                'solution_length': 0,
                'status': 'No solution found'
            }
        
        # Extract solution using A* with Level Cost heuristic
        solution, search_stats = self.extract_solution_with_tracking(goal_level, max_nodes)
        
        total_time = time.time() - start_time
        
        performance = {
            'build_time': build_time,
            'search_time': search_stats['search_time'],
            'total_time': total_time,
            'heuristic_time': self.heuristic.computation_time,
            'nodes_expanded': search_stats['nodes_expanded'],
            'solution_length': len(solution) if solution else 0,
            'status': search_stats.get('status', 'Success' if solution else 'Failed')
        }
        
        return solution, performance
    
    def extract_solution_with_tracking(self, goal_level, max_nodes):
        """Extract solution with detailed tracking of heuristic guidance"""
        start_time = time.time()
        nodes_expanded = 0
        
        # Create initial node
        initial_h = self.heuristic.estimate(self.goal_state, self.goal_state, self)
        initial_node = SearchNode(goal_level, frozenset(self.goal_state), 0, initial_h, [])
        
        # Priority queue: (f_score, counter, node)
        pq = [(initial_node.f_score, 0, initial_node)]
        visited = set()
        
        print(f"Starting A* search with Level Cost heuristic...")
        print(f"Initial node: level={goal_level}, goals={sorted(self.goal_state)}, h={initial_h}")
        
        while pq and nodes_expanded < max_nodes:
            f_score, counter, current_node = heapq.heappop(pq)
            nodes_expanded += 1
            
            # Track this node
            self.search_tree.append({
                'node_id': current_node.id,
                'level': current_node.level,
                'goals': current_node.goals,
                'g_score': current_node.g_score,
                'h_score': current_node.h_score,
                'f_score': current_node.f_score,
                'expanded_order': nodes_expanded
            })
            
            if nodes_expanded % 100 == 0:
                print(f"Expanded {nodes_expanded} nodes, current: level={current_node.level}, "
                      f"g={current_node.g_score}, h={current_node.h_score}, f={current_node.f_score}")
            
            # Base case
            if current_node.level == 0:
                if current_node.goals.issubset(self.initial_state):
                    search_time = time.time() - start_time
                    print(f"Solution found! Nodes expanded: {nodes_expanded}")
                    
                    # Reconstruct solution path
                    self.solution_path = self.reconstruct_solution_path(current_node)
                    
                    return list(reversed(current_node.partial_solution)), {
                        'search_time': search_time,
                        'nodes_expanded': nodes_expanded,
                        'status': 'Success'
                    }
                continue
            
            # Skip if already visited
            state_key = (current_node.level, current_node.goals)
            if state_key in visited:
                continue
            visited.add(state_key)
            
            # Generate successor nodes
            successors = self.generate_successors(current_node)
            
            # Track node comparisons
            for successor in successors:
                comparison = {
                    'parent_f': current_node.f_score,
                    'child_f': successor.f_score,
                    'parent_goals': current_node.goals,
                    'child_goals': successor.goals,
                    'better_choice': successor.f_score < current_node.f_score
                }
                self.node_comparisons.append(comparison)
            
            # Add successors to queue
            for successor in successors:
                self.counter += 1
                heapq.heappush(pq, (successor.f_score, self.counter, successor))
        
        search_time = time.time() - start_time
        status = 'Node limit reached' if nodes_expanded >= max_nodes else 'No solution found'
        return None, {
            'search_time': search_time,
            'nodes_expanded': nodes_expanded,
            'status': status
        }
    
    def generate_successors(self, node):
        """Generate successor nodes for a given node"""
        successors = []
        
        # Generate action sets
        action_sets = self.get_action_sets(node.goals, node.level)
        
        for action_set in action_sets:
            if self.is_valid_action_set(action_set, node.level):
                # Get preconditions
                preconditions = set()
                for action in action_set:
                    preconditions.update(action.preconditions)
                
                # Calculate scores
                new_g_score = node.g_score + 1
                h_score = self.heuristic.estimate(preconditions, preconditions, self)
                
                # Filter out no-ops
                real_actions = [a for a in action_set if not a.name.startswith('noop_')]
                
                # Create new node
                new_partial = [real_actions] + node.partial_solution if real_actions else node.partial_solution
                successor = SearchNode(
                    node.level - 1,
                    frozenset(preconditions),
                    new_g_score,
                    h_score,
                    new_partial,
                    node
                )
                
                successors.append(successor)
        
        return successors
    
    def reconstruct_solution_path(self, final_node):
        """Reconstruct the path from initial node to solution"""
        path = []
        current = final_node
        while current is not None:
            path.append({
                'level': current.level,
                'goals': current.goals,
                'g_score': current.g_score,
                'h_score': current.h_score,
                'f_score': current.f_score
            })
            current = current.parent
        return list(reversed(path))
    
    def get_action_sets(self, goals, level, max_sets=30):
        """Generate valid action sets for goals"""
        if level == 0 or level-1 >= len(self.action_levels):
            return []
        
        action_sets = []
        goals_list = list(goals)
        
        # Find all actions that can achieve each goal
        goal_actions = defaultdict(list)
        for goal in goals:
            for action in self.action_levels[level - 1]:
                if goal in action.add_effects:
                    goal_actions[goal].append(action)
        
        # Try single actions first
        for goal in goals_list:
            for action in goal_actions[goal]:
                if set([goal]).issubset(action.add_effects):
                    action_sets.append([action])
                    if len(action_sets) >= max_sets:
                        return action_sets
        
        # Try pairs of actions
        all_actions = []
        for goal in goals_list:
            all_actions.extend(goal_actions[goal])
        all_actions = list(set(all_actions))
        
        for r in range(2, min(len(all_actions), 4) + 1):
            for action_combo in itertools.combinations(all_actions, r):
                achieved = set()
                for action in action_combo:
                    achieved.update(action.add_effects)
                
                if goals.issubset(achieved):
                    action_sets.append(list(action_combo))
                    if len(action_sets) >= max_sets:
                        return action_sets
        
        return action_sets
    
    def is_valid_action_set(self, action_set, level):
        """Check if action set is valid (no mutex pairs)"""
        if level-1 >= len(self.action_mutexes):
            return False
            
        for i, a1 in enumerate(action_set):
            for j, a2 in enumerate(action_set):
                if i < j and self.are_mutex_at_level(a1, a2, level - 1):
                    return False
        return True
    
    def display_heuristic_guidance_analysis(self):
        """Show how the heuristic guided the search"""
        print(f"\nHEURISTIC GUIDANCE ANALYSIS")
        print(f"=" * 60)
        
        # Show initial heuristic evaluation
        print(f"Initial Goals: {sorted(self.goal_state)}")
        initial_eval = self.heuristic.evaluations[0] if self.heuristic.evaluations else None
        if initial_eval:
            print(f"Initial Heuristic Value: {initial_eval['cost']}")
            print(f"Goal Level Details:")
            for goal, level in initial_eval['goal_details'].items():
                print(f"  {goal}: level {level}")
        
        # Show search progression
        print(f"\nSearch Tree Progression (first 10 nodes):")
        print(f"{'Order':<6} {'Level':<6} {'G':<4} {'H':<4} {'F':<4} {'Goals'}")
        print("-" * 60)
        
        for i, node in enumerate(self.search_tree[:10]):
            goals_str = str(sorted(node['goals']))[:40] + "..." if len(str(sorted(node['goals']))) > 40 else str(sorted(node['goals']))
            print(f"{node['expanded_order']:<6} {node['level']:<6} {node['g_score']:<4} {node['h_score']:<4} {node['f_score']:<4} {goals_str}")
        
        if len(self.search_tree) > 10:
            print(f"... and {len(self.search_tree) - 10} more nodes")
        
        # Show solution path
        if self.solution_path:
            print(f"\nSolution Path:")
            print(f"{'Level':<6} {'G':<4} {'H':<4} {'F':<4} {'Goals'}")
            print("-" * 50)
            for step in self.solution_path:
                goals_str = str(sorted(step['goals']))[:40] + "..." if len(str(sorted(step['goals']))) > 40 else str(sorted(step['goals']))
                print(f"{step['level']:<6} {step['g_score']:<4} {step['h_score']:<4} {step['f_score']:<4} {goals_str}")
        
        # Analyze heuristic effectiveness
        print(f"\nHeuristic Effectiveness Analysis:")
        print(f"Total heuristic evaluations: {len(self.heuristic.evaluations)}")
        print(f"Average heuristic computation time: {self.heuristic.computation_time/len(self.heuristic.evaluations)*1000:.3f}ms per evaluation")
        
        # Show how heuristic values changed
        if len(self.heuristic.evaluations) > 1:
            h_values = [eval['cost'] for eval in self.heuristic.evaluations]
            print(f"Heuristic value range: {min(h_values)} to {max(h_values)}")
            print(f"Final heuristic value: {h_values[-1] if self.solution_path else 'N/A'}")
    
    def display_node_selection_analysis(self):
        """Show how the heuristic influenced node selection"""
        print(f"\nNODE SELECTION ANALYSIS")
        print(f"=" * 50)
        
        # Analyze f-score distribution
        f_scores = [node['f_score'] for node in self.search_tree]
        if f_scores:
            print(f"F-score distribution:")
            print(f"  Min: {min(f_scores)}, Max: {max(f_scores)}, Avg: {sum(f_scores)/len(f_scores):.2f}")
        
        # Show examples of good vs bad heuristic guidance
        print(f"\nExamples of Heuristic Guidance:")
        
        # Find nodes with low vs high heuristic values
        sorted_nodes = sorted(self.search_tree, key=lambda x: x['h_score'])
        
        print(f"\nNodes with LOWEST heuristic values (closer to goal):")
        for node in sorted_nodes[:3]:
            goals_str = str(sorted(node['goals']))[:50] + "..." if len(str(sorted(node['goals']))) > 50 else str(sorted(node['goals']))
            print(f"  Level {node['level']}: h={node['h_score']}, goals={goals_str}")
        
        print(f"\nNodes with HIGHEST heuristic values (further from goal):")
        for node in sorted_nodes[-3:]:
            goals_str = str(sorted(node['goals']))[:50] + "..." if len(str(sorted(node['goals']))) > 50 else str(sorted(node['goals']))
            print(f"  Level {node['level']}: h={node['h_score']}, goals={goals_str}")
    
    def display_plan(self, solution):
        """Display the plan in a readable format"""
        if not solution:
            return "No solution found"
        
        plan_str = []
        plan_str.append(f"Plan with Level Cost Heuristic ({len(solution)} steps):")
        for i, step in enumerate(solution):
            if isinstance(step, list):
                if len(step) == 1:
                    plan_str.append(f"  Step {i+1}: {step[0]}")
                else:
                    plan_str.append(f"  Step {i+1}: [Parallel] {' | '.join(str(a) for a in step)}")
            else:
                plan_str.append(f"  Step {i+1}: {step}")
        
        return "\n".join(plan_str)

def create_blocks_world_actions(blocks):
    """Create comprehensive blocks world actions for given blocks"""
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

def create_simple_state(blocks, stacks, table_blocks=None):
    """Create a blocks world state more carefully"""
    state = []
    
    if table_blocks is None:
        table_blocks = []
    
    all_blocks = set(blocks)
    blocks_in_stacks = set()
    
    # Process stacks
    for stack in stacks:
        if len(stack) == 0:
            continue
            
        # Add on relationships
        for i in range(len(stack) - 1):
            state.append(f'on({stack[i]},{stack[i+1]})')
            blocks_in_stacks.add(stack[i])
            blocks_in_stacks.add(stack[i+1])
        
        # Bottom block is on table
        state.append(f'ontable({stack[-1]})')
        blocks_in_stacks.add(stack[-1])
        
        # Top block is clear
        state.append(f'clear({stack[0]})')
        blocks_in_stacks.add(stack[0])
    
    # Add table blocks
    for block in table_blocks:
        state.append(f'ontable({block})')
        state.append(f'clear({block})')
        blocks_in_stacks.add(block)
    
    # Any remaining blocks go on table
    remaining = all_blocks - blocks_in_stacks
    for block in remaining:
        state.append(f'ontable({block})')
        state.append(f'clear({block})')
    
    return state

def test_heuristic_guidance():
    """Test to show how Level Cost heuristic guides the search"""
    print("TESTING HEURISTIC GUIDANCE IN GRAPHPLAN")
    print("=" * 70)
    
    # Test Case: Simple A on B problem
    print("\n=== TEST CASE: Simple A on B ===")
    blocks = ['A', 'B']
    actions = create_blocks_world_actions(blocks)
    
    # Initial: all blocks on table
    initial_state = create_simple_state(blocks, [], ['A', 'B'])
    
    # Goal: A on B
    goal_state = create_simple_state(blocks, [['A', 'B']], [])
    
    print(f"Initial state: {initial_state}")
    print(f"Goal state: {goal_state}")
    
    planner = GraphPlanWithLevelCost(initial_state, goal_state, actions)
    solution, performance = planner.solve()
    
    print(f"\nRESULTS:")
    print(f"Status: {performance['status']}")
    print(f"Total time: {performance['total_time']:.4f}s")
    print(f"Solution length: {performance['solution_length']}")
    
    if solution:
        print(f"\n{planner.display_plan(solution)}")
    
    # Show detailed heuristic analysis
    planner.display_heuristic_guidance_analysis()
    planner.display_node_selection_analysis()
    
    return planner

# Run the test
if __name__ == "__main__":
    print("Starting Heuristic Guidance Analysis...")
    planner = test_heuristic_guidance()
    print("\nHeuristic guidance analysis completed!")