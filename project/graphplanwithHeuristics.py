import heapq
from collections import defaultdict

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
    
    def is_applicable(self, state):
        return self.preconditions.issubset(state)

class GraphPlan:
    def __init__(self, initial_state, goal_state, actions):
        self.initial_state = set(initial_state)
        self.goal_state = set(goal_state)
        self.actions = actions
        self.proposition_levels = []
        self.action_levels = []
        self.proposition_mutexes = []
        self.action_mutexes = []
        self.max_levels = 20
        
        # Heuristic data structures
        self.goal_difficulty = {}
        self.action_costs = {}
        self.proposition_first_appearance = {}
        
    def build_graph(self):
        # Level 0: Initial state
        self.proposition_levels.append(self.initial_state.copy())
        self.proposition_mutexes.append(set())
        
        # Initialize heuristics
        self.initialize_heuristics()
        
        level = 0
        while level < self.max_levels:
            # Update heuristics for this level
            self.update_heuristics(level)
            
            # Check if goal is reachable and non-mutex
            if self.goal_state.issubset(self.proposition_levels[level]):
                if not self.goals_are_mutex(level):
                    return level
            
            # Find applicable actions
            applicable_actions = []
            for action in self.actions:
                if action.is_applicable(self.proposition_levels[level]):
                    applicable_actions.append(action)
            
            # Add no-op actions
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
            if level > 1 and self.proposition_levels[level] == self.proposition_levels[level-1]:
                if self.proposition_mutexes[level] == self.proposition_mutexes[level-1]:
                    break
        
        return -1
    
    def initialize_heuristics(self):
        """Initialize heuristic data structures"""
        # Calculate basic action costs
        for action in self.actions:
            self.action_costs[action.name] = len(action.preconditions) + len(action.effects)
        
        # Initialize goal difficulty based on how many actions can achieve each goal
        for goal in self.goal_state:
            achieving_actions = [a for a in self.actions if goal in a.add_effects]
            self.goal_difficulty[goal] = 1.0 / max(len(achieving_actions), 1)
    
    def update_heuristics(self, level):
        """Update heuristics for current level"""
        # Track when propositions first appear
        for prop in self.proposition_levels[level]:
            if prop not in self.proposition_first_appearance:
                self.proposition_first_appearance[prop] = level
    
    def calculate_goal_priority(self, goals):
        """Calculate priority for a set of goals (lower is better)"""
        priority = 0
        for goal in goals:
            # Factor 1: Goal difficulty
            priority += self.goal_difficulty.get(goal, 1.0)
            
            # Factor 2: How late the goal first appeared
            first_level = self.proposition_first_appearance.get(goal, 0)
            priority += first_level * 0.1
            
            # Factor 3: Number of mutex relationships
            mutex_count = sum(1 for other_goal in goals 
                            if goal != other_goal and self.goals_are_mutex_pair(goal, other_goal))
            priority += mutex_count * 0.5
        
        return priority
    
    def goals_are_mutex_pair(self, goal1, goal2):
        """Check if two goals are mutex at any level"""
        for level in range(len(self.proposition_mutexes)):
            if self.are_mutex_at_level(goal1, goal2, level):
                return True
        return False
    
    def get_action_heuristic_score(self, action, goals, level):
        """Calculate heuristic score for an action (lower is better)"""
        score = 0
        
        # Factor 1: Base action cost
        score += self.action_costs.get(action.name, 1)
        
        # Factor 2: Number of goals achieved
        goals_achieved = len(action.add_effects & goals)
        score -= goals_achieved * 2  # Prefer actions that achieve more goals
        
        # Factor 3: Precondition difficulty
        for precond in action.preconditions:
            if precond not in self.initial_state:
                score += self.goal_difficulty.get(precond, 0.5)
        
        # Factor 4: Penalty for actions that delete needed propositions
        for del_effect in action.del_effects:
            if del_effect in goals:
                score += 10  # Heavy penalty
        
        return score
    
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
    
    def extract_solution(self, goal_level):
        """Extract solution using A* search with heuristics"""
        if goal_level == -1:
            return None
        
        # Priority queue: (priority, level, goals, partial_solution)
        pq = [(self.calculate_goal_priority(self.goal_state), goal_level, frozenset(self.goal_state), [])]
        visited = set()
        
        while pq:
            _, level, goals, partial_solution = heapq.heappop(pq)
            
            # Base case
            if level == 0:
                if goals.issubset(self.initial_state):
                    return list(reversed(partial_solution))
                continue
            
            # Skip if already visited
            state_key = (level, goals)
            if state_key in visited:
                continue
            visited.add(state_key)
            
            # Generate action combinations using heuristics
            for action_set in self.get_best_action_sets(goals, level):
                if self.is_valid_action_set(action_set, level):
                    # Get preconditions
                    preconditions = set()
                    for action in action_set:
                        preconditions.update(action.preconditions)
                    
                    # Calculate priority for next state
                    priority = self.calculate_goal_priority(preconditions)
                    
                    # Filter out no-ops
                    real_actions = [a for a in action_set if not a.name.startswith('noop_')]
                    
                    # Add to priority queue
                    new_partial = [real_actions] + partial_solution if real_actions else partial_solution
                    heapq.heappush(pq, (priority, level - 1, frozenset(preconditions), new_partial))
        
        return None
    
    def get_best_action_sets(self, goals, level, max_sets=5):
        """Get the best action sets using heuristics"""
        if level == 0:
            return
        
        # Find actions that achieve each goal, sorted by heuristic score
        goal_achievers = {}
        for goal in goals:
            achievers = []
            for action in self.action_levels[level - 1]:
                if goal in action.add_effects:
                    score = self.get_action_heuristic_score(action, goals, level)
                    achievers.append((score, action))
            
            # Sort by heuristic score (lower is better)
            achievers.sort(key=lambda x: x[0])
            goal_achievers[goal] = [action for _, action in achievers[:3]]  # Top 3 for each goal
        
        # Generate action sets using best-first search
        generated_sets = []
        
        def generate_with_heuristics(remaining_goals, current_actions, achieved_so_far):
            if len(generated_sets) >= max_sets:
                return
            
            if not remaining_goals:
                generated_sets.append(list(current_actions))
                return
            
            # Choose the most difficult goal first
            goal = max(remaining_goals, key=lambda g: self.goal_difficulty.get(g, 1.0))
            
            for action in goal_achievers[goal]:
                # Check conflicts
                conflicts = False
                for existing_action in current_actions:
                    if self.are_mutex_at_level(action, existing_action, level - 1):
                        conflicts = True
                        break
                
                if not conflicts:
                    new_achieved = achieved_so_far | action.add_effects
                    new_remaining = remaining_goals - new_achieved
                    generate_with_heuristics(new_remaining, current_actions + [action], new_achieved)
        
        generate_with_heuristics(goals, [], set())
        return generated_sets
    
    def is_valid_action_set(self, action_set, level):
        """Check if an action set is valid (no mutex pairs)"""
        for i, a1 in enumerate(action_set):
            for j, a2 in enumerate(action_set):
                if i < j and self.are_mutex_at_level(a1, a2, level - 1):
                    return False
        return True
    
    def solve(self):
        print("Building planning graph with heuristics...")
        goal_level = self.build_graph()
        
        if goal_level == -1:
            print("Goal is not reachable.")
            return None
        
        print(f"Goal reachable at level {goal_level}")
        print("Extracting solution using A* search...")
        solution = self.extract_solution(goal_level)
        
        if solution:
            print("Solution found:")
            for i, level_actions in enumerate(solution):
                print(f"Level {i+1}: {[str(a) for a in level_actions]}")
            return solution
        else:
            print("No valid plan found.")
            return None

def create_blocks_world_actions():
    """Create actions for blocks world domain"""
    blocks = ['A', 'B', 'C']
    actions = []
    
    # Stack(x, y): put block x on block y (from table)
    for x in blocks:
        for y in blocks:
            if x != y:
                preconditions = [f'clear({x})', f'clear({y})', f'ontable({x})']
                effects = [f'on({x},{y})', f'not_clear({y})', f'not_ontable({x})']
                actions.append(Action(f'stack({x},{y})', preconditions, effects))
    
    # Unstack(x, y): take block x off block y (put on table)
    for x in blocks:
        for y in blocks:
            if x != y:
                preconditions = [f'clear({x})', f'on({x},{y})']
                effects = [f'not_on({x},{y})', f'clear({y})', f'ontable({x})']
                actions.append(Action(f'unstack({x},{y})', preconditions, effects))
    
    # Move(x, y, z): move block x from y to z
    for x in blocks:
        for y in blocks:
            for z in blocks:
                if x != y and x != z and y != z:
                    preconditions = [f'clear({x})', f'clear({z})', f'on({x},{y})']
                    effects = [f'on({x},{z})', f'not_on({x},{y})', f'clear({y})', f'not_clear({z})']
                    actions.append(Action(f'move({x},{y},{z})', preconditions, effects))
    
    return actions

def test_blocks_world():
    """Test GraphPlan on blocks world problems"""
    actions = create_blocks_world_actions()
    
    print("Running GraphPlan with heuristics for blocks world...")
    
    # Test case 1: All blocks on table
    print("\n=== TEST CASE 1: All blocks on table ===")
    initial_state = ['clear(A)', 'clear(B)', 'clear(C)', 'ontable(A)', 'ontable(B)', 'ontable(C)']
    goal_state = ['clear(A)', 'on(A,B)', 'on(B,C)', 'ontable(C)']
    
    planner = GraphPlan(initial_state, goal_state, actions)
    solution1 = planner.solve()
    
    # Test case 2: A on B initially
    print("\n=== TEST CASE 2: A on B initially ===")
    initial_state = ['clear(A)', 'clear(C)', 'on(A,B)', 'ontable(B)', 'ontable(C)']
    goal_state = ['clear(A)', 'on(A,B)', 'on(B,C)', 'ontable(C)']
    
    planner = GraphPlan(initial_state, goal_state, actions)
    solution2 = planner.solve()
    
    # Test case 3: Complex rearrangement
    print("\n=== TEST CASE 3: Complex rearrangement ===")
    initial_state = ['clear(A)', 'on(A,B)', 'on(B,C)', 'ontable(C)']
    goal_state = ['clear(B)', 'on(A,C)', 'on(B,A)', 'ontable(C)']
    
    planner = GraphPlan(initial_state, goal_state, actions)
    solution3 = planner.solve()
    
    # Summary
    print("\n=== SUMMARY ===")
    results = []
    results.append(("Test 1 (all on table)", "PASSED" if solution1 else "FAILED"))
    results.append(("Test 2 (A on B)", "PASSED" if solution2 else "FAILED"))
    results.append(("Test 3 (complex)", "PASSED" if solution3 else "FAILED"))
    
    for test_name, result in results:
        print(f"{test_name}: {result}")
    
    if all(result == "PASSED" for _, result in results):
        print("All tests passed! ✓")
    else:
        print("Some tests failed. ✗")

if __name__ == "__main__":
    test_blocks_world()