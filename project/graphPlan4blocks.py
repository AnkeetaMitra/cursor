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
        
    def build_graph(self):
        # Level 0: Initial state
        self.proposition_levels.append(self.initial_state.copy())
        self.proposition_mutexes.append(set())
        
        level = 0
        while level < self.max_levels:
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
        
        return -1  # Goal not reachable
    
    def actions_are_mutex(self, a1, a2, level):
        # Same action (but different instances)
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
        # All ways to achieve p1 are mutex with all ways to achieve p2
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
        """Extract solution using recursive backtracking"""
        if goal_level == -1:
            return None
        
        memo = {}
        
        def solve_goals(goals, level):
            # Base case: level 0
            if level == 0:
                return [] if goals.issubset(self.initial_state) else None
            
            # Memoization
            goals_key = (frozenset(goals), level)
            if goals_key in memo:
                return memo[goals_key]
            
            # Try to find a valid action set for this level
            for action_set in self.enumerate_action_sets(goals, level):
                if self.is_valid_action_set(action_set, level):
                    # Get preconditions
                    preconditions = set()
                    for action in action_set:
                        preconditions.update(action.preconditions)
                    
                    # Recursively solve preconditions
                    sub_solution = solve_goals(preconditions, level - 1)
                    if sub_solution is not None:
                        # Filter out no-ops
                        real_actions = [a for a in action_set if not a.name.startswith('noop_')]
                        if real_actions:
                            result = sub_solution + [real_actions]
                        else:
                            result = sub_solution
                        memo[goals_key] = result
                        return result
            
            memo[goals_key] = None
            return None
        
        return solve_goals(self.goal_state, goal_level)
    
    def enumerate_action_sets(self, goals, level):
        """Enumerate possible action sets that achieve the goals"""
        if level == 0:
            return
        
        # Find actions that achieve each goal
        goal_achievers = {}
        for goal in goals:
            goal_achievers[goal] = []
            for action in self.action_levels[level - 1]:
                if goal in action.add_effects:
                    goal_achievers[goal].append(action)
        
        # Use backtracking to find valid combinations
        def backtrack(remaining_goals, current_actions, achieved_so_far):
            if not remaining_goals:
                yield list(current_actions)
                return
            
            goal = next(iter(remaining_goals))
            for action in goal_achievers[goal]:
                # Check if this action conflicts with already chosen actions
                conflicts = False
                for existing_action in current_actions:
                    if self.are_mutex_at_level(action, existing_action, level - 1):
                        conflicts = True
                        break
                
                if not conflicts:
                    new_achieved = achieved_so_far | action.add_effects
                    new_remaining = remaining_goals - new_achieved
                    yield from backtrack(new_remaining, current_actions + [action], new_achieved)
        
        yield from backtrack(goals, [], set())
    
    def is_valid_action_set(self, action_set, level):
        """Check if an action set is valid (no mutex pairs)"""
        for i, a1 in enumerate(action_set):
            for j, a2 in enumerate(action_set):
                if i < j and self.are_mutex_at_level(a1, a2, level - 1):
                    return False
        return True
    
    def solve(self):
        print("Building planning graph...")
        goal_level = self.build_graph()
        
        if goal_level == -1:
            print("Goal is not reachable.")
            return None
        
        print(f"Goal reachable at level {goal_level}")
        print("Extracting solution...")
        solution = self.extract_solution(goal_level)
        
        if solution:
            print("Solution found:")
            for i, level_actions in enumerate(solution):
                print(f"Level {i+1}: {[str(a) for a in level_actions]}")
            return solution
        else:
            print("No valid plan found.")
            return None

def create_blocks_world_actions(blocks=None):
    """Create actions for blocks world domain"""
    if blocks is None:
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
    
    print("Running GraphPlan tests for blocks world...")
    
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
    
    # Test case 4: Four blocks test
    print("\n=== TEST CASE 4: Four blocks ===")
    blocks_4 = ['A', 'B', 'C', 'D']
    actions_4 = create_blocks_world_actions(blocks_4)
    
    initial_state = ['clear(A)', 'clear(B)', 'clear(C)', 'clear(D)', 
                    'ontable(A)', 'ontable(B)', 'ontable(C)', 'ontable(D)']
    goal_state = ['clear(A)', 'on(A,B)', 'on(B,C)', 'on(C,D)', 'ontable(D)']
    
    planner = GraphPlan(initial_state, goal_state, actions_4)
    solution4 = planner.solve()

    # Test case 5: Four blocks test
    print("\n=== TEST CASE 5: Four blocks ===")
    blocks_4 = ['A', 'B', 'C', 'D']
    actions_4 = create_blocks_world_actions(blocks_4)
    
    initial_state = ['clear(B)', 'clear(D)', 'on(B,A)', 'on(D,C)',
                    'ontable(A)', 'ontable(C)']
    goal_state = ['clear(A)', 'on(A,B)', 'on(B,C)', 'on(C,D)', 'ontable(D)']
    
    planner = GraphPlan(initial_state, goal_state, actions_4)
    solution5 = planner.solve()
    
    # Summary
    print("\n=== SUMMARY ===")
    results = []
    results.append(("Test 1 (all on table)", "PASSED" if solution1 else "FAILED"))
    results.append(("Test 2 (A on B)", "PASSED" if solution2 else "FAILED"))
    results.append(("Test 3 (complex)", "PASSED" if solution3 else "FAILED"))
    results.append(("Test 4 (four blocks)", "PASSED" if solution4 else "FAILED"))
    results.append(("Test 5 (four blocks)", "PASSED" if solution5 else "FAILED"))
    
    for test_name, result in results:
        print(f"{test_name}: {result}")
    
    if all(result == "PASSED" for _, result in results):
        print("All tests passed! ✓")
    else:
        print("Some tests failed. ✗")

def test_custom_blocks(blocks, initial_state, goal_state):
    """Test GraphPlan with custom blocks and states"""
    print(f"\n=== CUSTOM TEST: {len(blocks)} blocks ===")
    actions = create_blocks_world_actions(blocks)
    
    print(f"Blocks: {blocks}")
    print(f"Initial: {initial_state}")
    print(f"Goal: {goal_state}")
    
    planner = GraphPlan(initial_state, goal_state, actions)
    solution = planner.solve()
    
    return solution is not None

if __name__ == "__main__":
    test_blocks_world()