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
        # Inconsistent effects
        if a1.add_effects & a2.del_effects or a2.add_effects & a1.del_effects:
            return True
        
        # Interference --check this if required
        if a1.del_effects & a2.add_effects or a2.del_effects & a1.add_effects:
            return True
        
        # Competing needs
        for p1 in a1.preconditions:
            for p2 in a2.preconditions:
                if self.are_mutex_at_level(p1, p2, level):
                    return True
        
        return False
    
    def propositions_are_mutex(self, p1, p2, level):
        # All ways to achieve p1 are mutex with all ways to achieve p2. Finds all actions that can make each fact true
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
        if level < len(self.action_mutexes):
            return (item1, item2) in self.action_mutexes[level] or (item2, item1) in self.action_mutexes[level]
        else:
            return (item1, item2) in self.proposition_mutexes[level] or (item2, item1) in self.proposition_mutexes[level]
    
    def goals_are_mutex(self, level):
        goals = list(self.goal_state)
        for i in range(len(goals)):
            for j in range(i+1, len(goals)):
                if self.are_mutex_at_level(goals[i], goals[j], level):
                    return True
        return False
    
    def extract_solution(self, goal_level):
        if goal_level == -1:
            return None
        
        def search_solution(goals, level):
            if level == 0:
                return [] if goals.issubset(self.initial_state) else None
            
            # Find all possible ways to achieve the goals at this level
            for action_combo in self.get_action_combinations(goals, level):
                # Check if actions are mutex-free
                if self.actions_are_mutex_free(action_combo, level - 1):
                    # Get preconditions for this action combination
                    new_goals = set()
                    for action in action_combo:
                        new_goals.update(action.preconditions)
                    
                    # Recursively solve for preconditions
                    sub_solution = search_solution(new_goals, level - 1)
                    if sub_solution is not None:
                        non_noop_actions = [a for a in action_combo if not a.name.startswith('noop_')]
                        if non_noop_actions:
                            return sub_solution + [non_noop_actions]
                        else:
                            return sub_solution
            
            return None
        
        return search_solution(self.goal_state, goal_level)
    
    def get_action_combinations(self, goals, level):
        """Generate all possible combinations of actions that can achieve the goals"""
        if level == 0:
            return []
        
        # Find actions that achieve each goal
        goal_actions = {}
        for goal in goals:
            goal_actions[goal] = []
            for action in self.action_levels[level - 1]:
                if goal in action.add_effects:
                    goal_actions[goal].append(action)
        
        # Generate all combinations
        def generate_combinations(remaining_goals, current_combo, achieved_goals):
            if not remaining_goals:
                yield current_combo
                return
            
            goal = remaining_goals.pop()
            for action in goal_actions[goal]:
                new_achieved = achieved_goals | action.add_effects
                if goal in new_achieved:
                    new_remaining = remaining_goals - new_achieved
                    yield from generate_combinations(new_remaining, current_combo + [action], new_achieved)
            
            remaining_goals.add(goal)
        
        return generate_combinations(goals.copy(), [], set())
    
    def actions_are_mutex_free(self, actions, level):
        """Check if a set of actions are mutex-free"""
        for i, a1 in enumerate(actions):
            for j, a2 in enumerate(actions):
                if i < j and self.are_mutex_at_level(a1, a2, level):
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

def create_blocks_world_actions():
    """Create actions for blocks world domain"""
    blocks = ['A', 'B', 'C']
    actions = []
    
    # Stack(x, y): put block x on block y
    for x in blocks:
        for y in blocks:
            if x != y:
                preconditions = [f'clear({x})', f'clear({y})']
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

    # test case 4:
    print("\n=== TEST CASE 4: CBA ===")
    initial_state = ['clear(A)', 'clear(C)', 'on(A,B)', 'ontable(B)', 'ontable(C)']
    goal_state = ['clear(C)', 'on(B,A)', 'on(C,B)', 'ontable(A)']
    
    planner = GraphPlan(initial_state, goal_state, actions)
    solution4 = planner.solve()

    # test case 5:
    print("\n=== TEST CASE 5: B->A->C ===")
    initial_state = ['clear(B)', 'clear(A)', 'on(B,C)', 'ontable(C)', 'ontable(A)']
    goal_state = ['clear(B)', 'on(A,C)', 'on(B,A)', 'ontable(C)']
    
    planner = GraphPlan(initial_state, goal_state, actions)
    solution5 = planner.solve()

    # test case 6:
    print("\n=== TEST CASE 6: B->A->C ===")
    initial_state = ['clear(C)', 'clear(A)', 'on(C,B)', 'ontable(B)', 'ontable(A)']
    goal_state = ['clear(C)', 'on(B,A)', 'on(C,B)', 'ontable(A)']
    
    planner = GraphPlan(initial_state, goal_state, actions)
    solution6 = planner.solve()

    # test case 7:
    print("\n=== TEST CASE 7: B->A->C ===")
    initial_state = ['clear(C)', 'on(B,A)', 'on(C,B)', 'ontable(A)']
    goal_state = ['clear(B)', 'on(B,C)', 'on(C,A)', 'ontable(A)']
    
    planner = GraphPlan(initial_state, goal_state, actions)
    solution7 = planner.solve()

    # Summary
    print("\n=== SUMMARY ===")
    results = []
    results.append(("Test 1 (all on table)", "PASSED" if solution1 else "FAILED"))
    results.append(("Test 2 (A on B)", "PASSED" if solution2 else "FAILED"))
    results.append(("Test 3 (complex)", "PASSED" if solution3 else "FAILED"))
    results.append(("Test 4 (complex)", "PASSED" if solution4 else "FAILED"))
    results.append(("Test 5 (complex)", "PASSED" if solution5 else "FAILED"))
    results.append(("Test 6 (complex)", "PASSED" if solution6 else "FAILED"))
    results.append(("Test 7 (complex)", "PASSED" if solution7 else "FAILED"))
    
    for test_name, result in results:
        print(f"{test_name}: {result}")
    
    if all(result == "PASSED" for _, result in results):
        print("All tests passed")
    else:
        print("Some tests failed")

if __name__ == "__main__":
    test_blocks_world()