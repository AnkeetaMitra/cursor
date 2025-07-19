from typing import List, Set, Tuple, Optional
from collections import deque

# Action Classes 

class GP_Action:
    def __init__(self, name, preconditions, effects):
        self.name = name
        self.preconditions = set(preconditions)
        self.effects = set(effects)
        self.add_effects = set(e for e in effects if not e.startswith('not_'))
        self.del_effects = set(e[4:] for e in effects if e.startswith('not_'))
    def __str__(self): return self.name
    def __repr__(self): return self.name
    def is_applicable(self, state): return self.preconditions.issubset(state)

class Reg_Action:
    def __init__(self, name: str, preconds: Set[str], add_effects: Set[str], del_effects: Set[str]):
        self.name = name
        self.preconds = preconds
        self.add_effects = add_effects
        self.del_effects = del_effects
    def __repr__(self): return self.name

#  GraphPlan Implementation 

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
        self.proposition_levels.append(self.initial_state.copy())
        self.proposition_mutexes.append(set())
        level = 0
        while level < self.max_levels:
            if self.goal_state.issubset(self.proposition_levels[level]) and not self.goals_are_mutex(level):
                return level
            applicable_actions = [a for a in self.actions if a.is_applicable(self.proposition_levels[level])]
            for prop in self.proposition_levels[level]:
                applicable_actions.append(GP_Action(f"noop_{prop}", [prop], [prop]))
            self.action_levels.append(applicable_actions)
            action_mutex_pairs = set()
            for i, a1 in enumerate(applicable_actions):
                for j, a2 in enumerate(applicable_actions):
                    if i < j and self.actions_are_mutex(a1, a2, level):
                        action_mutex_pairs.add((a1, a2))
            self.action_mutexes.append(action_mutex_pairs)
            next_props = set()
            for action in applicable_actions:
                next_props.update(action.add_effects)
            self.proposition_levels.append(next_props)
            prop_mutex_pairs = set()
            for p1 in next_props:
                for p2 in next_props:
                    if p1 != p2 and self.propositions_are_mutex(p1, p2, level + 1):
                        prop_mutex_pairs.add((p1, p2))
            self.proposition_mutexes.append(prop_mutex_pairs)
            level += 1
            if level > 1 and self.proposition_levels[level] == self.proposition_levels[level-1] and self.proposition_mutexes[level] == self.proposition_mutexes[level-1]:
                break
        return -1

    def actions_are_mutex(self, a1, a2, level):
        if a1.add_effects & a2.del_effects or a2.add_effects & a1.del_effects:
            return True
        if a1.del_effects & a2.add_effects or a2.del_effects & a1.add_effects:
            return True
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
        if level < len(self.action_mutexes):
            return (item1, item2) in self.action_mutexes[level] or (item2, item1) in self.action_mutexes[level]
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
            for action_combo in self.get_action_combinations(goals, level):
                if self.actions_are_mutex_free(action_combo, level - 1):
                    new_goals = set()
                    for action in action_combo:
                        new_goals.update(action.preconditions)
                    sub_solution = search_solution(new_goals, level - 1)
                    if sub_solution is not None:
                        non_noop = [a for a in action_combo if not a.name.startswith('noop_')]
                        return sub_solution + [non_noop] if non_noop else sub_solution
            return None
        return search_solution(self.goal_state, goal_level)

    def get_action_combinations(self, goals, level):
        if level == 0: return []
        goal_actions = {g: [a for a in self.action_levels[level - 1] if g in a.add_effects] for g in goals}
        def generate_combinations(remaining, current_combo, achieved):
            if not remaining:
                yield current_combo
                return
            goal = remaining.pop()
            for action in goal_actions[goal]:
                new_achieved = achieved | action.add_effects
                if goal in new_achieved:
                    yield from generate_combinations(remaining - new_achieved, current_combo + [action], new_achieved)
            remaining.add(goal)
        return generate_combinations(goals.copy(), [], set())

    def actions_are_mutex_free(self, actions, level):
        for i, a1 in enumerate(actions):
            for j, a2 in enumerate(actions):
                if i < j and self.are_mutex_at_level(a1, a2, level):
                    return False
        return True

    def solve(self):
        goal_level = self.build_graph()
        if goal_level == -1:
            return None
        return self.extract_solution(goal_level)

#  Regression Planner 

def strict_is_consistent(goals: Set[str]) -> bool:
    holding = {g[8:-1] for g in goals if g.startswith("holding(")}
    on = {(g[3:g.find(',')], g[g.find(',')+1:-1]) for g in goals if g.startswith("on(")}
    ontable = {g[8:-1] for g in goals if g.startswith("ontable(")}
    clear = {g[6:-1] for g in goals if g.startswith("clear(")}
    handempty = "handempty" in goals
    if any(b in ontable for b in holding) or any(x == b for x, y in on for b in holding): return False
    if holding and handempty: return False
    if {x for x, y in on} & ontable: return False
    if clear & {y for x, y in on}: return False
    return True

def find_strict_regression_plan_path(initial_state: Set[str], goal_state: Set[str], actions: List[Reg_Action]) -> List[Tuple[Set[str], Optional[Reg_Action]]]:
    queue = deque()
    visited = set()
    queue.append((goal_state, []))
    parent_map = {}
    action_map = {}
    while queue:
        current_goals, plan = queue.popleft()
        frozen_goals = frozenset(current_goals)
        if frozen_goals in visited:
            continue
        visited.add(frozen_goals)
        if current_goals.issubset(initial_state):
            path = []
            while frozen_goals in parent_map:
                prev_state = parent_map[frozen_goals]
                action = action_map[frozen_goals]
                path.append((set(frozen_goals), action))
                frozen_goals = prev_state
            path.append((set(frozen_goals), None))
            return list(reversed(path))
        for action in actions:
            if not action.add_effects.isdisjoint(current_goals):
                new_goals = (current_goals - action.add_effects) | action.preconds
                if not strict_is_consistent(new_goals):
                    continue
                frozen_new = frozenset(new_goals)
                if frozen_new not in visited:
                    parent_map[frozen_new] = frozen_goals
                    action_map[frozen_new] = action
                    queue.append((new_goals, plan + [action]))
    return []

#  Action Definitions 

def create_blocks_world_actions_for_graphplan():
    blocks = ['A', 'B', 'C']
    actions = []
    for x in blocks:
        for y in blocks:
            if x != y:
                pre = [f'clear({x})', f'clear({y})']
                eff = [f'on({x},{y})', f'not_clear({y})', f'not_ontable({x})']
                actions.append(GP_Action(f'stack({x},{y})', pre, eff))
    for x in blocks:
        for y in blocks:
            if x != y:
                pre = [f'clear({x})', f'on({x},{y})']
                eff = [f'not_on({x},{y})', f'clear({y})', f'ontable({x})']
                actions.append(GP_Action(f'unstack({x},{y})', pre, eff))
    return actions

def get_blocks_world_actions_for_regression():
    blocks = ['A', 'B', 'C']
    actions = []
    for x in blocks:
        actions.append(Reg_Action(f"PickUp({x})", {f"clear({x})", f"ontable({x})", "handempty"}, {f"holding({x})"}, {f"ontable({x})", f"clear({x})", "handempty"}))
        actions.append(Reg_Action(f"PutDown({x})", {f"holding({x})"}, {f"ontable({x})", f"clear({x})", "handempty"}, {f"holding({x})"}))
        for y in blocks:
            if x != y:
                actions.append(Reg_Action(f"Stack({x},{y})", {f"holding({x})", f"clear({y})"}, {f"on({x},{y})", f"clear({x})", "handempty"}, {f"holding({x})", f"clear({y})"}))
                actions.append(Reg_Action(f"Unstack({x},{y})", {f"on({x},{y})", f"clear({x})", "handempty"}, {f"holding({x})", f"clear({y})"}, {f"on({x},{y})", f"clear({x})", "handempty"}))
    return actions

#  Comparison Function 

def compare_graphplan_vs_regression():
    test_cases = [
        {
            "name": "Stack A on B on C",
            "initial": ['clear(A)', 'clear(B)', 'clear(C)', 'ontable(A)', 'ontable(B)', 'ontable(C)', 'handempty'],
            "goal": ['on(A,B)', 'on(B,C)', 'ontable(C)', 'clear(A)']
        },
        {
            "name": "CBA to CAB",
            "initial": ['clear(C)', 'on(B,A)', 'on(C,B)', 'ontable(A)', 'handempty'],
            "goal": ['clear(B)', 'on(B,C)', 'on(C,A)', 'ontable(A)']
        },
        {
            "name": "C on B and A separate -> recreate C on B on A",
            "initial": ['clear(C)', 'clear(A)', 'on(C,B)', 'ontable(B)', 'ontable(A)', 'handempty'],
            "goal": ['clear(C)', 'on(B,A)', 'on(C,B)', 'ontable(A)']
        },
        {
            "name": "B on C and A on table -> recreate B on A on C",
            "initial": ['clear(B)', 'clear(A)', 'on(B,C)', 'ontable(C)', 'ontable(A)', 'handempty'],
            "goal": ['clear(B)', 'on(A,C)', 'on(B,A)', 'ontable(C)']
        }
    ]

    graphplan_actions = create_blocks_world_actions_for_graphplan()
    regression_actions = get_blocks_world_actions_for_regression()

    print("\n=== COMPARISON: GraphPlan vs Regression ===")
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        print("Initial:", test["initial"])
        print("Goal   :", test["goal"])

        # GraphPlan
        gp = GraphPlan(test["initial"], test["goal"], graphplan_actions)
        gp_plan = gp.solve()
        if gp_plan:
            print(f"GraphPlan: Steps: {sum(len(level) for level in gp_plan)}")
            for i, level in enumerate(gp_plan):
                actions = [str(a) for a in level]
                print(f"  Level {i + 1}: {actions}")
        else:
            print("GraphPlan: FAIL")

        # Regression
        reg_plan = find_strict_regression_plan_path(set(test["initial"]), set(test["goal"]), regression_actions)
        if reg_plan:
            print(f"Regression: Steps: {len(reg_plan) - 1}")
            for i, (goal_set, action) in enumerate(reg_plan[1:], start=1):
                print(f"  Step {i}:")
                print(f"    Action: {action.name}")
                print(f"    Goal Set: {sorted(goal_set)}")
            # Final step
            print(f"  Step 0:")
            print(f"    Reached Initial State")
            print(f"    Goal Set: {sorted(reg_plan[0][0])}")

        else:
            print("Regression: FAIL")



if __name__ == "__main__":
    compare_graphplan_vs_regression()
