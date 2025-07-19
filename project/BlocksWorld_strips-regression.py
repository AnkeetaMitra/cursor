# Re-import everything after code reset to run the test suite

from typing import List, Set, Tuple, Optional
from collections import deque

class Action:
    def __init__(self, name: str, preconds: Set[str], add_effects: Set[str], del_effects: Set[str]):
        self.name = name
        self.preconds = preconds
        self.add_effects = add_effects
        self.del_effects = del_effects

    def __repr__(self):
        return self.name

def get_blocks_world_actions(blocks: List[str]) -> List[Action]:
    actions = []
    for x in blocks:
        actions.append(Action(
            name=f"PickUp({x})",
            preconds={f"clear({x})", f"ontable({x})", "handempty"},
            add_effects={f"holding({x})"},
            del_effects={f"ontable({x})", f"clear({x})", "handempty"}
        ))
        actions.append(Action(
            name=f"PutDown({x})",
            preconds={f"holding({x})"},
            add_effects={f"ontable({x})", f"clear({x})", "handempty"},
            del_effects={f"holding({x})"}
        ))
        for y in blocks:
            if x != y:
                actions.append(Action(
                    name=f"Stack({x},{y})",
                    preconds={f"holding({x})", f"clear({y})"},
                    add_effects={f"on({x},{y})", f"clear({x})", "handempty"},
                    del_effects={f"holding({x})", f"clear({y})"}
                ))
                actions.append(Action(
                    name=f"Unstack({x},{y})",
                    preconds={f"on({x},{y})", f"clear({x})", "handempty"},
                    add_effects={f"holding({x})", f"clear({y})"},
                    del_effects={f"on({x},{y})", f"clear({x})", "handempty"}
                ))
    return actions

def strict_is_consistent(goals: Set[str]) -> bool:
    holding_blocks = {g[8:-1] for g in goals if g.startswith("holding(")}
    on_relations = {(g[3:g.find(',')], g[g.find(',')+1:-1]) for g in goals if g.startswith("on(")}
    ontable_blocks = {g[8:-1] for g in goals if g.startswith("ontable(")}
    clear_blocks = {g[6:-1] for g in goals if g.startswith("clear(")}
    handempty = "handempty" in goals

    for b in holding_blocks:
        if any(rel[0] == b for rel in on_relations) or b in ontable_blocks:
            return False
    if holding_blocks and handempty:
        return False
    on_blocks = {x for x, y in on_relations}
    if on_blocks & ontable_blocks:
        return False
    supported_blocks = {y for x, y in on_relations}
    if clear_blocks & supported_blocks:
        return False

    return True

def find_strict_regression_plan_path(initial_state: Set[str], goal_state: Set[str], actions: List[Action]) -> List[Tuple[Set[str], Optional[Action]]]:
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

def print_regression_trace(path: List[Tuple[Set[str], Optional[Action]]]):
    print(" Consistent Regression Plan Trace:\n")
    for i, (state, action) in enumerate(path):
        print(f"Step {i}:")
        print(f"  Goal Set: {sorted(state)}")
        if action:
            print(f"Regress through: {action.name}")
        else:
            print(" Initial state reached")
        print()

# Define all test cases
test_cases = [
    {
        "name": "Test 1: Standard stack A on B on C",
        "initial": {'clear(A)', 'clear(B)', 'clear(C)', 'ontable(A)', 'ontable(B)', 'ontable(C)', 'handempty'},
        "goal": {'on(A,B)', 'on(B,C)', 'ontable(C)', 'clear(A)'}
    },
    {
        "name": "Test 2: C on B on A -> C on A on table",
        "initial": {'clear(C)', 'on(B,A)', 'on(C,B)', 'ontable(A)', 'handempty'},
        "goal": {'clear(B)', 'on(B,C)', 'on(C,A)', 'ontable(A)'}
    },
    {
        "name": "Test 3: C on B and A separate -> recreate C on B on A",
        "initial": {'clear(C)', 'clear(A)', 'on(C,B)', 'ontable(B)', 'ontable(A)', 'handempty'},
        "goal": {'clear(C)', 'on(B,A)', 'on(C,B)', 'ontable(A)'}
    },
    {
        "name": "Test 4: B on C and A on table -> recreate B on A on C",
        "initial": {'clear(B)', 'clear(A)', 'on(B,C)', 'ontable(C)', 'ontable(A)', 'handempty'},
        "goal": {'clear(B)', 'on(A,C)', 'on(B,A)', 'ontable(C)'}
    }
]

def run_all_tests():
    blocks = ['A', 'B', 'C']
    actions = get_blocks_world_actions(blocks)

    for test in test_cases:
        #print("=" * 60)
        print(f" {test['name']}")
        print("Initial State:", sorted(test["initial"]))
        print("Goal State   :", sorted(test["goal"]))
        print("-" * 60)
        path = find_strict_regression_plan_path(test["initial"], test["goal"], actions)
        if path:
            print_regression_trace(path)
        else:
            print("No valid regression plan found.")
        print("=" * 60 + "\n")

# Run all test cases
run_all_tests()
