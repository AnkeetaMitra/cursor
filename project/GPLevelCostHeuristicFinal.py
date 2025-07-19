import time
from collections import deque

class Action:
    def __init__(self, name, preconds, add_effects, del_effects):
        self.name = name
        self.preconds = set(preconds)
        self.add_effects = set(add_effects)
        self.del_effects = set(del_effects)

    def is_applicable(self, state):
        return self.preconds.issubset(state)

    def apply(self, state):
        return (state - self.del_effects) | self.add_effects

    def __repr__(self):
        return self.name

class PlanningGraph:
    def __init__(self, actions, initial_state):
        self.actions = actions
        self.proposition_levels = [set(initial_state)]
        self.action_levels = []

    def extend(self):
        last_props = self.proposition_levels[-1]
        applicable_actions = [a for a in self.actions if a.preconds.issubset(last_props)]
        self.action_levels.append(applicable_actions)

        new_props = set(last_props)
        for a in applicable_actions:
            new_props.update(a.add_effects)
        self.proposition_levels.append(new_props)

    def expand_until_goals(self, goals, max_levels=100):
        while goals not in self.proposition_levels and len(self.proposition_levels) < max_levels:
            self.extend()
        return goals.issubset(self.proposition_levels[-1])

class GraphPlan:
    def __init__(self, actions):
        self.actions = actions

    def plan(self, initial_state, goal_state):
        pg = PlanningGraph(self.actions, initial_state)
        if not pg.expand_until_goals(goal_state):
            return None

        return self.backward_search(pg, initial_state, goal_state)

    def backward_search(self, pg, initial_state, goal_state):
        level = len(pg.proposition_levels) - 1
        goals = set(goal_state)
        plan = []

        while level > 0:
            actions_at_level = []
            new_goals = set()
            props = pg.proposition_levels[level - 1]
            acts = pg.action_levels[level - 1]

            needed_goals = set(goals)
            for a in acts:
                if not (a.add_effects & goals):
                    continue
                if not a.preconds.issubset(props):
                    continue
                actions_at_level.append(a)
                new_goals.update(a.preconds)
                needed_goals -= a.add_effects
                if not needed_goals:
                    break

            if needed_goals:
                return None  # Plan failure if some goals can't be satisfied

            plan.insert(0, actions_at_level)
            goals = new_goals
            level -= 1

        # Apply actions step by step
        flat_plan = []
        state = set(initial_state)
        for step in plan:
            for a in step:
                if a.is_applicable(state):
                    flat_plan.append(a)
                    state = a.apply(state)
        return flat_plan

def build_blocks_world(blocks):
    actions = []
    for x in blocks:
        actions.append(Action(
            f"PickUp({x})",
            [f"Clear({x})", f"OnTable({x})", "HandEmpty"],
            [f"Holding({x})"],
            [f"Clear({x})", f"OnTable({x})", "HandEmpty"]
        ))
        actions.append(Action(
            f"PutDown({x})",
            [f"Holding({x})"],
            [f"OnTable({x})", f"Clear({x})", "HandEmpty"],
            [f"Holding({x})"]
        ))

        for y in blocks:
            if x != y:
                actions.append(Action(
                    f"Stack({x},{y})",
                    [f"Holding({x})", f"Clear({y})"],
                    [f"On({x},{y})", f"Clear({x})", "HandEmpty"],
                    [f"Holding({x})", f"Clear({y})"]
                ))
                actions.append(Action(
                    f"Unstack({x},{y})",
                    [f"On({x},{y})", f"Clear({x})", "HandEmpty"],
                    [f"Holding({x})", f"Clear({y})"],
                    [f"On({x},{y})", f"Clear({x})", "HandEmpty"]
                ))
    return actions

def run_test():
    blocks = ['A', 'B', 'C']
    actions = build_blocks_world(blocks)
    initial_state = {"On(A,B)", "OnTable(B)", "OnTable(C)", "Clear(A)", "Clear(C)", "HandEmpty"}
    goal_state = {"On(B,C)", "On(A,B)"}

    print("Initial State:", sorted(initial_state))
    print("Goal State:", sorted(goal_state))

    planner = GraphPlan(actions)
    start = time.time()
    plan = planner.plan(initial_state, goal_state)
    end = time.time()

    if plan:
        print("\nPlan:")
        for step, action in enumerate(plan):
            print(f"  {step+1}. {action.name}")
        print(f"\nTotal steps: {len(plan)}, Time: {end - start:.5f}s")
    else:
        print("No plan found.")

if __name__ == '__main__':
    run_test()
