from typing import Set, List, Tuple

# Type Aliases
State = Set[str]
Action = Tuple[str, str, str]  # e.g., ("Stack", "C", "B")

# ----- Action Functions -----
def pickup(block: str, state: State) -> State | None:
    pre = {f"On({block}, Table)", f"Clear({block})", "HandEmpty"}
    if pre.issubset(state):
        new = set(state)
        new.remove(f"On({block}, Table)")
        new.remove(f"Clear({block})")
        new.remove("HandEmpty")
        new.add(f"Holding({block})")
        return new
    return None

def putdown(block: str, state: State) -> State | None:
    pre = {f"Holding({block})"}
    if pre.issubset(state):
        new = set(state)
        new.remove(f"Holding({block})")
        new.add(f"On({block}, Table)")
        new.add(f"Clear({block})")
        new.add("HandEmpty")
        return new
    return None

def stack(block: str, target: str, state: State) -> State | None:
    pre = {f"Holding({block})", f"Clear({target})"}
    if block == target or not pre.issubset(state):
        return None
    new = set(state)
    new.remove(f"Holding({block})")
    new.remove(f"Clear({target})")
    new.add(f"On({block}, {target})")
    new.add(f"Clear({block})")
    new.add("HandEmpty")
    return new

def unstack(block: str, target: str, state: State) -> State | None:
    pre = {f"On({block}, {target})", f"Clear({block})", "HandEmpty"}
    if block == target or not pre.issubset(state):
        return None
    new = set(state)
    new.remove(f"On({block}, {target})")
    new.remove(f"Clear({block})")
    new.remove("HandEmpty")
    new.add(f"Holding({block})")
    new.add(f"Clear({target})")
    return new

# ----- Generate all applicable actions -----
def get_possible_actions(state: State, blocks: List[str]) -> List[Tuple[str, str, str, State]]:
    actions = []
    for b in blocks:
        a = pickup(b, state)
        if a: actions.append(("PickUp", b, "", a))
        a = putdown(b, state)
        if a: actions.append(("PutDown", b, "", a))
        for t in blocks:
            if b != t:
                a = stack(b, t, state)
                if a: actions.append(("Stack", b, t, a))
                a = unstack(b, t, state)
                if a: actions.append(("UnStack", b, t, a))
    return actions

# ----- DFS Search -----
def dfs(initial: State, goal: State, blocks: List[str], depth_limit=100) -> Tuple[List[Action], State] | Tuple[None, None]:
    stack = [(initial, [])]
    visited = set()

    while stack:
        state, path = stack.pop()
        fs = frozenset(state)
        if fs in visited:
            continue
        visited.add(fs)

        if goal.issubset(state):
            return path, state

        if len(path) >= depth_limit:
            continue

        for act in get_possible_actions(state, blocks):
            name, b1, b2, new_state = act
            stack.append((new_state, path + [(name, b1, b2)]))

    return None, None

# ----- Visualizer -----
def visualize_state(state: State):
    stacks = {}
    table_blocks = []

    for fact in state:
        if fact.startswith("On("):
            x, y = fact[3:-1].split(", ")
            if y == "Table":
                table_blocks.append(x)
            else:
                stacks[x] = y

    def build_stack(bottom):
        result = [bottom]
        while result[-1] in stacks:
            result.append(stacks[result[-1]])
        return result

    output = []
    for b in sorted(table_blocks):
        output.append(" > ".join(build_stack(b)))

    print("Stack View:")
    for line in output:
        print(" ", line)

# ----- Main Execution -----
if __name__ == "__main__":
    blocks = ["A", "B", "C"]

    initial_state = {
        "On(A, B)",
        "On(B, Table)",
        "On(C, Table)",
        "Clear(A)",
        "Clear(C)",
        "HandEmpty"
    }

    goal_state = {
        "On(A, B)",
        "On(B, C)",
        "On(C, Table)"
    }

    print("Initial State:")
    for fact in sorted(initial_state): print(" ", fact)
    visualize_state(initial_state)

    print("\nGoal State:")
    for fact in sorted(goal_state): print(" ", fact)

    plan, final_state = dfs(initial_state, goal_state, blocks)

    if plan:
        print("\nFinal State Reached:")
        for fact in sorted(final_state): print(" ", fact)
        visualize_state(final_state)

        print("\n=== Plan Output ===")
        for i, act in enumerate(plan, 1):
            name, b1, b2 = act
            print(f"{i}. {name}({b1}, {b2})" if b2 else f"{i}. {name}({b1})")
    else:
        print("\nNo plan found.")
