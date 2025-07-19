from typing import Set, List, Tuple, Optional
import heapq

# Type aliases
State = Set[str]
Action = Tuple[str, str, str]  # e.g., ("Stack", "C", "B")

# Action functions
def pickup(block: str, state: State) -> Optional[State]:
    if {f"On({block}, Table)", f"Clear({block})", "HandEmpty"}.issubset(state):
        new = set(state)
        new.remove(f"On({block}, Table)")
        new.remove(f"Clear({block})")
        new.remove("HandEmpty")
        new.add(f"Holding({block})")
        return new
    return None

def putdown(block: str, state: State) -> Optional[State]:
    if {f"Holding({block})"}.issubset(state):
        new = set(state)
        new.remove(f"Holding({block})")
        new.add(f"On({block}, Table)")
        new.add(f"Clear({block})")
        new.add("HandEmpty")
        return new
    return None

def stack(block: str, target: str, state: State) -> Optional[State]:
    if block == target: return None
    if {f"Holding({block})", f"Clear({target})"}.issubset(state):
        new = set(state)
        new.remove(f"Holding({block})")
        new.remove(f"Clear({target})")
        new.add(f"On({block}, {target})")
        new.add(f"Clear({block})")
        new.add("HandEmpty")
        return new
    return None

def unstack(block: str, target: str, state: State) -> Optional[State]:
    if block == target: return None
    if {f"On({block}, {target})", f"Clear({block})", "HandEmpty"}.issubset(state):
        new = set(state)
        new.remove(f"On({block}, {target})")
        new.remove(f"Clear({block})")
        new.remove("HandEmpty")
        new.add(f"Holding({block})")
        new.add(f"Clear({target})")
        return new
    return None

# Generate all applicable actions
def get_possible_actions(state: State, blocks: List[str]) -> List[Tuple[str, str, str, State]]:
    actions = []
    for b in blocks:
        if (a := pickup(b, state)): actions.append(("PickUp", b, "", a))
        if (a := putdown(b, state)): actions.append(("PutDown", b, "", a))
        for t in blocks:
            if b != t:
                if (a := stack(b, t, state)): actions.append(("Stack", b, t, a))
                if (a := unstack(b, t, state)): actions.append(("UnStack", b, t, a))
    return actions

# Heuristic: number of goal facts not yet achieved
def heuristic(state: State, goal: State) -> int:
    return len([fact for fact in goal if fact not in state])

# A* Search
def astar(initial: State, goal: State, blocks: List[str]) -> Tuple[Optional[List[Action]], Optional[State]]:
    open_set = []
    heapq.heappush(open_set, (heuristic(initial, goal), 0, initial, []))
    visited = {}

    while open_set:
        f_cost, g_cost, state, path = heapq.heappop(open_set)
        fs = frozenset(state)

        if fs in visited and visited[fs] <= g_cost:
            continue
        visited[fs] = g_cost

        if goal.issubset(state):
            return path, state

        for name, b1, b2, new_state in get_possible_actions(state, blocks):
            new_path = path + [(name, b1, b2)]
            g = g_cost + 1
            h = heuristic(new_state, goal)
            heapq.heappush(open_set, (g + h, g, new_state, new_path))

    return None, None

# Visualizer
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

# Main Execution
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

    plan, final_state = astar(initial_state, goal_state, blocks)

    if plan:
        print("\nFinal State Reached:")
        for fact in sorted(final_state): print(" ", fact)
        visualize_state(final_state)

        print("\n=== A* Plan Output ===")
        for i, (name, b1, b2) in enumerate(plan, 1):
            print(f"{i}. {name}({b1}, {b2})" if b2 else f"{i}. {name}({b1})")
    else:
        print("\nNo plan found.")
