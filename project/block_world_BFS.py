from collections import deque
from dataclasses import dataclass
from typing import List, Set


@dataclass
class Action:
    name: str
    preconditions: Set[str]
    add: Set[str]
    delete: Set[str]


def create_strips_actions() -> List[Action]:
    blocks = ['A', 'B', 'C']
    actions = []

    for x in blocks:
        # PickUp(x)
        actions.append(Action(
            name=f"PickUp({x})",
            preconditions={f"Clear({x})", f"On({x}, Table)", "HandEmpty"},
            add={f"Holding({x})"},
            delete={f"Clear({x})", f"On({x}, Table)", "HandEmpty"}
        ))

        # PutDown(x)
        actions.append(Action(
            name=f"PutDown({x})",
            preconditions={f"Holding({x})"},
            add={f"On({x}, Table)", f"Clear({x})", "HandEmpty"},
            delete={f"Holding({x})"}
        ))

        for y in blocks:
            if x == y:
                continue

            # UnStack(x, y)
            actions.append(Action(
                name=f"UnStack({x}, {y})",
                preconditions={f"On({x}, {y})", f"Clear({x})", "HandEmpty"},
                add={f"Holding({x})", f"Clear({y})"},
                delete={f"On({x}, {y})", f"Clear({x})", "HandEmpty"}
            ))

            # Stack(x, y)
            actions.append(Action(
                name=f"Stack({x}, {y})",
                preconditions={f"Holding({x})", f"Clear({y})"},
                add={f"On({x}, {y})", f"Clear({x})", "HandEmpty"},
                delete={f"Holding({x})", f"Clear({y})"}
            ))
    return actions


def apply_action(state: Set[str], action: Action) -> Set[str] | None:
    if not action.preconditions.issubset(state):
        return None
    new_state = set(state)
    new_state.difference_update(action.delete)
    new_state.update(action.add)
    return new_state


def is_goal(state: Set[str], goal: Set[str]) -> bool:
    return goal.issubset(state)


def plan(initial: Set[str], goal: Set[str], actions: List[Action]) -> List[str] | None:
    visited = set()
    queue = deque()
    queue.append((initial, []))

    while queue:
        state, path = queue.popleft()
        frozen = frozenset(state)
        if frozen in visited:
            continue
        visited.add(frozen)

        if is_goal(state, goal):
            print("\nFinal State Reached:")
            print(state)
            return path

        for action in actions:
            new_state = apply_action(state, action)
            if new_state and frozenset(new_state) not in visited:
                queue.append((new_state, path + [action.name]))

    return None


def test_block_world():
    actions = create_strips_actions()

    initial_state = {
        "On(A, B)",
        "On(C, Table)",
        "On(B, Table)",
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
    for f in initial_state:
        print(" ", f)
    print("Goal State:")
    for f in goal_state:
        print(" ", f)

    result = plan(initial_state, goal_state, actions)

    print("\n=== Plan Output ===")
    if result:
        for i, step in enumerate(result, 1):
            print(f"{i}. {step}")
    else:
        print("No valid plan found.")


if __name__ == "__main__":
    test_block_world()
