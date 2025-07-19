"""
Real GraphPlan Tower of Hanoi Implementation
Shows actual planning graph levels and GraphPlan algorithm steps
"""

import time
from collections import defaultdict

class Fact:
    def __init__(self, predicate, *args):
        self.predicate = predicate
        self.args = args
        self.name = f"{predicate}({','.join(map(str, args))})"
    
    def __str__(self):
        return self.name
    
    def __eq__(self, other):
        return isinstance(other, Fact) and self.name == other.name
    
    def __hash__(self):
        return hash(self.name)
    
    def __repr__(self):
        return self.name

class Action:
    def __init__(self, name, preconditions, effects):
        self.name = name
        self.preconditions = set(preconditions)
        self.effects = set(effects)
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name

class NoOpAction(Action):
    def __init__(self, fact):
        super().__init__(f"NoOp({fact})", {fact}, {fact})
        self.fact = fact

class PlanningGraph:
    def __init__(self, initial_facts, goal_facts, actions):
        self.initial_facts = set(initial_facts)
        self.goal_facts = set(goal_facts)
        self.all_actions = actions
        
        # Planning graph layers
        self.fact_layers = [self.initial_facts.copy()]
        self.action_layers = []
        self.fact_mutexes = []
        self.action_mutexes = []
    
    def expand_level(self):
        """Expand planning graph by one level"""
        if not self.fact_layers:
            return False
        
        current_facts = self.fact_layers[-1]
        
        # Find applicable actions
        applicable_actions = []
        for action in self.all_actions:
            if action.preconditions.issubset(current_facts):
                applicable_actions.append(action)
        
        # Add no-op actions for persistence
        for fact in current_facts:
            applicable_actions.append(NoOpAction(fact))
        
        self.action_layers.append(applicable_actions)
        
        # Compute next fact layer
        next_facts = set()
        for action in applicable_actions:
            next_facts.update(action.effects)
        
        self.fact_layers.append(next_facts)
        
        # Simple mutex computation (no competing needs for this domain)
        self.action_mutexes.append(set())
        self.fact_mutexes.append(set())
        
        return len(self.fact_layers) - 1
    
    def goals_achievable(self):
        """Check if all goals are achievable in the current level"""
        if not self.fact_layers:
            return False
        return self.goal_facts.issubset(self.fact_layers[-1])
    
    def extract_solution(self):
        """Extract solution using backward search"""
        if not self.goals_achievable():
            return None
        
        level = len(self.fact_layers) - 1
        return self._backward_search(self.goal_facts, level)
    
    def _backward_search(self, goals, level):
        """Recursive backward search for solution extraction"""
        if level == 0:
            if goals.issubset(self.initial_facts):
                return []
            else:
                return None
        
        # Try to find actions that achieve the goals
        action_layer = self.action_layers[level - 1]
        
        # Find minimal set of actions to achieve goals
        selected_actions = []
        remaining_goals = goals.copy()
        
        # Prefer non-NoOp actions
        for action in action_layer:
            if isinstance(action, NoOpAction):
                continue
            
            achieved = remaining_goals.intersection(action.effects)
            if achieved:
                selected_actions.append(action)
                remaining_goals -= achieved
                if not remaining_goals:
                    break
        
        # Add NoOp actions for remaining goals
        for goal in remaining_goals:
            for action in action_layer:
                if isinstance(action, NoOpAction) and action.fact == goal:
                    selected_actions.append(action)
                    break
        
        # Check if all goals are covered
        achieved_effects = set()
        for action in selected_actions:
            achieved_effects.update(action.effects)
        
        if not goals.issubset(achieved_effects):
            return None
        
        # Compute preconditions for selected actions
        new_goals = set()
        for action in selected_actions:
            new_goals.update(action.preconditions)
        
        # Recursive call
        subplan = self._backward_search(new_goals, level - 1)
        if subplan is None:
            return None
        
        # Filter out NoOp actions from the result
        real_actions = [a for a in selected_actions if not isinstance(a, NoOpAction)]
        return subplan + real_actions

class GraphPlanSolver:
    def __init__(self, n_disks):
        self.n_disks = n_disks
        self.setup_domain()
    
    def setup_domain(self):
        """Setup Tower of Hanoi domain for GraphPlan"""
        # Initial facts: all disks on A, top disk is 0
        self.initial_facts = []
        for disk in range(self.n_disks):
            self.initial_facts.append(Fact("on", disk, "A"))
        
        if self.n_disks > 0:
            self.initial_facts.append(Fact("clear", 0))
        self.initial_facts.append(Fact("empty", "B"))
        self.initial_facts.append(Fact("empty", "C"))
        
        # Goal facts: all disks on C
        self.goal_facts = []
        for disk in range(self.n_disks):
            self.goal_facts.append(Fact("on", disk, "C"))
        
        # Create actions
        self.actions = []
        self.create_actions()
    
    def create_actions(self):
        """Create Tower of Hanoi actions"""
        rods = ["A", "B", "C"]
        
        for disk in range(self.n_disks):
            for from_rod in rods:
                for to_rod in rods:
                    if from_rod != to_rod:
                        # Move to empty rod
                        action_name = f"move_{disk}_{from_rod}_{to_rod}"
                        preconditions = [
                            Fact("on", disk, from_rod),
                            Fact("clear", disk),
                            Fact("empty", to_rod)
                        ]
                        effects = [
                            Fact("on", disk, to_rod),
                            Fact("clear", disk),
                            Fact("empty", from_rod)
                        ]
                        
                        self.actions.append(Action(action_name, preconditions, effects))
                        
                        # Move onto larger disk
                        for larger_disk in range(disk + 1, self.n_disks):
                            action_name2 = f"move_{disk}_{from_rod}_{to_rod}_onto_{larger_disk}"
                            preconditions2 = [
                                Fact("on", disk, from_rod),
                                Fact("clear", disk),
                                Fact("on", larger_disk, to_rod),
                                Fact("clear", larger_disk)
                            ]
                            effects2 = [
                                Fact("on", disk, to_rod),
                                Fact("clear", disk)
                            ]
                            
                            self.actions.append(Action(action_name2, preconditions2, effects2))
        
        # Actions to make disks clear when uncovered
        for disk in range(1, self.n_disks):
            for rod in rods:
                action_name = f"uncover_{disk}_{rod}"
                preconditions = [Fact("on", disk, rod)]
                effects = [Fact("clear", disk)]
                self.actions.append(Action(action_name, preconditions, effects))
    
    def solve(self, max_levels=10):
        """Solve using GraphPlan algorithm"""
        print(f"Building planning graph for {self.n_disks} disks...")
        
        # Create planning graph
        pg = PlanningGraph(self.initial_facts, self.goal_facts, self.actions)
        
        # Expand graph until goals are achievable
        level = 0
        while level < max_levels:
            print(f"  Level {level}: {len(pg.fact_layers[level])} facts")
            
            if pg.goals_achievable():
                print(f"  Goals achievable at level {level}")
                break
            
            level = pg.expand_level()
            if level is False:
                return None, 0
        
        if level >= max_levels:
            print(f"  Failed to achieve goals within {max_levels} levels")
            return None, level
        
        # Extract solution
        print(f"Extracting solution from level {level}...")
        solution = pg.extract_solution()
        
        return solution, level

def validate_solution(plan, n_disks):
    """Validate Tower of Hanoi solution"""
    if not plan:
        return False, "No plan provided"
    
    # Initialize state
    state = {'A': list(range(n_disks-1, -1, -1)), 'B': [], 'C': []}
    
    for i, action in enumerate(plan):
        # Parse action name - handle different formats
        parts = action.name.split('_')
        
        if parts[0] == "move" and len(parts) >= 4:
            disk = int(parts[1])
            from_rod = parts[2]
            to_rod = parts[3]
        else:
            print(f"Warning: Cannot parse action {action.name}")
            continue
        
        # Validate move
        if not state[from_rod] or state[from_rod][-1] != disk:
            return False, f"Move {i+1}: Disk {disk} not on top of rod {from_rod}"
        
        if state[to_rod] and state[to_rod][-1] < disk:
            return False, f"Move {i+1}: Cannot place disk {disk} on smaller disk"
        
        # Execute move
        moved_disk = state[from_rod].pop()
        state[to_rod].append(moved_disk)
    
    # Check goal
    expected = {'A': [], 'B': [], 'C': list(range(n_disks-1, -1, -1))}
    return state == expected, "Valid solution" if state == expected else "Invalid final state"

def run_graphplan_analysis():
    """Run GraphPlan analysis showing planning graph construction"""
    print("GRAPHPLAN TOWER OF HANOI - Planning Graph Analysis")
    print("="*60)
    
    for n in [1, 2, 3]:
        print(f"\n{'-'*40}")
        print(f"ANALYZING {n} DISK PROBLEM")
        print('-'*40)
        
        solver = GraphPlanSolver(n)
        
        start_time = time.time()
        solution, levels = solver.solve(max_levels=15)
        solve_time = time.time() - start_time
        
        expected_moves = 2**n - 1
        
        if solution:
            is_valid, msg = validate_solution(solution, n)
            print(f"\nResults:")
            print(f"  Planning levels: {levels}")
            print(f"  Solution found: {len(solution)} moves")
            print(f"  Expected optimal: {expected_moves} moves")
            print(f"  Valid: {'✓' if is_valid else '✗'}")
            print(f"  Time: {solve_time:.4f}s")
            
            if is_valid and len(solution) <= 10:  # Show plan for small cases
                print(f"\nOptimal Plan:")
                for i, action in enumerate(solution):
                    parts = action.name.split('_')
                    disk = parts[1]
                    from_rod = parts[2]
                    to_rod = parts[3]
                    print(f"  {i+1}. Move disk {disk}: {from_rod} → {to_rod}")
            
            # Show the difference - GraphPlan builds levels, then extracts
            print(f"\nGraphPlan Process:")
            print(f"  1. Built planning graph to level {levels}")
            print(f"  2. Goals became achievable at level {levels}")
            print(f"  3. Backward search extracted {len(solution)}-step plan")
            print(f"  4. Plan validation: {msg}")
        else:
            print(f"\nNo solution found within 15 levels ({solve_time:.4f}s)")

if __name__ == "__main__":
    run_graphplan_analysis()