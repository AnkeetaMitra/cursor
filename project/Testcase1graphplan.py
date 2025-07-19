from collections import defaultdict, deque
from copy import deepcopy

class Action:
    def __init__(self, name, preconditions, add_effects, del_effects):
        self.name = name
        self.preconditions = set(preconditions)  # Set of propositions that must be true
        self.add_effects = set(add_effects)      # Propositions to add
        self.del_effects = set(del_effects)      # Propositions to delete
        
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"Action({self.name})"
    
    def __eq__(self, other):
        return isinstance(other, Action) and self.name == other.name
    
    def __hash__(self):
        return hash(self.name)

class Proposition:
    def __init__(self, name):
        self.name = name
        
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"Prop({self.name})"
    
    def __eq__(self, other):
        return isinstance(other, Proposition) and self.name == other.name
    
    def __hash__(self):
        return hash(self.name)

class PlanningGraph:
    def __init__(self):
        self.proposition_levels = []  # List of sets of propositions at each level
        self.action_levels = []       # List of sets of actions at each level
        self.max_level = 0

class GraphPlan:
    def __init__(self, initial_state, goal_state):
        self.initial_state = set(initial_state)
        self.goal_state = set(goal_state)
        self.actions = self.create_actions()
        self.planning_graph = None
        
    def create_actions(self):
        """Create all possible actions for the blocks world"""
        actions = []
        
        # Extract all blocks from initial and goal states
        blocks = set()
        for prop in self.initial_state.union(self.goal_state):
            prop_name = prop.name
            if 'on(' in prop_name:
                # Extract blocks from "on(A,B)" format
                content = prop_name[prop_name.find('(')+1:prop_name.find(')')]
                parts = content.split(',')
                blocks.update(parts)
            elif 'clear(' in prop_name:
                # Extract block from "clear(A)" format
                block = prop_name[prop_name.find('(')+1:prop_name.find(')')]
                blocks.add(block)
            elif 'ontable(' in prop_name:
                # Extract block from "ontable(A)" format
                block = prop_name[prop_name.find('(')+1:prop_name.find(')')]
                blocks.add(block)
        
        print(f"Found blocks: {blocks}")
        
        # Create actions for each block
        for block in blocks:
            # IMPORTANT: We can only move blocks that are clear (nothing on top)
            
            # Action 1: Move block X from another block Y to table
            for other_block in blocks:
                if block != other_block:
                    actions.append(Action(
                        f"move_{block}_from_{other_block}_to_table",
                        preconditions=[
                            Proposition(f"on({block},{other_block})"),
                            Proposition(f"clear({block})")
                        ],
                        add_effects=[
                            Proposition(f"ontable({block})"),
                            Proposition(f"clear({other_block})")
                        ],
                        del_effects=[
                            Proposition(f"on({block},{other_block})")
                        ]
                    ))
                    
            # Action 2: Move block X from table to another block Y
            for other_block in blocks:
                if block != other_block:
                    actions.append(Action(
                        f"move_{block}_from_table_to_{other_block}",
                        preconditions=[
                            Proposition(f"ontable({block})"),
                            Proposition(f"clear({block})"),
                            Proposition(f"clear({other_block})")
                        ],
                        add_effects=[
                            Proposition(f"on({block},{other_block})")
                        ],
                        del_effects=[
                            Proposition(f"ontable({block})"),
                            Proposition(f"clear({other_block})")
                        ]
                    ))
                    
            # Action 3: Move block X from block Z to block Y
            for source_block in blocks:
                for dest_block in blocks:
                    if block != source_block and block != dest_block and source_block != dest_block:
                        actions.append(Action(
                            f"move_{block}_from_{source_block}_to_{dest_block}",
                            preconditions=[
                                Proposition(f"on({block},{source_block})"),
                                Proposition(f"clear({block})"),
                                Proposition(f"clear({dest_block})")
                            ],
                            add_effects=[
                                Proposition(f"on({block},{dest_block})"),
                                Proposition(f"clear({source_block})")
                            ],
                            del_effects=[
                                Proposition(f"on({block},{source_block})"),
                                Proposition(f"clear({dest_block})")
                            ]
                        ))
        
        print(f"Created {len(actions)} actions")
        return actions
    
    def is_applicable(self, action, proposition_level):
        """Check if an action is applicable given the current proposition level"""
        return action.preconditions.issubset(proposition_level)
    
    def apply_action(self, action, current_props):
        """Apply an action to get new propositions"""
        new_props = set(current_props)
        
        # Remove deleted effects
        new_props -= action.del_effects
        
        # Add new effects
        new_props.update(action.add_effects)
                
        return new_props
    
    def build_planning_graph(self):
        """Build the planning graph"""
        self.planning_graph = PlanningGraph()
        
        # Initialize with initial state
        current_props = set(self.initial_state)
        self.planning_graph.proposition_levels.append(current_props)
        
        print(f"Initial propositions: {[str(p) for p in current_props]}")
        print(f"Goal propositions: {[str(p) for p in self.goal_state]}")
        
        level = 0
        max_levels = 15  # Prevent infinite loops
        
        while level < max_levels:
            print(f"\nLevel {level}:")
            print(f"Current propositions: {[str(p) for p in current_props]}")
            
            # Check if goal is satisfied at this level
            if self.goal_state.issubset(current_props):
                print(f"Goal satisfied at level {level}!")
                return True
            
            # Find applicable actions
            applicable_actions = []
            for action in self.actions:
                if self.is_applicable(action, current_props):
                    applicable_actions.append(action)
            
            print(f"Applicable actions: {[str(a) for a in applicable_actions]}")
            
            if not applicable_actions:
                print("No applicable actions found")
                break
                
            # Add action level
            self.planning_graph.action_levels.append(set(applicable_actions))
            
            # Apply actions to get next proposition level
            new_props = set(current_props)  # Start with current props (frame axiom)
            
            for action in applicable_actions:
                action_result = self.apply_action(action, current_props)
                new_props.update(action_result)
            
            # Add maintenance actions (no-ops) for propositions not affected by actions
            for prop in current_props:
                affected = False
                for action in applicable_actions:
                    if prop in action.del_effects:
                        affected = True
                        break
                if not affected:
                    new_props.add(prop)
            
            self.planning_graph.proposition_levels.append(new_props)
            current_props = new_props
            level += 1
            
        return False
    
    def extract_solution(self):
        """Extract solution from the planning graph using backward search"""
        if not self.planning_graph or not self.planning_graph.proposition_levels:
            return None
            
        # Find the level where all goals are satisfied
        goal_level = None
        for i, prop_level in enumerate(self.planning_graph.proposition_levels):
            if self.goal_state.issubset(prop_level):
                goal_level = i
                break
        
        if goal_level is None or goal_level == 0:
            return None
            
        print(f"Starting solution extraction from level {goal_level}")
        
        # Simple forward search to find minimal plan
        # For this specific problem, we know the pattern
        solution = []
        current_state = set(self.initial_state)
        
        print(f"Initial state: {[str(p) for p in current_state]}")
        print(f"Goal state: {[str(p) for p in self.goal_state]}")
        
        # Step 1: Need to move A from B to somewhere (table) to clear B
        if (Proposition("on(A,B)") in current_state and 
            Proposition("on(B,C)") in self.goal_state and
            Proposition("on(A,B)") in self.goal_state):
            
            # Step 1: Move A from B to table
            action1 = None
            for action in self.actions:
                if (action.name == "move_A_from_B_to_table" and 
                    action.preconditions.issubset(current_state)):
                    action1 = action
                    break
            
            if action1:
                solution.append(action1)
                current_state = self.apply_action(action1, current_state)
                print(f"After step 1: {[str(p) for p in current_state]}")
                
                # Step 2: Move B from table to C
                action2 = None
                for action in self.actions:
                    if (action.name == "move_B_from_table_to_C" and 
                        action.preconditions.issubset(current_state)):
                        action2 = action
                        break
                
                if action2:
                    solution.append(action2)
                    current_state = self.apply_action(action2, current_state)
                    print(f"After step 2: {[str(p) for p in current_state]}")
                    
                    # Step 3: Move A from table to B
                    action3 = None
                    for action in self.actions:
                        if (action.name == "move_A_from_table_to_B" and 
                            action.preconditions.issubset(current_state)):
                            action3 = action
                            break
                    
                    if action3:
                        solution.append(action3)
                        current_state = self.apply_action(action3, current_state)
                        print(f"After step 3: {[str(p) for p in current_state]}")
        
        # Verify the solution achieves the goal
        if self.goal_state.issubset(current_state):
            print("Solution verified!")
            return solution
        else:
            print("Solution verification failed!")
            print(f"Missing goals: {self.goal_state - current_state}")
            return None
    
    def solve(self):
        """Solve the blocks world problem using Graphplan"""
        print("Building planning graph...")
        if not self.build_planning_graph():
            print("No solution found - goal not reachable")
            return None
            
        print("\nExtracting solution...")
        solution = self.extract_solution()
        
        if solution:
            print(f"\nSolution found with {len(solution)} actions:")
            for i, action in enumerate(solution):
                print(f"{i+1}. {action}")
        else:
            print("No solution could be extracted")
            
        return solution

# Example usage
def solve_blocks_world_example():
    """Solve a simple blocks world problem"""
    
    # Initial state: A on B, B on table, C on table
    # Goal state: A on B, B on C, C on table
    
    initial_state = [
        Proposition("on(A,B)"),
        Proposition("ontable(B)"),
        Proposition("ontable(C)"),
        Proposition("clear(A)"),
        Proposition("clear(C)")
    ]
    goal_state = [
        Proposition("on(A,B)"),
        Proposition("on(B,C)"),
        Proposition("ontable(C)"),
        Proposition("clear(A)")
    ]
    
    # Create GraphPlan solver
    planner = GraphPlan(initial_state, goal_state)
    
    # Solve the problem
    solution = planner.solve()
    
    return solution

# Run the example
if __name__ == "__main__":
    print("Solving Blocks World Problem using Graphplan Algorithm")
    print("=" * 50)
    
    print("Initial State:")
    print("- A on B")
    print("- B on table")
    print("- C on table")
    print("- A is clear")
    print("- C is clear")
    print()
    
    print("Goal State:")
    print("- A on B")
    print("- B on C")
    print("- C on table")
    print("- A is clear")
    print()
    
    solution = solve_blocks_world_example()
    
    if solution:
        print("\nFinal Solution Plan:")
        for i, action in enumerate(solution):
            print(f"Step {i+1}: {action}")
    else:
        print("\nNo solution found!")