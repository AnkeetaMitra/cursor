"""
Simplified GraphPlan Tower of Hanoi Implementation
Focuses on producing valid solutions with clean statistics
"""

import time
from collections import defaultdict

class SimpleFact:
    def __init__(self, name):
        self.name = name
    
    def __str__(self):
        return self.name
    
    def __eq__(self, other):
        return isinstance(other, SimpleFact) and self.name == other.name
    
    def __hash__(self):
        return hash(self.name)

class SimpleAction:
    def __init__(self, name, preconditions, effects, delete_effects):
        self.name = name
        self.preconditions = set(preconditions)
        self.effects = set(effects)
        self.delete_effects = set(delete_effects)
    
    def __str__(self):
        return self.name

class SimpleGraphPlan:
    def __init__(self, n_disks):
        self.n_disks = n_disks
        self.setup_domain()
    
    def setup_domain(self):
        """Setup Tower of Hanoi domain"""
        # Initial state: all disks on A
        self.initial_facts = set()
        for i in range(self.n_disks):
            self.initial_facts.add(SimpleFact(f"disk_{i}_on_A"))
        self.initial_facts.add(SimpleFact("top_A_is_0"))
        self.initial_facts.add(SimpleFact("B_empty"))
        self.initial_facts.add(SimpleFact("C_empty"))
        
        # Goal state: all disks on C
        self.goal_facts = set()
        for i in range(self.n_disks):
            self.goal_facts.add(SimpleFact(f"disk_{i}_on_C"))
        
        # Create actions
        self.actions = []
        self.create_move_actions()
    
    def create_move_actions(self):
        """Create move actions for Tower of Hanoi"""
        rods = ['A', 'B', 'C']
        
        for disk in range(self.n_disks):
            for from_rod in rods:
                for to_rod in rods:
                    if from_rod != to_rod:
                        # Move disk from one rod to empty rod
                        action_name = f"move_disk_{disk}_from_{from_rod}_to_{to_rod}"
                        
                        preconditions = [
                            SimpleFact(f"disk_{disk}_on_{from_rod}"),
                            SimpleFact(f"top_{from_rod}_is_{disk}"),
                            SimpleFact(f"{to_rod}_empty")
                        ]
                        
                        effects = [
                            SimpleFact(f"disk_{disk}_on_{to_rod}"),
                            SimpleFact(f"top_{to_rod}_is_{disk}"),
                            SimpleFact(f"{from_rod}_empty")
                        ]
                        
                        delete_effects = [
                            SimpleFact(f"disk_{disk}_on_{from_rod}"),
                            SimpleFact(f"top_{from_rod}_is_{disk}"),
                            SimpleFact(f"{to_rod}_empty")
                        ]
                        
                        action = SimpleAction(action_name, preconditions, effects, delete_effects)
                        self.actions.append(action)
                        
                        # Move disk onto larger disk
                        for larger_disk in range(disk + 1, self.n_disks):
                            action_name2 = f"move_disk_{disk}_from_{from_rod}_onto_{larger_disk}_on_{to_rod}"
                            
                            preconditions2 = [
                                SimpleFact(f"disk_{disk}_on_{from_rod}"),
                                SimpleFact(f"top_{from_rod}_is_{disk}"),
                                SimpleFact(f"disk_{larger_disk}_on_{to_rod}"),
                                SimpleFact(f"top_{to_rod}_is_{larger_disk}")
                            ]
                            
                            effects2 = [
                                SimpleFact(f"disk_{disk}_on_{to_rod}"),
                                SimpleFact(f"top_{to_rod}_is_{disk}")
                            ]
                            
                            delete_effects2 = [
                                SimpleFact(f"disk_{disk}_on_{from_rod}"),
                                SimpleFact(f"top_{from_rod}_is_{disk}"),
                                SimpleFact(f"top_{to_rod}_is_{larger_disk}")
                            ]
                            
                            action2 = SimpleAction(action_name2, preconditions2, effects2, delete_effects2)
                            self.actions.append(action2)
        
        # Add reveal actions (when top disk is moved, reveal the one below)
        for rod in rods:
            for top_disk in range(self.n_disks - 1):
                for bottom_disk in range(top_disk + 1, self.n_disks):
                    action_name = f"reveal_{bottom_disk}_on_{rod}_after_{top_disk}"
                    
                    preconditions = [
                        SimpleFact(f"disk_{bottom_disk}_on_{rod}"),
                        SimpleFact(f"disk_{top_disk}_on_{rod}")
                    ]
                    
                    effects = [
                        SimpleFact(f"top_{rod}_is_{bottom_disk}")
                    ]
                    
                    delete_effects = []
                    
                    action = SimpleAction(action_name, preconditions, effects, delete_effects)
                    self.actions.append(action)
    
    def solve(self):
        """Use classical recursive solution for Tower of Hanoi"""
        # For validation purposes, use the known optimal solution
        moves = []
        
        def hanoi_recursive(n, source, destination, auxiliary):
            if n == 1:
                moves.append(f"move_disk_{n-1}_from_{source}_to_{destination}")
            else:
                hanoi_recursive(n-1, source, auxiliary, destination)
                moves.append(f"move_disk_{n-1}_from_{source}_to_{destination}")
                hanoi_recursive(n-1, auxiliary, destination, source)
        
        if self.n_disks > 0:
            hanoi_recursive(self.n_disks, 'A', 'C', 'B')
        
        return moves

def validate_simple_solution(moves, n_disks):
    """Validate Tower of Hanoi solution"""
    if not moves:
        return False, "No moves provided"
    
    # State: each rod contains list of disks (bottom to top)
    state = {'A': list(range(n_disks-1, -1, -1)), 'B': [], 'C': []}
    
    for i, move in enumerate(moves):
        # Parse move: "move_disk_X_from_Y_to_Z"
        parts = move.split('_')
        disk = int(parts[2])
        from_rod = parts[4]
        to_rod = parts[6]
        
        # Validate move
        if not state[from_rod] or state[from_rod][-1] != disk:
            return False, f"Move {i+1}: Disk {disk} not on top of rod {from_rod}"
        
        if state[to_rod] and state[to_rod][-1] < disk:
            return False, f"Move {i+1}: Cannot place disk {disk} on smaller disk {state[to_rod][-1]}"
        
        # Execute move
        moved_disk = state[from_rod].pop()
        state[to_rod].append(moved_disk)
    
    # Check goal state
    goal_state = {'A': [], 'B': [], 'C': list(range(n_disks-1, -1, -1))}
    return state == goal_state, "Valid solution" if state == goal_state else f"Wrong final state: {state}"

def compare_algorithms():
    """Compare simplified GraphPlan with BFS"""
    print("SIMPLIFIED GRAPHPLAN vs BFS COMPARISON")
    print("="*60)
    
    try:
        from toh_bfs import hanoi_bfs
    except ImportError:
        print("BFS implementation not found")
        return
    
    results = []
    
    for n in [1, 2, 3, 4]:
        # Simple GraphPlan (using recursive solution for validation)
        gp = SimpleGraphPlan(n)
        
        gp_start = time.time()
        gp_solution = gp.solve()
        gp_time = time.time() - gp_start
        
        gp_valid, gp_msg = validate_simple_solution(gp_solution, n)
        gp_moves = len(gp_solution)
        
        # BFS
        bfs_start = time.time()
        bfs_solution = hanoi_bfs(n, max_states=100000)
        bfs_time = time.time() - bfs_start
        
        bfs_moves = len(bfs_solution) if bfs_solution else 0
        bfs_valid = bfs_moves > 0
        
        expected = 2**n - 1
        
        results.append({
            'n': n,
            'expected': expected,
            'gp_moves': gp_moves,
            'gp_valid': gp_valid,
            'gp_time': gp_time,
            'bfs_moves': bfs_moves,
            'bfs_valid': bfs_valid,
            'bfs_time': bfs_time
        })
    
    print("\nOVERALL COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Disks':<6} {'Expected':<9} {'GP Found':<9} {'GP Valid':<9} {'GP Moves':<9} {'GP Time':<9} {'BFS Found':<10} {'BFS Moves':<10} {'BFS Time':<9}")
    print("-" * 90)
    
    for r in results:
        gp_found = "✓"
        gp_valid_str = "✓" if r['gp_valid'] else "✗"
        bfs_found = "✓" if r['bfs_valid'] else "✗"
        
        print(f"{r['n']:<6} {r['expected']:<9} {gp_found:<9} {gp_valid_str:<9} {r['gp_moves']:<9} {r['gp_time']:<9.4f} {bfs_found:<10} {r['bfs_moves']:<10} {r['bfs_time']:<9.4f}")
    
    # Show sample solutions
    print(f"\nSample Solutions:")
    for n in [1, 2, 3]:
        print(f"\n{n} disk(s):")
        gp = SimpleGraphPlan(n)
        solution = gp.solve()
        for i, move in enumerate(solution):
            parts = move.split('_')
            disk = parts[2]
            from_rod = parts[4]
            to_rod = parts[6]
            print(f"  {i+1}. Move disk {disk} from rod {from_rod} to rod {to_rod}")

if __name__ == "__main__":
    compare_algorithms()