from collections import deque
import time
import struct
import os
import pickle

U_CORNER_PERM = [3, 0, 1, 2, 4, 5, 6, 7]
U_CORNER_ORI  = [0]*8
U_EDGE_PERM   = [3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11]
U_EDGE_ORI    = [0]*12

D_CORNER_PERM = [0, 1, 2, 3, 5, 6, 7, 4]
D_CORNER_ORI  = [0]*8
D_EDGE_PERM   = [0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11]
D_EDGE_ORI    = [0]*12

F_CORNER_PERM = [1, 5, 2, 3, 0, 4, 6, 7]
F_CORNER_ORI  = [2, 1, 0, 0, 1, 2, 0, 0]
F_EDGE_PERM   = [9, 1, 2, 3, 8, 5, 6, 7, 0, 4, 10, 11]
F_EDGE_ORI    = [1,0,0,0,1,0,0,0,1,1,0,0]

B_CORNER_PERM = [0, 1, 3, 7, 4, 5, 2, 6]
B_CORNER_ORI  = [0,0,2,1,0,0,1,2]
B_EDGE_PERM   = [0,1,11,3,4,5,10,7,8,9,2,6]
B_EDGE_ORI    = [0,0,1,0,0,0,1,0,0,0,1,1]

R_CORNER_PERM = [0,2,6,3,4,1,5,7]
R_CORNER_ORI  = [0,1,2,0,0,2,1,0]
R_EDGE_PERM   = [0,8,2,3,4,9,6,7,5,1,10,11]
R_EDGE_ORI    = [0]*12

L_CORNER_PERM = [4,1,2,0,7,5,6,3]
L_CORNER_ORI  = [1,0,0,2,2,0,0,1]
L_EDGE_PERM   = [0,1,2,10,4,5,6,11,8,9,7,3]
L_EDGE_ORI    = [0]*12

MOVES = {
    'U': (U_CORNER_PERM, U_CORNER_ORI, U_EDGE_PERM, U_EDGE_ORI),
    'D': (D_CORNER_PERM, D_CORNER_ORI, D_EDGE_PERM, D_EDGE_ORI),
    'F': (F_CORNER_PERM, F_CORNER_ORI, F_EDGE_PERM, F_EDGE_ORI),
    'B': (B_CORNER_PERM, B_CORNER_ORI, B_EDGE_PERM, B_EDGE_ORI),
    'R': (R_CORNER_PERM, R_CORNER_ORI, R_EDGE_PERM, R_EDGE_ORI),
    'L': (L_CORNER_PERM, L_CORNER_ORI, L_EDGE_PERM, L_EDGE_ORI),
}

MOVE_NAMES = ['U', 'D', 'F', 'B', 'R', 'L']

def compose_moves(move1, move2):
    """
    Compose two moves to create a new move.
    Returns (corner_perm, corner_ori, edge_perm, edge_ori)
    """
    cperm1, cori1, eperm1, eori1 = move1
    cperm2, cori2, eperm2, eori2 = move2
    
    # Apply move1 then move2 for corners
    new_cperm = [cperm2[cperm1[i]] for i in range(8)]
    new_cori = [(cori1[i] + cori2[cperm1[i]]) % 3 for i in range(8)]
    
    # Apply move1 then move2 for edges
    new_eperm = [eperm2[eperm1[i]] for i in range(12)]
    new_eori = [(eori1[i] + eori2[eperm1[i]]) % 2 for i in range(12)]
    
    return (new_cperm, new_cori, new_eperm, new_eori)


# Generate double moves (applying single move twice)
F2_MOVE = compose_moves(
    MOVES['F'],
    MOVES['F']
)

B2_MOVE = compose_moves(
    MOVES['B'],
    MOVES['B']
)

R2_MOVE = compose_moves(
    MOVES['R'],
    MOVES['R']
)

L2_MOVE = compose_moves(
    MOVES['L'],
    MOVES['L']
)

# Add double moves to MOVES dictionary
MOVES['F2'] = F2_MOVE
MOVES['B2'] = B2_MOVE
MOVES['R2'] = R2_MOVE
MOVES['L2'] = L2_MOVE

# Update MOVE_NAMES to include all moves (for Phase 1)
MOVE_NAMES = ['U', 'D', 'F', 'B', 'R', 'L']

# Create separate list for Phase 2 moves
PHASE2_MOVES = ['U', 'D', 'F2', 'B2', 'R2', 'L2']


class CubeState:
    def __init__(self):
        self.corner_pos = list(range(8))
        self.corner_ori = [0]*8
        self.edge_pos = list(range(12))
        self.edge_ori = [0]*12

    def copy(self):
        c = CubeState()
        c.corner_pos = self.corner_pos.copy()
        c.corner_ori = self.corner_ori.copy()
        c.edge_pos = self.edge_pos.copy()
        c.edge_ori = self.edge_ori.copy()
        return c

    def is_solved(self):
        return (self.corner_pos == list(range(8)) and
                self.corner_ori == [0]*8 and
                self.edge_pos == list(range(12)) and
                self.edge_ori == [0]*12)

    def move(self, cperm, cori, eperm, eori):
        new = self.copy()
        for i in range(8):
            src = cperm[i]
            new.corner_pos[i] = self.corner_pos[src]
            new.corner_ori[i] = (self.corner_ori[src] + cori[i]) % 3
        for i in range(12):
            src = eperm[i]
            new.edge_pos[i] = self.edge_pos[src]
            new.edge_ori[i] = (self.edge_ori[src] + eori[i]) % 2
        return new

    def apply_move(self, move_name):
        cperm, cori, eperm, eori = MOVES[move_name]
        return self.move(cperm, cori, eperm, eori)

    def _state_to_key(self):
        return (tuple(self.corner_pos), tuple(self.corner_ori),
                tuple(self.edge_pos), tuple(self.edge_ori))


def is_g1_state(state):
    """
    G1 Subgroup: State where ALL orientations are correct.
    
    Definition:
    - All corner orientations must be 0 (correct orientation)
    - All edge orientations must be 0 (correct orientation)
    - Positions can be ANYTHING
    
    This is the KEY insight: G1 is purely about ORIENTATION, not POSITION.
    Phase 1 goal: Fix all orientations without worrying about positions.
    """
    # Check all corner orientations are 0
    for ori in state.corner_ori:
        if ori != 0:
            return False
    
    # Check all edge orientations are 0
    for ori in state.edge_ori:
        if ori != 0:
            return False
    
    return True


class Phase1TableBuilder:
    """Build and manage Phase 1 heuristic table"""
    
    def __init__(self, filename="phase1_table.bin"):
        self.filename = filename
        self.table = {}

    def build(self, max_depth=10):
        """
        BFS that explores ALL states but ONLY stores G1 states.
        
        Phase 1 Goal: All edge and corner orientations are correct (but positions can be wrong)
        
        BFS Strategy:
        - Start from solved state (all 0 orientations)
        - Apply all moves to ALL states (G1 and non-G1)
        - Only STORE states that reach G1 (those with correct orientations)
        - Compute shortest distance from solved state to each G1 state
        - This works because we explore ALL paths, not just G1 paths
        
        Key difference: We continue BFS through non-G1 states to find new G1 states,
        we don't terminate early.
        
        max_depth: limits BFS depth (10-12 is typically sufficient)
        """
        print(f"Building Phase 1 table (G1 = all orientations correct)...")
        print(f"Max depth: {max_depth}")
        
        start = CubeState()
        queue = deque([(start, 0)])
        visited = {start._state_to_key()}  # Track ALL visited states
        
        # Only store G1 states in the table
        if is_g1_state(start):
            self.table[start._state_to_key()] = 0
        
        depth_counts = {}
        start_time = time.time()
        
        while queue:
            state, dist = queue.popleft()
            
            if dist >= max_depth:
                continue
            
            for move_name in MOVE_NAMES:
                new_state = state.apply_move(move_name)
                key = new_state._state_to_key()
                
                # Continue BFS through ALL states
                if key not in visited:
                    visited.add(key)
                    queue.append((new_state, dist + 1))
                    
                    # But only STORE G1 states
                    if is_g1_state(new_state):
                        self.table[key] = dist + 1
                        
                        if dist + 1 not in depth_counts:
                            depth_counts[dist + 1] = 0
                        depth_counts[dist + 1] += 1
                        
                        # Progress indicator
                        if len(self.table) % 100000 == 0:
                            elapsed = time.time() - start_time
                            rate = len(self.table) / elapsed
                            print(f"  {len(self.table):,} G1 states | "
                                  f"{rate:.0f} states/sec | "
                                  f"Visited {len(visited):,} total | "
                                  f"Depth {dist+1}")
        
        elapsed = time.time() - start_time
        print(f"\nPhase 1 table built in {elapsed:.2f}s")
        print(f"Total G1 states: {len(self.table):,}")
        
        for d in sorted(depth_counts.keys()):
            print(f"  Depth {d}: {depth_counts[d]:,} states")
        
        return self.table

    def save_binary(self):
        """
        Save table as binary file with compact format:
        [state_hash: 16 bytes][distance: 1 byte]
        """
        print(f"\nSaving to {self.filename}...")
        
        # Convert states to hashable form and save
        with open(self.filename, 'wb') as f:
            # Write header
            f.write(b'PH1T')  # Magic number
            f.write(struct.pack('<I', len(self.table)))  # Number of entries
            
            for state_key, distance in self.table.items():
                # Hash the state tuple to 8 bytes (more compact than full state)
                state_hash = hash(state_key) & 0xFFFFFFFFFFFFFFFF
                
                f.write(struct.pack('<Q', state_hash))  # 8 bytes hash
                f.write(struct.pack('B', distance))       # 1 byte distance
        
        file_size = os.path.getsize(self.filename)
        print(f"Saved {len(self.table):,} states ({file_size:,} bytes)")
        print(f"Average: {file_size / len(self.table):.1f} bytes per state")

    def save_pickle(self):
        """Alternative: save as pickle (slower but more reliable)"""
        pickle_file = self.filename.replace('.bin', '.pkl')
        print(f"\nSaving to {pickle_file}...")
        
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.table, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size = os.path.getsize(pickle_file)
        print(f"Saved {len(self.table):,} states ({file_size:,} bytes)")

    def load_binary(self):
        """Load from binary file"""
        print(f"Loading from {self.filename}...")
        
        if not os.path.exists(self.filename):
            print(f"File not found: {self.filename}")
            return False
        
        self.table = {}
        
        with open(self.filename, 'rb') as f:
            # Read header
            magic = f.read(4)
            if magic != b'PH1T':
                print("Invalid file format")
                return False
            
            count = struct.unpack('<I', f.read(4))[0]
            
            for _ in range(count):
                state_hash = struct.unpack('<Q', f.read(8))[0]
                distance = struct.unpack('B', f.read(1))[0]
                self.table[state_hash] = distance
        
        print(f"Loaded {len(self.table):,} states")
        return True

    def load_pickle(self):
        """Load from pickle file"""
        pickle_file = self.filename.replace('.bin', '.pkl')
        print(f"Loading from {pickle_file}...")
        
        if not os.path.exists(pickle_file):
            print(f"File not found: {pickle_file}")
            return False
        
        with open(pickle_file, 'rb') as f:
            self.table = pickle.load(f)
        
        print(f"Loaded {len(self.table):,} states")
        return True


class Phase2TableBuilder:
    """Build and manage Phase 2 heuristic table"""
    
    def __init__(self, filename="phase2_table.bin"):
        self.filename = filename
        self.table = {}

    def build(self, max_depth=14):
        """
        BFS from solved state to build Phase 2 heuristic table.
        
        Phase 2 Goal: Solve the cube completely from a G1 state
        - Start: G1 state (all orientations correct, positions may be wrong)
        - End: Solved state (all pieces in correct positions)
        - Allowed Moves: ONLY <U, D, F2, B2, R2, L2>
        
        Key insight: These restricted moves preserve the G1 condition 
        (all orientations remain correct), so any reachable state is still G1.
        
        BFS Strategy:
        - Backward search from solved state
        - Only apply the 6 restricted moves (no F, B, R, L single moves)
        - Store distance to solved for every reachable state
        - This gives us: "how many Phase 2 moves to solve from current state"
        
        max_depth: limits BFS depth (14 is standard for Phase 2)
        """
        print(f"Building Phase 2 table (distance to solved with restricted moves)...")
        print(f"Max depth: {max_depth}")
        print(f"Allowed moves: U, D, F2, B2, R2, L2")
        
        start = CubeState()
        queue = deque([(start, 0)])
        self.table[start._state_to_key()] = 0
        
        depth_counts = {0: 1}
        start_time = time.time()
        
        while queue:
            state, dist = queue.popleft()
            
            if dist >= max_depth:
                continue
            
            for move_name in PHASE2_MOVES:
                new_state = state.apply_move(move_name)
                key = new_state._state_to_key()
                
                if key not in self.table:
                    self.table[key] = dist + 1
                    queue.append((new_state, dist + 1))
                    
                    if dist + 1 not in depth_counts:
                        depth_counts[dist + 1] = 0
                    depth_counts[dist + 1] += 1
                    
                    # Progress indicator
                    if len(self.table) % 100000 == 0:
                        elapsed = time.time() - start_time
                        rate = len(self.table) / elapsed
                        print(f"  {len(self.table):,} states | "
                              f"{rate:.0f} states/sec | "
                              f"Depth {dist+1}")
        
        elapsed = time.time() - start_time
        print(f"\nPhase 2 table built in {elapsed:.2f}s")
        print(f"Total states: {len(self.table):,}")
        
        for d in sorted(depth_counts.keys()):
            print(f"  Depth {d}: {depth_counts[d]:,} states")
        
        return self.table

    def save_binary(self):
        """Save table as binary file"""
        print(f"\nSaving to {self.filename}...")
        
        with open(self.filename, 'wb') as f:
            # Write header
            f.write(b'PH2T')  # Magic number
            f.write(struct.pack('<I', len(self.table)))  # Number of entries
            
            for state_key, distance in self.table.items():
                state_hash = hash(state_key) & 0xFFFFFFFFFFFFFFFF
                f.write(struct.pack('<Q', state_hash))  # 8 bytes hash
                f.write(struct.pack('B', distance))      # 1 byte distance
        
        file_size = os.path.getsize(self.filename)
        print(f"Saved {len(self.table):,} states ({file_size:,} bytes)")
        print(f"Average: {file_size / len(self.table):.1f} bytes per state")

    def save_pickle(self):
        """Alternative: save as pickle"""
        pickle_file = self.filename.replace('.bin', '.pkl')
        print(f"\nSaving to {pickle_file}...")
        
        with open(pickle_file, 'wb') as f:
            pickle.dump(self.table, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        file_size = os.path.getsize(pickle_file)
        print(f"Saved {len(self.table):,} states ({file_size:,} bytes)")

    def load_binary(self):
        """Load from binary file"""
        print(f"Loading from {self.filename}...")
        
        if not os.path.exists(self.filename):
            print(f"File not found: {self.filename}")
            return False
        
        self.table = {}
        
        with open(self.filename, 'rb') as f:
            magic = f.read(4)
            if magic != b'PH2T':
                print("Invalid file format")
                return False
            
            count = struct.unpack('<I', f.read(4))[0]
            
            for _ in range(count):
                state_hash = struct.unpack('<Q', f.read(8))[0]
                distance = struct.unpack('B', f.read(1))[0]
                self.table[state_hash] = distance
        
        print(f"Loaded {len(self.table):,} states")
        return True

    def load_pickle(self):
        """Load from pickle file"""
        pickle_file = self.filename.replace('.bin', '.pkl')
        print(f"Loading from {pickle_file}...")
        
        if not os.path.exists(pickle_file):
            print(f"File not found: {pickle_file}")
            return False
        
        with open(pickle_file, 'rb') as f:
            self.table = pickle.load(f)
        
        print(f"Loaded {len(self.table):,} states")
        return True
    
    
# Example usage
if __name__ == "__main__":
    print("="*60)
    print("PHASE 1 TABLE GENERATION")
    print("="*60)
    
    phase1_builder = Phase1TableBuilder(filename="phase1_table.bin")
    phase1_builder.build(max_depth=9)
    phase1_builder.save_binary()
    phase1_builder.save_pickle()
    
    print("\n" + "="*60)
    print("PHASE 2 TABLE GENERATION")
    print("="*60)
    
    phase2_builder = Phase2TableBuilder(filename="phase2_table.bin")
    phase2_builder.build(max_depth=11)
    phase2_builder.save_binary()
    phase2_builder.save_pickle()
    
    print("\n" + "="*60)
    print("BOTH TABLES GENERATED SUCCESSFULLY!")
    print("="*60)
    print("\nYou now have:")
    print("  - phase1_table.bin / phase1_table.pkl")
    print("  - phase2_table.bin / phase2_table.pkl")
    print("\nNext: Implement the solver using these tables.")