import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque

# =============================================================================
# MULTI-BUS SYSTEM DATA INITIALIZATION
# =============================================================================
# Supports: 33-bus, 34-bus, 69-bus, and 85-bus systems
# [line_no, from_bus, to_bus, r, x]

# ==================== 33-BUS SYSTEM (IEEE) ====================
nt_33 = np.array([
    [1,1,2,0.0922,0.0470], [2,2,3,0.4930,0.2511], [3,3,4,0.3660,0.1864],
    [4,4,5,0.3811,0.1941], [5,5,6,0.8190,0.7070], [6,6,7,0.1872,0.6188],
    [7,7,8,0.7114,0.2351], [8,8,9,1.0300,0.7400], [9,9,10,1.0440,0.7400],
    [10,10,11,0.1966,0.0650], [11,11,12,0.3744,0.1238], [12,12,13,1.4680,1.1550],
    [13,13,14,0.5416,0.7129], [14,14,15,0.5910,0.7837], [15,15,16,0.7463,0.5450],
    [16,16,17,1.2890,1.7210], [17,17,18,0.7320,0.5740], [18,2,19,0.1640,0.1565],
    [19,19,20,1.5042,1.3554], [20,20,21,0.4095,0.4784], [21,21,22,0.7089,0.9373],
    [22,3,23,0.4512,0.3083], [23,23,24,0.8980,0.7091], [24,24,25,0.8960,0.7011],
    [25,6,26,0.2842,0.2235], [26,26,27,1.2591,0.9850], [27,27,28,0.9042,0.7006],
    [28,28,29,0.5075,0.2585], [29,29,30,0.9744,0.9629], [30,30,31,0.3105,0.3619],
    [31,31,32,0.3619,0.1900], [32,32,33,0.7837,0.2642]
])

pq_load_33 = np.array([
    [2,100,60], [3,90,40], [4,120,80], [5,60,30], [6,300,150],
    [7,200,100], [8,150,70], [9,210,100], [10,60,20], [11,45,30],
    [12,60,35], [13,60,35], [14,120,80], [15,60,10], [16,60,20],
    [17,90,40], [18,90,40], [19,90,40], [20,90,40], [21,90,40],
    [22,90,40], [23,420,200], [24,420,200], [25,60,25], [26,60,25],
    [27,60,20], [28,120,70], [29,200,100], [30,150,70], [31,210,100],
    [32,60,40], [33,150,70]
])

# ==================== 34-BUS SYSTEM (IEEE) ====================
nt_34 = np.array([
    [1,1,2,0.0089,0.0056], [2,2,3,0.0139,0.0110], [3,3,4,0.0111,0.0088],
    [4,4,5,0.0111,0.0088], [5,5,6,0.0178,0.0141], [6,6,7,0.0089,0.0070],
    [7,7,8,0.0089,0.0070], [8,8,9,0.0356,0.0283], [9,9,10,0.0356,0.0283],
    [10,10,11,0.0356,0.0283], [11,11,12,0.0267,0.0212], [12,12,13,0.0133,0.0106],
    [13,13,14,0.0089,0.0070], [14,14,15,0.0089,0.0070], [15,15,16,0.0267,0.0212],
    [16,16,17,0.0178,0.0141], [17,17,18,0.0356,0.0283], [18,18,19,0.0267,0.0212],
    [19,19,20,0.0178,0.0141], [20,20,21,0.0356,0.0283], [21,21,22,0.0178,0.0141],
    [22,22,23,0.0178,0.0141], [23,23,24,0.0267,0.0212], [24,2,25,0.0089,0.0070],
    [25,25,26,0.0178,0.0141], [26,26,27,0.0133,0.0106], [27,27,28,0.0089,0.0070],
    [28,28,29,0.0356,0.0283], [29,29,30,0.0267,0.0212], [30,30,31,0.0356,0.0283],
    [31,31,32,0.0178,0.0141], [32,32,33,0.0089,0.0070], [33,33,34,0.0356,0.0283]
])

pq_load_34 = np.array([
    [2,55,25], [3,30,10], [4,20,10], [5,60,25], [6,25,12],
    [7,65,29], [8,40,20], [9,75,35], [10,30,22], [11,28,5],
    [12,16,8], [13,82,28], [14,45,14], [15,48,20], [16,42,28],
    [17,60,20], [18,45,30], [19,18,5], [20,12,8], [21,82,55],
    [22,98,10], [23,15,10], [24,63,30], [25,72,14], [26,8,5],
    [27,8,5], [28,60,35], [29,45,22], [30,30,15], [31,58,20],
    [32,48,20], [33,32,16], [34,48,20]
])

# ==================== 15-BUS SYSTEM ====================
nt_15 = np.array([
    [1,1,2,0.0922,0.0470], [2,2,3,0.4930,0.2511], [3,3,4,0.3660,0.1864],
    [4,4,5,0.3811,0.1941], [5,2,6,0.8190,0.7070], [6,6,7,0.1872,0.6188],
    [7,7,8,0.7114,0.2351], [8,4,9,1.0300,0.7400], [9,9,10,1.0440,0.7400],
    [10,10,11,0.1966,0.0650], [11,5,12,0.3744,0.1238], [12,12,13,1.4680,1.1550],
    [13,8,14,0.5075,0.2585], [14,14,15,0.7837,0.2642]
])

pq_load_15 = np.array([
    [2,100,60], [3,90,40], [4,120,80], [5,60,30], [6,300,150],
    [7,200,100], [8,150,70], [9,210,100], [10,60,20], [11,45,30],
    [12,60,35], [13,60,35], [14,120,80], [15,60,10]
])

# ==================== 69-BUS SYSTEM (IEEE) ====================
nt_69 = np.array([
    [1,1,2,0.0005,0.0012], [2,2,3,0.0005,0.0012], [3,3,4,0.0015,0.0036],
    [4,4,5,0.0251,0.0294], [5,5,6,0.3660,0.1864], [6,6,7,0.3811,0.1941],
    [7,7,8,0.0922,0.0470], [8,8,9,0.4930,0.2511], [9,9,10,0.8119,0.2152],
    [10,10,11,0.2093,0.1069], [11,11,12,0.2129,0.1138], [12,12,13,0.6837,0.2685],
    [13,13,14,0.4414,0.1957], [14,14,15,0.5960,0.2476], [15,15,16,0.7463,0.5450],
    [16,16,17,1.2890,1.7210], [17,17,18,0.7320,0.5740], [18,18,19,0.5416,0.7129],
    [19,19,20,0.9498,0.7446], [20,20,21,0.7640,0.6006], [21,21,22,0.8250,0.7023],
    [22,22,23,0.8250,0.7023], [23,23,24,0.8951,0.7837], [24,24,25,0.8957,0.7802],
    [25,25,26,0.8041,0.6837], [26,26,27,0.5075,0.2585], [27,27,28,0.9744,0.9629],
    [28,28,29,0.8762,0.7091], [29,29,30,0.3105,0.3619], [30,30,31,0.3619,0.1900],
    [31,31,32,0.0044,0.0108], [32,32,33,0.4512,0.3083], [33,33,34,0.8980,0.7091],
    [34,34,35,0.8957,0.7802], [35,35,36,0.2012,0.0611], [36,36,37,0.2565,0.1738],
    [37,37,38,0.4024,0.1316], [38,38,39,0.2011,0.0611], [39,39,40,0.2565,0.1738],
    [40,40,41,0.2014,0.0611], [41,41,42,0.7087,0.9373], [42,42,43,0.9744,0.9629],
    [43,43,44,0.4512,0.3083], [44,44,45,0.8980,0.7091], [45,45,46,0.8957,0.7802],
    [46,46,47,0.8762,0.7091], [47,47,48,0.2012,0.0611], [48,48,49,0.5075,0.2585],
    [49,49,50,0.8762,0.7091], [50,8,51,0.0922,0.0470], [51,51,52,0.4930,0.2511],
    [52,52,53,0.4050,0.1325], [53,53,54,0.7838,0.2663], [54,54,55,0.3041,0.0993],
    [55,55,56,0.3410,0.1110], [56,56,57,0.5650,0.1845], [57,57,58,0.5310,0.1732],
    [58,58,59,0.6747,0.2198], [59,59,60,1.3300,0.4346], [60,60,61,1.3830,0.4506],
    [61,61,62,0.1872,0.6188], [62,62,63,0.7114,0.2351], [63,63,64,1.0300,0.7400],
    [64,64,65,1.0440,0.7400], [65,65,66,0.1966,0.0650], [66,66,67,0.3744,0.1238],
    [67,67,68,0.0047,0.0047], [68,68,69,0.7394,0.2444]
])

pq_load_69 = np.array([
    [2,100,60], [3,90,40], [4,120,80], [5,60,30], [6,300,150],
    [7,200,100], [8,150,70], [9,210,100], [10,60,20], [11,45,30],
    [12,60,35], [13,60,35], [14,120,80], [15,60,10], [16,60,20],
    [17,90,40], [18,90,40], [19,90,40], [20,90,40], [21,90,40],
    [22,90,40], [23,420,200], [24,420,200], [25,60,25], [26,60,25],
    [27,60,20], [28,120,70], [29,200,100], [30,150,70], [31,210,100],
    [32,60,40], [33,150,70], [34,60,20], [35,60,20], [36,60,25],
    [37,60,25], [38,120,70], [39,200,100], [40,150,70], [41,210,100],
    [42,60,40], [43,150,70], [44,60,20], [45,60,20], [46,60,25],
    [47,60,25], [48,45,30], [49,60,35], [50,90,40], [51,30,15],
    [52,60,35], [53,60,35], [54,120,80], [55,60,10], [56,60,20],
    [57,90,40], [58,90,40], [59,90,40], [60,90,40], [61,90,40],
    [62,420,200], [63,420,200], [64,60,25], [65,60,25], [66,60,20],
    [67,120,70], [68,200,100], [69,150,70]
])

# ==================== 85-BUS SYSTEM ====================
nt_85 = np.array([
    [1,1,2,0.108,0.075], [2,2,3,0.163,0.112], [3,3,4,0.217,0.149], [4,4,5,0.108,0.074],
    [5,5,6,0.435,0.298], [6,6,7,0.272,0.186], [7,7,8,1.197,0.820], [8,8,9,0.108,0.074],
    [9,9,10,0.598,0.410], [10,10,11,0.544,0.373], [11,11,12,0.544,0.373], [12,12,13,0.598,0.410],
    [13,13,14,0.272,0.186], [14,14,15,0.326,0.223], [15,2,16,0.728,0.302], [16,3,17,0.455,0.189],
    [17,5,18,0.820,0.340], [18,18,19,0.637,0.264], [19,19,20,0.455,0.189], [20,20,21,0.819,0.340],
    [21,21,22,1.548,0.642], [22,19,23,0.182,0.075], [23,7,24,0.910,0.378], [24,8,25,0.455,0.189],
    [25,25,26,0.364,0.151], [26,26,27,0.546,0.226], [27,27,28,0.273,0.113], [28,28,29,0.546,0.226],
    [29,29,30,0.546,0.226], [30,30,31,0.273,0.113], [31,31,32,0.182,0.075], [32,32,33,0.182,0.075],
    [33,33,34,0.819,0.340], [34,34,35,0.637,0.264], [35,35,36,0.182,0.075], [36,26,37,0.364,0.151],
    [37,27,38,1.002,0.416], [38,29,39,0.546,0.226], [39,32,40,0.455,0.189], [40,40,41,1.002,0.416],
    [41,41,42,0.273,0.113], [42,41,43,0.455,0.189], [43,34,44,1.002,0.416], [44,44,45,0.911,0.378],
    [45,45,46,0.911,0.378], [46,46,47,0.546,0.226], [47,35,48,0.637,0.264], [48,48,49,0.182,0.075],
    [49,49,50,0.364,0.151], [50,50,51,0.455,0.189], [51,48,52,1.366,0.567], [52,52,53,0.455,0.189],
    [53,53,54,0.546,0.226], [54,52,55,0.546,0.226], [55,49,56,0.546,0.226], [56,9,57,0.273,0.113],
    [57,57,58,0.819,0.340], [58,58,59,0.182,0.075], [59,58,60,0.546,0.226], [60,60,61,0.728,0.302],
    [61,61,62,1.002,0.415], [62,60,63,0.182,0.075], [63,63,64,0.728,0.302], [64,64,65,0.182,0.075],
    [65,65,66,0.182,0.075], [66,64,67,0.455,0.189], [67,67,68,0.910,0.378], [68,68,69,1.092,0.453],
    [69,69,70,0.455,0.189], [70,70,71,0.546,0.226], [71,67,72,0.182,0.075], [72,68,73,1.184,0.419],
    [73,73,74,0.273,0.113], [74,73,75,1.002,0.416], [75,70,76,0.546,0.226], [76,65,77,0.910,0.370],
    [77,10,78,0.637,0.264], [78,67,79,0.546,0.226], [79,12,80,0.728,0.302], [80,80,81,0.364,0.151],
    [81,81,82,0.910,0.370], [82,81,83,1.092,0.453], [83,83,84,1.002,0.416], [84,13,85,0.819,0.340]
])

pq_load_85 = np.array([
    [4,56,57.13],[6,35.28,35.99],[8,35.28,35.99],[11,56,57.13],[14,35.28,35.99],[15,35.28,35.99],
    [16,35.28,35.99],[17,112,114.26],[18,56,57.13],[19,56,57.13],[20,35.28,35.99],[21,35.28,35.99],
    [22,35.28,35.99],[23,56,57.13],[24,35.28,35.99],[25,35.28,35.99],[26,56,57.13],[28,56,57.13],
    [30,35.28,35.99],[31,35.28,35.99],[33,14,14.28],[36,35.28,35.99],[37,56,57.13],[38,56,57.13],
    [39,56,57.13],[40,35.28,35.99],[42,35.28,35.99],[43,35.28,35.99],[44,35.28,35.99],[45,35.28,35.99],
    [46,35.28,35.99],[47,14,14.28],[50,36.28,37.01],[51,56,57.13],[53,35.28,35.99],[54,56,57.13],
    [55,56,57.13],[56,14,14.28],[57,56,57.13],[59,56,57.13],[60,56,57.13],[61,56,57.13],[62,56,57.13],
    [63,14,14.28],[66,56,57.13],[69,56,57.13],[71,35.28,35.99],[72,56,57.13],[74,56,57.13],[75,35.28,35.99],
    [76,56,57.13],[77,14,14.28],[78,56,57.13],[79,35.28,35.99],[80,56,57.13],[82,56,57.13],[83,35.28,35.99],
    [84,14,14.28],[85,35.28,35.99]
])

# =============================================================================
# SYSTEM CONFIGURATIONS DICTIONARY
# =============================================================================
BUS_SYSTEMS = {
    '33': {
        'network': nt_33,
        'loads': pq_load_33,
        'num_buses': 33,
        'num_lines': 32,
        'base_kv': 12.66,
        'base_kva': 100.0,
        'default_buses': [6, 14, 24, 30]
    },
    '34': {
        'network': nt_34,
        'loads': pq_load_34,
        'num_buses': 34,
        'num_lines': 33,
        'base_kv': 24.9,
        'base_kva': 100.0,
        'default_buses': [8, 16, 23, 32]
    },
    '15': {
        'network': nt_15,
        'loads': pq_load_15,
        'num_buses': 15,
        'num_lines': 14,
        'base_kv': 11.0,
        'base_kva': 100.0,
        'default_buses': [5, 9, 12, 15]
    },
    '69': {
        'network': nt_69,
        'loads': pq_load_69,
        'num_buses': 69,
        'num_lines': 68,
        'base_kv': 12.66,
        'base_kva': 100.0,
        'default_buses': [12, 15, 42, 62]
    },
    '85': {
        'network': nt_85,
        'loads': pq_load_85,
        'num_buses': 85,
        'num_lines': 84,
        'base_kv': 11.0,
        'base_kva': 100.0,
        'default_buses': [3, 39, 2, 18]
    }
}

def get_system_data(system_id='85'):
    """Returns network topology, loads, and parameters for specified system"""
    if system_id not in BUS_SYSTEMS:
        system_id = '85'
    return BUS_SYSTEMS[system_id]

# Default system (85-bus)
DEFAULT_SYSTEM = BUS_SYSTEMS['85']
BASE_KV = DEFAULT_SYSTEM['base_kv']
BASE_KVA = DEFAULT_SYSTEM['base_kva']
Z_BASE = (BASE_KV**2 * 1000) / BASE_KVA
nt = DEFAULT_SYSTEM['network']
pq_load_data = DEFAULT_SYSTEM['loads']

def load_flow(q_caps, bus_indices=None, system_id='85'):
    """Calculates total power loss (kW), bus voltages, and branch power flows for specified system."""
    
    sys_config = get_system_data(system_id)
    network = sys_config['network']
    loads = sys_config['loads']
    n_bus = sys_config['num_buses']
    n_lines = sys_config['num_lines']
    base_kv = sys_config['base_kv']
    base_kva = sys_config['base_kva']
    z_base = (base_kv**2 * 1000) / base_kva
    
    if bus_indices is None:
        bus_indices = sys_config['default_buses']
    
    p_load = np.zeros(n_bus)
    q_load = np.zeros(n_bus)
    for row in loads:
        p_load[int(row[0]) - 1] = row[1]
        q_load[int(row[0]) - 1] = row[2]
    
    for idx, bus_no in enumerate(bus_indices):
        q_load[int(bus_no) - 1] -= q_caps[idx]
    
    P = p_load / base_kva
    Q = q_load / base_kva
    R = network[:, 3] / z_base
    X = network[:, 4] / z_base
    f_bus = network[:, 1].astype(int) - 1
    t_bus = network[:, 2].astype(int) - 1

    V = np.ones(n_bus, dtype=complex)
    pl_loss = np.zeros(n_lines, dtype=float)
    I_br = np.zeros(n_lines, dtype=complex)

    children = {bus: [] for bus in range(n_bus)}
    for idx, parent in enumerate(f_bus):
        children[parent].append(idx)

    bus_depth = np.zeros(n_bus, dtype=int)
    queue = [0]
    while queue:
        node = queue.pop(0)
        for branch_idx in children[node]:
            child = t_bus[branch_idx]
            bus_depth[child] = bus_depth[node] + 1
            queue.append(child)
    branch_depth = bus_depth[t_bus]
    backward_order = np.argsort(-branch_depth)
    forward_order = np.argsort(branch_depth)

    for _ in range(50):
        V_prev = V.copy()
        I_load = np.zeros(n_bus, dtype=complex)
        for bus in range(1, n_bus):
            if np.abs(V[bus]) < 1e-6:
                V[bus] = 1.0 + 0j
            S_bus = P[bus] + 1j * Q[bus]
            I_load[bus] = np.conj(S_bus / V[bus])

        I_br.fill(0.0)
        for idx in backward_order:
            node = t_bus[idx]
            I_br[idx] = I_load[node] + np.sum(I_br[children[node]])

        for idx in forward_order:
            sending = f_bus[idx]
            receiving = t_bus[idx]
            V[receiving] = V[sending] - (R[idx] + 1j * X[idx]) * I_br[idx]

        if np.max(np.abs(V - V_prev)) < 1e-6:
            break

    for idx in range(n_lines):
        pl_loss[idx] = R[idx] * (np.abs(I_br[idx]) ** 2) * base_kva

    S_br = V[f_bus] * np.conj(I_br)
    P_branch = S_br.real
    Q_branch = S_br.imag

    return np.sum(pl_loss), np.abs(V), P_branch, Q_branch


def get_bus_layout(network):
    """Create IEEE-style schematic layout with main trunk and lateral branches."""
    children = {}
    for row in network:
        parent = int(row[1])
        child = int(row[2])
        children.setdefault(parent, []).append(child)

    # BFS to identify trunk vs lateral branches
    positions = {1: (0, 0)}  # Substation at origin
    visited = {1}
    trunk = [1]  # Main feeder path
    branch_spacing = 1.0 if len(network) <= 40 else 1.5
    lateral_offset = 2 if len(network) <= 40 else 3
    
    # Find longest path as main trunk
    def get_longest_path(node, depth=0):
        if not children.get(node, []):
            return [node], depth
        longest_child_path = [], 0
        for child in sorted(children.get(node, [])):
            path, d = get_longest_path(child, depth + 1)
            if len(path) > len(longest_child_path[0]):
                longest_child_path = path, d
        return [node] + longest_child_path[0], longest_child_path[1]
    
    trunk, _ = get_longest_path(1)
    visited.update(trunk)
    
    is_69 = len(network) == 68
    trunk_spacing = 5.0 if is_69 else 2.0 if len(network) > 40 else 1.2
    branch_spacing = 4.5 if is_69 else 2.2 if len(network) > 40 else 1.2
    trunk_x_base = -6.0 if is_69 else 0.0

    # Position trunk vertically
    for idx, bus in enumerate(trunk):
        positions[bus] = (trunk_x_base, -idx * trunk_spacing)
    
    # Position lateral branches
    lateral_offset = 7 if len(network) > 40 else 3
    for trunk_bus in trunk:
        laterals = [c for c in sorted(children.get(trunk_bus, [])) if c not in visited]
        if not laterals:
            continue
        root_y = positions[trunk_bus][1]
        for lat_idx, lateral_root in enumerate(laterals):
            # BFS for each lateral subtree
            lat_queue = deque([(lateral_root, 0)])
            if is_69:
                branch_x = 14.0
                branch_x_step = 0.0
                branch_start_y = root_y - 2.5
            else:
                branch_x = lateral_offset * (1 if lat_idx % 2 == 0 else -1)
                branch_x_step = 0.0
                branch_start_y = root_y
            while lat_queue:
                node, lat_depth = lat_queue.popleft()
                visited.add(node)
                positions[node] = (branch_x + lat_depth * branch_x_step, branch_start_y - lat_depth * branch_spacing)
                for child in sorted(children.get(node, [])):
                    if child not in visited:
                        lat_queue.append((child, lat_depth + 1))
            lateral_offset += 2
    
    return positions


def get_bus_diagram_data(system_id='85'):
    """Return node and edge data for the selected IEEE bus system."""
    sys_config = get_system_data(system_id)
    network = sys_config['network']
    loads = sys_config['loads']
    default_buses = sys_config['default_buses']

    load_map = {int(row[0]): (float(row[1]), float(row[2])) for row in loads}
    nodes = []
    for bus in range(1, sys_config['num_buses'] + 1):
        p, q = load_map.get(bus, (0.0, 0.0))
        nodes.append({
            'bus': bus,
            'p_mw': p,
            'q_mvar': q,
            'has_load': p != 0.0 or q != 0.0,
            'is_candidate': bus in default_buses
        })

    edges = [
        {
            'from_bus': int(row[1]),
            'to_bus': int(row[2]),
            'r': float(row[3]),
            'x': float(row[4])
        }
        for row in network
    ]

    positions = get_bus_layout(network)
    return {
        'system_id': system_id,
        'nodes': nodes,
        'edges': edges,
        'positions': positions,
        'default_buses': default_buses
    }


def plot_bus_diagram(system_id='85', figsize=(16, 12), show=True, save_path=None):
    """Plot IEEE-standard schematic bus diagram matching PDF reference style."""
    data = get_bus_diagram_data(system_id)
    positions = data['positions']

    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw substation box at top
    sub_x, sub_y = positions[1]
    rect = plt.Rectangle((sub_x - 0.4, sub_y + 0.4), 0.8, 0.5, 
                          fill=True, facecolor='lightblue', edgecolor='black', linewidth=2.5)
    ax.add_patch(rect)
    ax.text(sub_x, sub_y + 0.65, 'Substation', ha='center', va='center', fontweight='bold', fontsize=11)
    
    # Draw edges as line segments
    for edge in data['edges']:
        start = positions[edge['from_bus']]
        end = positions[edge['to_bus']]
        if abs(start[0] - end[0]) > 0.2 and abs(start[1] - end[1]) > 0.2:
            if abs(start[0] - end[0]) > abs(start[1] - end[1]):
                elbow = (end[0], start[1])
            else:
                elbow = (start[0], end[1])
            ax.plot([start[0], elbow[0]], [start[1], elbow[1]], 'k-', linewidth=2.5, zorder=1)
            ax.plot([elbow[0], end[0]], [elbow[1], end[1]], 'k-', linewidth=2.5, zorder=1)
        else:
            ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=2.5, zorder=1)
    
    # Draw buses as numbered circles (single-line diagram style)
    for node in data['nodes']:
        x, y = positions.get(node['bus'], (0.0, 0.0))
        
        # Bus circle - small filled circle with number inside
        circle = plt.Circle((x, y), 0.35, color='white', ec='black', linewidth=2, zorder=3, fill=True)
        ax.add_patch(circle)
        
        # Bus number positioned to the right or left depending on branch side
        if x >= 0:
            label_x = x + 0.55
            label_ha = 'left'
        else:
            label_x = x - 0.55
            label_ha = 'right'
        ax.text(label_x, y, str(node['bus']), ha=label_ha, va='center', 
               fontweight='bold', fontsize=10, zorder=4,
               bbox=dict(boxstyle='circle,pad=0.25', facecolor='white', edgecolor='black', linewidth=1.5))
        
        # Load symbol (T-shaped) - drawn below bus
        if node['has_load']:
            # Vertical line
            ax.plot([x, x], [y - 0.35, y - 0.85], 'k-', linewidth=2, zorder=2)
            # Horizontal crossbar
            ax.plot([x - 0.2, x + 0.2], [y - 0.55, y - 0.55], 'k-', linewidth=2, zorder=2)
            
            # Load value label
            load_text = f"{node['p_mw']:.0f}MW\n{node['q_mvar']:.0f}MVAR"
            ax.text(x, y - 1.0, load_text, fontsize=7, ha='center', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.9))
        
        # Capacitor indicator (C symbol)
        if node['is_candidate']:
            ax.text(x + 0.6, y - 0.55, 'C', ha='left', va='center', fontweight='bold', 
                   fontsize=9, color='green',
                   bbox=dict(boxstyle='circle,pad=0.2', facecolor='lightgreen', edgecolor='darkgreen', linewidth=1.5))
    
    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', linewidth=2.5, label='Transmission Line'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
               markeredgecolor='black', markersize=8, label='Bus', markeredgewidth=2),
        Line2D([0], [0], marker='_', color='k', linewidth=2, markersize=15, label='Load (T-symbol)', linestyle='None'),
        Patch(facecolor='lightgreen', edgecolor='darkgreen', label='Capacitor Location')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.95)
    
    ax.set_title(f"IEEE {system_id}-Bus System - Single Line Diagram", fontsize=14, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')
    ax.margins(0.15)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_power_flow_diagram(system_id='85', q_caps=None, bus_indices=None, figsize=(16, 10), show=True, save_path=None):
    """Plot a standard IEEE bus system diagram showing branch power flows and bus injections."""
    if q_caps is None:
        q_caps = [0] * len(get_system_data(system_id)['default_buses'])
    if bus_indices is None:
        bus_indices = get_system_data(system_id)['default_buses']
    
    # Get system data
    sys_config = get_system_data(system_id)
    network = sys_config['network']
    loads = sys_config['loads']
    base_kva = sys_config['base_kva']
    
    # Run load flow to get power flows
    _, voltages, P_br, Q_br = load_flow(q_caps, bus_indices, system_id)
    
    # Get diagram data
    data = get_bus_diagram_data(system_id)
    positions = data['positions']
    
    # Calculate bus injections (loads - capacitors)
    p_load = np.zeros(sys_config['num_buses'])
    q_load = np.zeros(sys_config['num_buses'])
    for row in loads:
        p_load[int(row[0])-1] = row[1]
        q_load[int(row[0])-1] = row[2]
    
    # Apply capacitors
    for idx, bus_no in enumerate(bus_indices):
        q_load[int(bus_no) - 1] -= q_caps[idx]
    
    # Create figure
    plt.figure(figsize=figsize)
    ax = plt.gca()
    
    # Draw edges with power flows
    for i, edge in enumerate(data['edges']):
        start = positions[edge['from_bus']]
        end = positions[edge['to_bus']]
        
        # Draw line
        ax.plot([start[0], end[0]], [start[1], end[1]], color='black', linewidth=2)
        
        # Add power flow label
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        p_flow = P_br[i]
        q_flow = Q_br[i]
        flow_label = f"{p_flow:.1f}/{q_flow:.1f}"
        ax.text(mid_x, mid_y + 0.1, flow_label, fontsize=8, ha='center', va='bottom', 
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="black"))
        
        # Add arrow indicating direction (from -> to)
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        arrow_x = start[0] + dx * 0.7
        arrow_y = start[1] + dy * 0.7
        ax.arrow(arrow_x, arrow_y, dx * 0.1, dy * 0.1, head_width=0.05, head_length=0.05, 
                fc='blue', ec='blue', length_includes_head=True)
    
    # Draw nodes with injections
    for node in data['nodes']:
        x, y = positions.get(node['bus'], (0.0, 0.0))
        
        # Determine color
        if node['bus'] in bus_indices:
            color = 'green'  # Capacitor bus
            edge_color = 'black'
        elif node['has_load']:
            color = 'red'   # Load bus
            edge_color = 'black'
        else:
            color = 'blue'  # Slack or intermediate
            edge_color = 'none'
        
        ax.scatter([x], [y], s=300, c=color, edgecolors=edge_color, zorder=3)
        
        # Bus label
        label = f"Bus {node['bus']}\nV={voltages[node['bus']-1]:.3f} p.u."
        
        # Add injection if any
        p_inj = p_load[node['bus']-1]
        q_inj = q_load[node['bus']-1]
        if p_inj != 0 or q_inj != 0:
            label += f"\nP/Q: {p_inj:.0f}/{q_inj:.0f}"
        
        ax.text(x, y, label, fontsize=7, ha='center', va='center', color='white', zorder=4)
    
    ax.set_title(f"IEEE {system_id}-Bus System Power Flow Diagram\nBranch Flows (MW/MVAR), Bus Injections (MW/MVAR)")
    ax.axis('off')
    ax.set_aspect('equal')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def load_flow_33(q_caps, bus_indices=None):
    """Wrapper for 33-bus system load flow calculation."""
    loss, v, p_br, q_br = load_flow(q_caps, bus_indices=bus_indices, system_id='33')
    return loss, v, p_br, q_br


def load_flow_34(q_caps, bus_indices=None):
    """Wrapper for 34-bus system load flow calculation."""
    loss, v, p_br, q_br = load_flow(q_caps, bus_indices=bus_indices, system_id='34')
    return loss, v, p_br, q_br


def load_flow_69(q_caps, bus_indices=None):
    """Wrapper for 69-bus system load flow calculation."""
    loss, v, p_br, q_br = load_flow(q_caps, bus_indices=bus_indices, system_id='69')
    return loss, v, p_br, q_br

# =============================================================================
# 2. PSO OPTIMIZATION SETTINGS
# =============================================================================
if __name__ == "__main__":
    num_particles = 20
    max_iter = 15
    dim = 4
    v_max = 110.0
    w, c1, c2 = 0.7, 2.05, 2.05

    # Initialize
    pos = np.random.uniform(0, 1100, (num_particles, dim))
    vel = np.random.uniform(-v_max, v_max, (num_particles, dim))
    pbest_pos = pos.copy()
    pbest_val = np.array([load_flow(p)[0] for p in pos])
    gbest_pos = pbest_pos[np.argmin(pbest_val)]
    gbest_val = np.min(pbest_val)

    convergence = []

    print("Running PSO for Optimal Capacitor Placement...")

    for i in range(max_iter):
        for j in range(num_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            vel[j] = w*vel[j] + c1*r1*(pbest_pos[j] - pos[j]) + c2*r2*(gbest_pos - pos[j])
            vel[j] = np.clip(vel[j], -v_max, v_max)
            pos[j] = np.clip(pos[j] + vel[j], 0, 1100)
            
            current_loss, _, _, _ = load_flow(pos[j])
            if current_loss < pbest_val[j]:
                pbest_val[j] = current_loss
                pbest_pos[j] = pos[j].copy()
                
        if np.min(pbest_val) < gbest_val:
            gbest_val = np.min(pbest_val)
            gbest_pos = pbest_pos[np.argmin(pbest_val)].copy()
            
        convergence.append(gbest_val)
        print(f"Iter {i+1}: Best Loss = {gbest_val:.4f} kW")

    # Final Results
    initial_loss, base_v, _, _ = load_flow([0,0,0,0])
    final_loss, final_v, _, _ = load_flow(gbest_pos)

    print("\n" + "="*40)
    print(f"Optimal Capacitor at Bus 3:  {gbest_pos[0]:.2f} kVAR")
    print(f"Optimal Capacitor at Bus 39: {gbest_pos[1]:.2f} kVAR")
    print(f"Optimal Capacitor at Bus 2:  {gbest_pos[2]:.2f} kVAR")
    print(f"Optimal Capacitor at Bus 18: {gbest_pos[3]:.2f} kVAR")
    print("-" * 40)
    print(f"Initial Power Loss: {initial_loss:.4f} kW")
    print(f"Optimal Power Loss: {final_loss:.4f} kW")
    print(f"Loss Reduction:     {((initial_loss-final_loss)/initial_loss)*100:.2f}%")
    print("="*40)

    # =============================================================================
    # 4. POWER FLOW DIAGRAM
    # =============================================================================
    print("Generating power flow diagram...")
    plot_power_flow_diagram(system_id='85', q_caps=gbest_pos, bus_indices=[3,39,2,18], save_path='power_flow_diagram.png')

    # =============================================================================
    # 3. PLOTTING
    # =============================================================================
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(convergence, 'r-o')
    plt.title("PSO Convergence")
    plt.xlabel("Iteration"); plt.ylabel("Loss (kW)")

    plt.subplot(1, 2, 2)
    plt.plot(base_v, 'k--', label='Base Case')
    plt.plot(final_v, 'b-', label='With Capacitors')
    plt.title("Voltage Profile"); plt.legend()
    plt.xlabel("Bus Number"); plt.ylabel("Voltage (p.u.)")

    plt.tight_layout()
    plt.show()