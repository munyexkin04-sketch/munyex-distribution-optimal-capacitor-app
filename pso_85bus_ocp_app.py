import tkinter as tk
from tkinter.scrolledtext import ScrolledText
from threading import Thread
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =============================================================================
# 1. DATA INITIALIZATION (85-Bus System)
# =============================================================================
# [line_no, from_bus, to_bus, r, x]
nt = np.array([
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

# [Bus_No, P(kW), Q(kVAR)]
pq_load_data = np.array([
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

BASE_KV = 11.0
BASE_KVA = 100.0
Z_BASE = (BASE_KV**2 * 1000) / BASE_KVA

# Default PSO settings
DEFAULT_NUM_PARTICLES = 20
DEFAULT_MAX_ITER = 15
DEFAULT_V_MAX = 110.0
DEFAULT_W, DEFAULT_C1, DEFAULT_C2 = 0.7, 2.05, 2.05
DEFAULT_QCAP_MAX = 1100.0
DEFAULT_BUS_INDICES = [3, 39, 2, 18]


def load_flow(q_caps, bus_indices=None):
    """Calculate total power loss (kW) and bus voltages for the 85-bus system."""
    if bus_indices is None:
        bus_indices = DEFAULT_BUS_INDICES

    n_bus = 85
    p_load = np.zeros(n_bus)
    q_load = np.zeros(n_bus)

    for row in pq_load_data:
        p_load[int(row[0]) - 1] = row[1]
        q_load[int(row[0]) - 1] = row[2]

    for idx, bus_no in enumerate(bus_indices):
        q_load[int(bus_no) - 1] -= q_caps[idx]

    P = p_load / BASE_KVA
    Q = q_load / BASE_KVA
    R = nt[:, 3] / Z_BASE
    X = nt[:, 4] / Z_BASE

    V = np.ones(n_bus)
    pl_loss = np.zeros(84)

    for _ in range(4):
        P_br = np.zeros(84)
        Q_br = np.zeros(84)
        for i in range(83, -1, -1):
            t_bus = int(nt[i, 2]) - 1
            P_br[i] = P[t_bus] + np.sum(P_br[nt[:, 1] == (t_bus + 1)])
            Q_br[i] = Q[t_bus] + np.sum(Q_br[nt[:, 1] == (t_bus + 1)])

        for i in range(84):
            f_bus = int(nt[i, 1]) - 1
            t_bus = int(nt[i, 2]) - 1
            vs = V[f_bus]

            A = (P_br[i] * R[i] + Q_br[i] * X[i]) - 0.5 * (vs**2)
            B = np.sqrt(np.maximum(0, A**2 - (R[i]**2 + X[i]**2) * (P_br[i]**2 + Q_br[i]**2)))
            V[t_bus] = np.sqrt(np.maximum(0, B - A))
            pl_loss[i] = R[i] * (P_br[i]**2 + Q_br[i]**2) / (V[t_bus]**2)

    return np.sum(pl_loss) * BASE_KVA, V


def run_pso(params, update_callback=None):
    num_particles = params['num_particles']
    max_iter = params['max_iter']
    qcap_max = params['qcap_max']
    dim = len(params['bus_indices'])
    v_max = params['v_max']
    w = params['w']
    c1 = params['c1']
    c2 = params['c2']
    bus_indices = params['bus_indices']

    pos = np.random.uniform(0, qcap_max, (num_particles, dim))
    vel = np.random.uniform(-v_max, v_max, (num_particles, dim))
    pbest_pos = pos.copy()
    pbest_val = np.array([load_flow(p, bus_indices)[0] for p in pos])
    gbest_pos = pbest_pos[np.argmin(pbest_val)]
    gbest_val = np.min(pbest_val)
    convergence = []

    for i in range(max_iter):
        for j in range(num_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            vel[j] = w * vel[j] + c1 * r1 * (pbest_pos[j] - pos[j]) + c2 * r2 * (gbest_pos - pos[j])
            vel[j] = np.clip(vel[j], -v_max, v_max)
            pos[j] = np.clip(pos[j] + vel[j], 0, qcap_max)

            current_loss, _ = load_flow(pos[j], bus_indices)
            if current_loss < pbest_val[j]:
                pbest_val[j] = current_loss
                pbest_pos[j] = pos[j].copy()

        if np.min(pbest_val) < gbest_val:
            gbest_val = np.min(pbest_val)
            gbest_pos = pbest_pos[np.argmin(pbest_val)].copy()

        convergence.append(gbest_val)
        if update_callback:
            update_callback(i + 1, gbest_val)

    initial_loss, base_v = load_flow(np.zeros(dim), bus_indices)
    final_loss, final_v = load_flow(gbest_pos, bus_indices)

    bus_labels = ', '.join(str(bus) for bus in bus_indices)
    result_text = (
        f"Selected buses: {bus_labels}\n"
        f"Optimal Capacitor at Bus {bus_indices[0]}:  {gbest_pos[0]:.2f} kVAR\n"
        f"Optimal Capacitor at Bus {bus_indices[1]}: {gbest_pos[1]:.2f} kVAR\n"
        f"Optimal Capacitor at Bus {bus_indices[2]}:  {gbest_pos[2]:.2f} kVAR\n"
        f"Optimal Capacitor at Bus {bus_indices[3]}: {gbest_pos[3]:.2f} kVAR\n"
        f"\nInitial Power Loss: {initial_loss:.4f} kW\n"
        f"Optimal Power Loss: {final_loss:.4f} kW\n"
        f"Loss Reduction:     {((initial_loss - final_loss) / initial_loss) * 100:.2f}%\n"
    )

    return result_text, convergence, base_v, final_v


def create_plot_frame(parent, convergence, base_v, final_v):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(convergence, 'r-o')
    axes[0].set_title('PSO Convergence')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Loss (kW)')

    axes[1].plot(base_v, 'k--', label='Base Case')
    axes[1].plot(final_v, 'b-', label='With Capacitors')
    axes[1].set_title('Voltage Profile')
    axes[1].set_xlabel('Bus Number')
    axes[1].set_ylabel('Voltage (p.u.)')
    axes[1].legend()
    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=parent)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)


def validate_parameters(particles_entry, iter_entry, vmax_entry, w_entry, c1_entry, c2_entry, qcap_entry, bus_entries):
    try:
        num_particles = int(particles_entry.get())
        max_iter = int(iter_entry.get())
        v_max = float(vmax_entry.get())
        w = float(w_entry.get())
        c1 = float(c1_entry.get())
        c2 = float(c2_entry.get())
        qcap_max = float(qcap_entry.get())
        bus_indices = [int(entry.get()) for entry in bus_entries]
    except ValueError:
        raise ValueError('All parameters must be numeric.')

    if num_particles < 2 or max_iter < 1 or qcap_max <= 0 or v_max <= 0:
        raise ValueError('Particles, iterations, capacitor limit, and v_max must be positive values.')

    for bus in bus_indices:
        if bus < 1 or bus > 85:
            raise ValueError('Bus numbers must be between 1 and 85.')

    return {
        'num_particles': num_particles,
        'max_iter': max_iter,
        'v_max': v_max,
        'w': w,
        'c1': c1,
        'c2': c2,
        'qcap_max': qcap_max,
        'bus_indices': bus_indices,
    }


def run_in_thread(root, status_label, output_box, button, plot_frame, params):
    def update_progress(iteration, loss):
        def task():
            status_label.config(text=f'Running... Iter {iteration} Best Loss = {loss:.4f} kW')
        root.after(0, task)

    def worker():
        def start_task():
            button.config(state='disabled')
            status_label.config(text='Running PSO...')
            output_box.delete('1.0', tk.END)
        root.after(0, start_task)

        try:
            result_text, convergence, base_v, final_v = run_pso(params, update_progress)
        except Exception as exc:
            def error_task():
                status_label.config(text='Error during execution')
                output_box.insert(tk.END, f'Error: {exc}')
                button.config(state='normal')
            root.after(0, error_task)
            return

        def finish_task():
            output_box.insert(tk.END, result_text)
            status_label.config(text='PSO Completed')
            button.config(state='normal')
            for widget in plot_frame.winfo_children():
                widget.destroy()
            create_plot_frame(plot_frame, convergence, base_v, final_v)
        root.after(0, finish_task)

    Thread(target=worker, daemon=True).start()


def build_ui():
    root = tk.Tk()
    root.title('85-Bus PSO Capacitor Placement')
    root.geometry('1000x760')

    header = tk.Label(root, text='85-Bus PSO Capacitor Placement', font=('Segoe UI', 18, 'bold'))
    header.pack(pady=10)

    param_frame = tk.LabelFrame(root, text='Simulation Parameters', padx=12, pady=12)
    param_frame.pack(fill='x', padx=12, pady=(0, 10))

    left_frame = tk.Frame(param_frame)
    left_frame.pack(side='left', fill='both', expand=True, padx=(0, 8))

    right_frame = tk.Frame(param_frame)
    right_frame.pack(side='left', fill='both', expand=True)

    tk.Label(left_frame, text='Particles:').grid(row=0, column=0, sticky='w', pady=4)
    particles_entry = tk.Entry(left_frame, width=12)
    particles_entry.insert(0, str(DEFAULT_NUM_PARTICLES))
    particles_entry.grid(row=0, column=1, sticky='w', pady=4)

    tk.Label(left_frame, text='Iterations:').grid(row=1, column=0, sticky='w', pady=4)
    iter_entry = tk.Entry(left_frame, width=12)
    iter_entry.insert(0, str(DEFAULT_MAX_ITER))
    iter_entry.grid(row=1, column=1, sticky='w', pady=4)

    tk.Label(left_frame, text='Max capacitor kVAR:').grid(row=2, column=0, sticky='w', pady=4)
    qcap_entry = tk.Entry(left_frame, width=12)
    qcap_entry.insert(0, str(DEFAULT_QCAP_MAX))
    qcap_entry.grid(row=2, column=1, sticky='w', pady=4)

    tk.Label(left_frame, text='Velocity limit:').grid(row=3, column=0, sticky='w', pady=4)
    vmax_entry = tk.Entry(left_frame, width=12)
    vmax_entry.insert(0, str(DEFAULT_V_MAX))
    vmax_entry.grid(row=3, column=1, sticky='w', pady=4)

    tk.Label(left_frame, text='Inertia (w):').grid(row=4, column=0, sticky='w', pady=4)
    w_entry = tk.Entry(left_frame, width=12)
    w_entry.insert(0, str(DEFAULT_W))
    w_entry.grid(row=4, column=1, sticky='w', pady=4)

    tk.Label(left_frame, text='C1:').grid(row=5, column=0, sticky='w', pady=4)
    c1_entry = tk.Entry(left_frame, width=12)
    c1_entry.insert(0, str(DEFAULT_C1))
    c1_entry.grid(row=5, column=1, sticky='w', pady=4)

    tk.Label(left_frame, text='C2:').grid(row=6, column=0, sticky='w', pady=4)
    c2_entry = tk.Entry(left_frame, width=12)
    c2_entry.insert(0, str(DEFAULT_C2))
    c2_entry.grid(row=6, column=1, sticky='w', pady=4)

    tk.Label(right_frame, text='Capacitor Bus Locations', font=('Segoe UI', 11, 'bold')).grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 8))
    bus_entries = []
    for idx, bus_no in enumerate(DEFAULT_BUS_INDICES, start=1):
        tk.Label(right_frame, text=f'Bus {idx}:').grid(row=idx, column=0, sticky='w', pady=4)
        entry = tk.Entry(right_frame, width=12)
        entry.insert(0, str(bus_no))
        entry.grid(row=idx, column=1, sticky='w', pady=4)
        bus_entries.append(entry)

    action_frame = tk.Frame(root)
    action_frame.pack(fill='x', padx=12)

    run_button = tk.Button(action_frame, text='Run Optimization', width=18, font=('Segoe UI', 10, 'bold'))
    run_button.pack(side='left', padx=(0, 8))

    status_label = tk.Label(action_frame, text='Ready to run', anchor='w')
    status_label.pack(side='left', fill='x', expand=True)

    def start_optimization():
        try:
            params = validate_parameters(
                particles_entry, iter_entry, vmax_entry, w_entry, c1_entry, c2_entry, qcap_entry, bus_entries
            )
        except ValueError as exc:
            status_label.config(text='Invalid input')
            output_box.delete('1.0', tk.END)
            output_box.insert(tk.END, f'Error: {exc}')
            return

        run_in_thread(root, status_label, output_box, run_button, plot_frame, params)

    run_button.config(command=start_optimization)

    output_label = tk.Label(root, text='Results', font=('Segoe UI', 12, 'bold'))
    output_label.pack(padx=12, pady=(10, 0), anchor='w')

    output_box = ScrolledText(root, height=10, font=('Consolas', 11), wrap='word')
    output_box.pack(fill='x', padx=12, pady=(0, 12))

    plot_frame = tk.LabelFrame(root, text='Plots', relief='groove', bd=2)
    plot_frame.pack(fill='both', expand=True, padx=12, pady=(0, 12))

    return root


if __name__ == '__main__':
    app = build_ui()
    app.mainloop()
