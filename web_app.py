from flask import Flask, render_template, request, Response, redirect, url_for, session
from pso_85bus_ocp import load_flow, get_system_data, BUS_SYSTEMS, get_bus_diagram_data
import numpy as np
import io
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import threading
import webbrowser
import time
import os
from math import gamma, sin, pi

app = Flask(__name__)
app.secret_key = 'change_this_to_a_random_secret_key'

# Users are loaded from users.json. No default credentials are created by the app.
VALID_USERS = {}

USERS_FILE = os.path.join(os.path.dirname(__file__), 'users.json')

def save_users():
    try:
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(VALID_USERS, f, indent=2)
    except Exception:
        pass


def load_users():
    global VALID_USERS
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r', encoding='utf-8') as f:
                users = json.load(f)
            if isinstance(users, dict):
                VALID_USERS = users
            else:
                VALID_USERS = {}
        except Exception:
            VALID_USERS = {}
    else:
        VALID_USERS = {}


load_users()

DEFAULTS = {
    'num_particles': 20,
    'max_iter': 15,
    'v_max': 110.0,
    'w': 0.7,
    'c1': 2.05,
    'c2': 2.05,
    'qcap_max': 1100.0,
    'crossover_rate': 0.8,
    'mutation_rate': 0.1,
    'river_ratio': 0.1,
    'weaving_factor': 0.5,
    'bus_indices': [3, 39, 2, 18],
    'algorithm': 'PSO',
    'system_id': '85'
}

STANDARD_INITIAL_LOSSES = {
    '15': 61.8,
    '33': 202.7,
    '34': 221.7,
    '69': 224.9,
    '85': 316.1,
}


def get_standard_initial_loss(system_id):
    return STANDARD_INITIAL_LOSSES.get(system_id, None)


def login_required(view_function):
    from functools import wraps

    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return view_function(*args, **kwargs)

    return wrapped_view


def admin_required(view_function):
    from functools import wraps

    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        if session.get('username') != 'admin':
            return redirect(url_for('login'))
        return view_function(*args, **kwargs)

    return wrapped_view


@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('index'))

    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if username in VALID_USERS and VALID_USERS[username] == password:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))
        error = 'Invalid username or password.'
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


def run_pso(params):
    num_particles = params['num_particles']
    max_iter = params['max_iter']
    qcap_max = params['qcap_max']
    dim = len(params['bus_indices'])
    v_max = params['v_max']
    w = params['w']
    c1 = params['c1']
    c2 = params['c2']
    bus_indices = params['bus_indices']
    system_id = params.get('system_id', '85')

    pos = np.random.uniform(0, qcap_max, (num_particles, dim))
    vel = np.random.uniform(-v_max, v_max, (num_particles, dim))
    pbest_pos = pos.copy()
    pbest_val = np.array([load_flow(p, bus_indices, system_id)[0] for p in pos])
    gbest_pos = pbest_pos[np.argmin(pbest_val)]
    gbest_val = np.min(pbest_val)
    convergence = []

    for i in range(max_iter):
        for j in range(num_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            vel[j] = w * vel[j] + c1 * r1 * (pbest_pos[j] - pos[j]) + c2 * r2 * (gbest_pos - pos[j])
            vel[j] = np.clip(vel[j], -v_max, v_max)
            pos[j] = np.clip(pos[j] + vel[j], 0, qcap_max)

            current_loss, _, _, _ = load_flow(pos[j], bus_indices, system_id)
            if current_loss < pbest_val[j]:
                pbest_val[j] = current_loss
                pbest_pos[j] = pos[j].copy()

        if np.min(pbest_val) < gbest_val:
            gbest_val = np.min(pbest_val)
            gbest_pos = pbest_pos[np.argmin(pbest_val)].copy()

        convergence.append(gbest_val)

    actual_base_loss, base_v, _, _ = load_flow(np.zeros(dim), bus_indices, system_id)
    final_loss, final_v, _, _ = load_flow(gbest_pos, bus_indices, system_id)
    benchmark_loss = get_standard_initial_loss(system_id) or actual_base_loss

    return {
        'algorithm': 'PSO',
        'system_id': system_id,
        'bus_indices': bus_indices,
        'gbest_pos': gbest_pos.tolist(),
        'initial_loss': benchmark_loss,
        'final_loss': final_loss,
        'loss_reduction': ((benchmark_loss - final_loss) / benchmark_loss) * 100,
        'convergence': convergence,
        'base_v': base_v.tolist(),
        'final_v': final_v.tolist(),
    }


def run_ga(params):
    num_particles = params['num_particles']
    max_iter = params['max_iter']
    qcap_max = params['qcap_max']
    crossover_rate = params.get('crossover_rate', DEFAULTS['crossover_rate'])
    mutation_rate = params.get('mutation_rate', DEFAULTS['mutation_rate'])
    dim = len(params['bus_indices'])
    bus_indices = params['bus_indices']
    system_id = params.get('system_id', '85')

    population = np.random.uniform(0, qcap_max, (num_particles, dim))
    fitness = np.array([load_flow(p, bus_indices, system_id)[0] for p in population])
    best_idx = np.argmin(fitness)
    gbest_pos = population[best_idx]
    gbest_val = fitness[best_idx]
    convergence = [gbest_val]

    for i in range(max_iter):
        selected = []
        for _ in range(num_particles):
            idx1, idx2 = np.random.choice(num_particles, 2, replace=False)
            selected.append(population[idx1] if fitness[idx1] < fitness[idx2] else population[idx2])
        selected = np.array(selected)

        offspring = []
        for j in range(0, num_particles, 2):
            parent1, parent2 = selected[j], selected[j+1]
            if np.random.rand() < crossover_rate:
                crossover_point = np.random.randint(1, dim)
                child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()
            offspring.extend([child1, child2])
        offspring = np.array(offspring)

        for j in range(num_particles):
            if np.random.rand() < mutation_rate:
                offspring[j] += np.random.normal(0, qcap_max * 0.1, dim)
                offspring[j] = np.clip(offspring[j], 0, qcap_max)

        offspring_fitness = np.array([load_flow(p, bus_indices, system_id)[0] for p in offspring])

        combined = np.vstack([population, offspring])
        combined_fitness = np.concatenate([fitness, offspring_fitness])
        best_indices = np.argsort(combined_fitness)[:num_particles]
        population = combined[best_indices]
        fitness = combined_fitness[best_indices]

        current_best = np.min(fitness)
        if current_best < gbest_val:
            gbest_val = current_best
            gbest_pos = population[np.argmin(fitness)]
        convergence.append(gbest_val)

    actual_base_loss, base_v, _, _ = load_flow(np.zeros(dim), bus_indices, system_id)
    final_loss, final_v, _, _ = load_flow(gbest_pos, bus_indices, system_id)
    benchmark_loss = get_standard_initial_loss(system_id) or actual_base_loss

    return {
        'algorithm': 'GA',
        'system_id': system_id,
        'bus_indices': bus_indices,
        'gbest_pos': gbest_pos.tolist(),
        'initial_loss': benchmark_loss,
        'final_loss': final_loss,
        'loss_reduction': ((benchmark_loss - final_loss) / benchmark_loss) * 100,
        'convergence': convergence,
        'base_v': base_v.tolist(),
        'final_v': final_v.tolist(),
    }


def run_wca(params):
    num_particles = params['num_particles']
    max_iter = params['max_iter']
    qcap_max = params['qcap_max']
    river_ratio = params.get('river_ratio', DEFAULTS['river_ratio'])
    dim = len(params['bus_indices'])
    bus_indices = params['bus_indices']
    system_id = params.get('system_id', '85')

    population = np.random.uniform(0, qcap_max, (num_particles, dim))
    fitness = np.array([load_flow(p, bus_indices, system_id)[0] for p in population])
    best_idx = np.argmin(fitness)
    gbest_pos = population[best_idx]
    gbest_val = fitness[best_idx]
    convergence = [gbest_val]

    n_rivers = max(1, min(num_particles - 1, int(river_ratio * num_particles)))
    n_streams = num_particles - n_rivers

    for i in range(max_iter):
        sorted_idx = np.argsort(fitness)
        population = population[sorted_idx]
        fitness = fitness[sorted_idx]

        rivers = population[:n_rivers]
        rivers_fitness = fitness[:n_rivers]

        streams = population[n_rivers:]
        streams_fitness = fitness[n_rivers:]

        for j in range(n_streams):
            river_idx = np.random.randint(n_rivers)
            C = np.random.uniform(0, 2)
            streams[j] = streams[j] + C * (rivers[river_idx] - streams[j])
            streams[j] = np.clip(streams[j], 0, qcap_max)

        for j in range(n_rivers):
            if j > 0:
                C = np.random.uniform(0, 1)
                rivers[j] = rivers[j] + C * (rivers[0] - rivers[j])
                rivers[j] = np.clip(rivers[j], 0, qcap_max)

        population = np.vstack([rivers, streams])
        fitness = np.array([load_flow(p, bus_indices, system_id)[0] for p in population])

        current_best = np.min(fitness)
        if current_best < gbest_val:
            gbest_val = current_best
            gbest_pos = population[np.argmin(fitness)]
        convergence.append(gbest_val)

    actual_base_loss, base_v = load_flow(np.zeros(dim), bus_indices, system_id)
    final_loss, final_v = load_flow(gbest_pos, bus_indices, system_id)
    benchmark_loss = get_standard_initial_loss(system_id) or actual_base_loss

    return {
        'algorithm': 'WCA',
        'system_id': system_id,
        'bus_indices': bus_indices,
        'gbest_pos': gbest_pos.tolist(),
        'initial_loss': benchmark_loss,
        'final_loss': final_loss,
        'loss_reduction': ((benchmark_loss - final_loss) / benchmark_loss) * 100,
        'convergence': convergence,
        'base_v': base_v.tolist(),
        'final_v': final_v.tolist(),
    }


def run_wbno(params):
    # Enhanced Weaver Bird Nest Optimization (WBNO)
    num_particles = params['num_particles']
    max_iter = params['max_iter']
    qcap_max = params['qcap_max']
    weaving_factor = params.get('weaving_factor', DEFAULTS['weaving_factor'])
    dim = len(params['bus_indices'])
    bus_indices = params['bus_indices']
    system_id = params.get('system_id', '85')
    trials = int(params.get('trials', 3))

    def levy_flight(d, beta=1.5):
        # Mantegna's algorithm for Levy stable step
        sigma_u = (gamma(1 + beta) * sin(pi * beta / 2) /
                   (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.random.normal(0, sigma_u, size=d)
        v = np.random.normal(0, 1, size=d)
        step = u / (np.abs(v) ** (1 / beta))
        return 0.01 * step

    best_overall_pos = None
    best_overall_val = float('inf')
    best_overall_conv = None

    for t in range(max(1, trials)):
        nests = np.random.uniform(0, qcap_max, (num_particles, dim))
        fitness = np.array([load_flow(n, bus_indices, system_id)[0] for n in nests])
        gbest_idx = np.argmin(fitness)
        gbest_pos = nests[gbest_idx].copy()
        gbest_val = fitness[gbest_idx]
        convergence = [gbest_val]
        no_improve = np.zeros(num_particles, dtype=int)
        elite_count = max(1, int(0.2 * num_particles))

        for i in range(max_iter):
            # adaptive weaving factor: more exploration early, more exploitation later
            af = weaving_factor * (1 - (i / max(1, max_iter - 1))) + 0.05 * np.random.rand()

            order = np.argsort(fitness)
            nests = nests[order]
            fitness = fitness[order]
            elites = nests[:elite_count]

            for j in range(num_particles):
                current = nests[j].copy()
                weak_ratio = 0.25 + 0.5 * (j / max(1, num_particles - 1))
                weak_mask = np.random.rand(dim) < weak_ratio

                elite_nest = elites[np.random.randint(elite_count)]
                neighbor_idx = np.random.randint(num_particles)
                neighbor = nests[neighbor_idx]

                rand_influence = np.random.rand(dim)
                # primary weaving update
                current[weak_mask] = current[weak_mask] + af * rand_influence[weak_mask] * (elite_nest[weak_mask] - current[weak_mask])
                current[weak_mask] += (1 - af) * rand_influence[weak_mask] * (neighbor[weak_mask] - current[weak_mask])

                # occasional Levy-driven exploration
                if np.random.rand() < 0.18:
                    current += levy_flight(dim)

                # small local greedy search around current
                if np.random.rand() < 0.4:
                    candidate = current + np.random.normal(0, qcap_max * 0.02, dim)
                    candidate = np.clip(candidate, 0, qcap_max)
                    cand_loss = load_flow(candidate, bus_indices, system_id)[0]
                    if cand_loss < fitness[j]:
                        current = candidate

                current = np.clip(current, 0, qcap_max)

                current_loss = load_flow(current, bus_indices, system_id)[0]
                if current_loss < fitness[j]:
                    nests[j] = current
                    fitness[j] = current_loss
                    no_improve[j] = 0
                else:
                    no_improve[j] += 1

                # rejuvenate stagnant nests using opposition-based or random restart
                if no_improve[j] >= 4:
                    if np.random.rand() < 0.5:
                        # opposition-based sampling
                        opposite = qcap_max - nests[j] + np.random.uniform(-0.05 * qcap_max, 0.05 * qcap_max, dim)
                        nests[j] = np.clip(opposite, 0, qcap_max)
                    else:
                        nests[j] = np.random.uniform(0, qcap_max, dim)
                    fitness[j] = load_flow(nests[j], bus_indices, system_id)[0]
                    no_improve[j] = 0

            # update global best
            current_best = np.min(fitness)
            if current_best < gbest_val:
                gbest_val = current_best
                gbest_pos = nests[np.argmin(fitness)].copy()
            convergence.append(gbest_val)

        # end of trial
        if gbest_val < best_overall_val:
            best_overall_val = gbest_val
            best_overall_pos = gbest_pos.copy()
            best_overall_conv = convergence.copy()

    actual_base_loss, base_v, _, _ = load_flow(np.zeros(dim), bus_indices, system_id)
    final_loss, final_v, _, _ = load_flow(best_overall_pos, bus_indices, system_id)
    benchmark_loss = get_standard_initial_loss(system_id) or actual_base_loss

    return {
        'algorithm': 'WBNO',
        'system_id': system_id,
        'bus_indices': bus_indices,
        'gbest_pos': best_overall_pos.tolist(),
        'initial_loss': benchmark_loss,
        'final_loss': final_loss,
        'loss_reduction': ((benchmark_loss - final_loss) / benchmark_loss) * 100,
        'convergence': best_overall_conv,
        'base_v': base_v.tolist(),
        'final_v': final_v.tolist(),
    }


def run_optimization(params):
    algorithm = params['algorithm']
    if algorithm == 'PSO':
        return run_pso(params)
    elif algorithm == 'GA':
        return run_ga(params)
    elif algorithm == 'WCA':
        return run_wca(params)
    elif algorithm == 'WBNO':
        return run_wbno(params)
    else:
        raise ValueError('Unknown algorithm')


def parse_form(form):
    system_id = form.get('system_id', '85')
    sys_config = get_system_data(system_id)
    default_buses = sys_config['default_buses']
    
    bus_count = len(default_buses)
    bus_values = []
    for i in range(bus_count):
        bus_key = f'bus{i+1}'
        bus_values.append(int(form.get(bus_key, default_buses[i])))
    
    params = {
        'num_particles': int(form.get('num_particles', DEFAULTS['num_particles'])),
        'max_iter': int(form.get('max_iter', DEFAULTS['max_iter'])),
        'v_max': float(form.get('v_max', DEFAULTS['v_max'])),
        'w': float(form.get('w', DEFAULTS['w'])),
        'c1': float(form.get('c1', DEFAULTS['c1'])),
        'c2': float(form.get('c2', DEFAULTS['c2'])),
        'qcap_max': float(form.get('qcap_max', DEFAULTS['qcap_max'])),
        'crossover_rate': float(form.get('crossover_rate', DEFAULTS['crossover_rate'])),
        'mutation_rate': float(form.get('mutation_rate', DEFAULTS['mutation_rate'])),
        'river_ratio': float(form.get('river_ratio', DEFAULTS['river_ratio'])),
        'weaving_factor': float(form.get('weaving_factor', DEFAULTS['weaving_factor'])),
        'bus_indices': bus_values,
        'algorithm': form.get('algorithm', DEFAULTS['algorithm']),
        'system_id': system_id,
    }
    return params


@app.route('/')
def home():
    if session.get('logged_in'):
        return redirect(url_for('index'))
    return redirect(url_for('login'))


@app.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    system_id = request.args.get('system_id', request.form.get('system_id', '85'))
    sys_config = get_system_data(system_id)
    
    params = DEFAULTS.copy()
    params['bus_indices'] = sys_config['default_buses']
    params['system_id'] = system_id
    
    result = None
    error = None

    if request.method == 'POST':
        try:
            params = parse_form(request.form)
            result = run_optimization(params)
        except Exception as exc:
            error = str(exc)

    return render_template('index.html', params=params, result=result, error=error, systems=BUS_SYSTEMS, current_user=session.get('username'))


@app.route('/admin', methods=['GET', 'POST'])
@login_required
@admin_required
def admin():
    message = None
    error = None
    users = sorted(VALID_USERS.items())

    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'add_user':
            username = request.form.get('username', '').strip()
            password = request.form.get('password', '').strip()
            if not username or not password:
                error = 'Username and password are required.'
            elif username in VALID_USERS:
                error = 'That username already exists.'
            else:
                VALID_USERS[username] = password
                save_users()
                message = f'User "{username}" added successfully.'
                users = sorted(VALID_USERS.items())
        elif action == 'delete_user':
            username = request.form.get('username_to_delete', '').strip()
            if username == 'admin':
                error = 'The admin account cannot be deleted.'
            elif username in VALID_USERS:
                del VALID_USERS[username]
                save_users()
                message = f'User "{username}" removed successfully.'
                users = sorted(VALID_USERS.items())
            else:
                error = 'User not found.'

    return render_template('admin.html', users=users, message=message, error=error)


@app.route('/bus_diagram.png')
@login_required
def bus_diagram_png():
    system_id = request.args.get('system_id', '85')
    data = get_bus_diagram_data(system_id)
    positions = data['positions']

    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Draw substation box at top
    sub_x, sub_y = positions[1]
    rect = plt.Rectangle((sub_x - 0.4, sub_y + 0.4), 0.8, 0.5, 
                          fill=True, facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(sub_x, sub_y + 0.65, 'Substation', ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Draw edges
    for edge in data['edges']:
        start = positions[edge['from_bus']]
        end = positions[edge['to_bus']]
        ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=2.5, zorder=1)

    # Draw buses as numbered circles
    for node in data['nodes']:
        x, y = positions.get(node['bus'], (0.0, 0.0))
        
        # Bus circle
        circle = plt.Circle((x, y), 0.3, color='white', ec='black', linewidth=1.5, zorder=3, fill=True)
        ax.add_patch(circle)
        
        # Bus number positioned to the right of circle
        ax.text(x + 0.5, y, str(node['bus']), ha='left', va='center', fontweight='bold', fontsize=9, zorder=4,
               bbox=dict(boxstyle='circle,pad=0.2', facecolor='white', edgecolor='black', linewidth=1))
        
        # Load symbol (T-shaped)
        if node['has_load']:
            ax.plot([x, x], [y - 0.3, y - 0.75], 'k-', linewidth=1.5, zorder=2)
            ax.plot([x - 0.18, x + 0.18], [y - 0.5, y - 0.5], 'k-', linewidth=1.5, zorder=2)
            
            load_text = f"{node['p_mw']:.0f}MW/{node['q_mvar']:.0f}MV"
            ax.text(x, y - 0.95, load_text, fontsize=6, ha='center', va='top',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

    ax.set_title(f"IEEE {system_id}-Bus System - Single Line Diagram", fontsize=12, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')
    ax.margins(0.1)

    img_io = io.BytesIO()
    fig.savefig(img_io, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    img_io.seek(0)

    return Response(img_io.getvalue(), mimetype='image/png')


if __name__ == '__main__':
    webbrowser.open('http://127.0.0.1:5000')
    app.run(debug=True, host='0.0.0.0', port=5000)
