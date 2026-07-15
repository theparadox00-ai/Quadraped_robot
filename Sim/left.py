import mujoco
import mujoco.viewer
import numpy as np
import time

# --- 1. SETUP & INITIALIZATION ---
model = mujoco.MjModel.from_xml_path('/home/m2_sdk/urdf/m2_metal_description/mujoco/scene.xml')
data = mujoco.MjData(model)

# --- 2. GROUND TRUTH POSES ---
SLEEP_POS = np.array([0.0, 1.3, -2.81] * 4)
STAND_POS = np.array([0.0, 0.745, -1.5] * 4)

# --- 3. TIMELINE CONSTANTS ---
WAIT_SLEEP_DURATION = 3.0
STANDUP_DURATION    = 4.0
WAIT_STAND_DURATION = 3.0

for i in range(12):
    data.qpos[7 + i] = SLEEP_POS[i]
data.qpos[2] = 0.1493
mujoco.mj_forward(model, data)

# --- 4. MOTOR CONTROLLER (PD CONTROL) ---
def pd_control(data, target_pos, kp=120.0, kd=6.0):
    for i in range(12):
        q_act  = data.qpos[7 + i]
        dq_act = data.qvel[6 + i]
        tau = kp * (target_pos[i] - q_act) - kd * dq_act
        data.ctrl[i] = np.clip(tau, -22.0, 22.0)

def smoothstep(x):
    return x * x * (3 - 2 * x)

# --- 5. INVERSE KINEMATICS (IK) ENGINE ---
def solve_ik_3d(x, y, z):
    L1 = 1.0
    L2 = 1.0

    hip = np.arctan2(y, -z)
    z_eff = -np.sqrt(y**2 + z**2)
    D_sq = x**2 + z_eff**2

    if D_sq > (L1 + L2)**2:
        scale = (L1 + L2 - 0.0001) / np.sqrt(D_sq)
        x *= scale; z_eff *= scale; D_sq = x**2 + z_eff**2

    cos_calf = np.clip((D_sq - L1**2 - L2**2) / (2 * L1 * L2), -1.0, 1.0)
    calf = -np.arccos(cos_calf)
    thigh = np.arctan2(x, -z_eff) + np.arctan2(L2 * np.sin(-calf), L1 + L2 * np.cos(calf))

    return hip, thigh, calf

# --- 6. FOOT PATH GENERATOR ---
def get_foot_trajectory(phase, step_x, step_y, step_height, z_stand):
    if phase < 0.5:
        s = phase * 2.0
        x = -(step_x / 2.0) * np.cos(s * np.pi)
        y = -(step_y / 2.0) * np.cos(s * np.pi)
        z = z_stand + step_height * np.sin(s * np.pi)
    else:
        s = (phase - 0.5) * 2.0
        x = (step_x / 2.0) * np.cos(s * np.pi)
        y = (step_y / 2.0) * np.cos(s * np.pi)
        z = z_stand
    return x, y, z

# --- 7. TROT GAIT SCHEDULER (STRAFE LEFT) ---
def get_trot_target_ik(t):
    """Synchronizes the 4 legs into a diagonal Trot pattern, strafing sideways."""
    target = STAND_POS.copy()
    phase = (t * 2.0) % 1.0

    # DIRECTION SETTINGS: Strafe Left
    step_x = 0.0
    step_y = 0.55
    step_height = 0.25
    z_stand = -1.46

    # Pair 1: Front-Left (FL) & Rear-Right (RR)
    x1, y1, z1 = get_foot_trajectory(phase, step_x, step_y, step_height, z_stand)
    h1, t1, c1 = solve_ik_3d(x1, y1, z1)
    target[0] = h1; target[1] = t1; target[2] = c1
    target[9] = h1; target[10] = t1; target[11] = c1

    # Pair 2: Front-Right (FR) & Rear-Left (RL)
    x2, y2, z2 = get_foot_trajectory((phase + 0.5) % 1.0, step_x, step_y, step_height, z_stand)
    h2, t2, c2 = solve_ik_3d(x2, y2, z2)
    target[3] = h2; target[4] = t2; target[5] = c2
    target[6] = h2; target[7] = t2; target[8] = c2
    return target

# --- 8. MAIN SIMULATION LOOP ---
viewer = mujoco.viewer.launch_passive(model, data)
start_time = time.time()
start_walk_pos = None

while viewer.is_running():
    t = time.time() - start_time

    if t < WAIT_SLEEP_DURATION:
        target_pos = SLEEP_POS
    elif t < WAIT_SLEEP_DURATION + STANDUP_DURATION:
        alpha = smoothstep((t - WAIT_SLEEP_DURATION) / STANDUP_DURATION)
        target_pos = (1 - alpha) * SLEEP_POS + alpha * STAND_POS
    elif t < WAIT_SLEEP_DURATION + STANDUP_DURATION + WAIT_STAND_DURATION:
        target_pos = STAND_POS
    else:
        if start_walk_pos is None:
            start_walk_pos = data.qpos[0:2].copy()

        if np.abs(data.qpos[1] - start_walk_pos[1]) < 3.0:
            walk_t = t - (WAIT_SLEEP_DURATION + STANDUP_DURATION + WAIT_STAND_DURATION)
            target_pos = get_trot_target_ik(walk_t)
        else:
            target_pos = STAND_POS

    pd_control(data, target_pos)
    mujoco.mj_step(model, data)
    viewer.sync()

viewer.close()