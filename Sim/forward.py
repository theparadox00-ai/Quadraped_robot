import mujoco
import mujoco.viewer
import numpy as np
import time

# --- 1. SETUP & INITIALIZATION ---
# Load the robot's physical model (XML) and state data
model = mujoco.MjModel.from_xml_path('/home/m2_sdk/urdf/m2_metal_description/mujoco/scene.xml')
data = mujoco.MjData(model)

# --- 2. GROUND TRUTH POSES ---
# Observed directly from the M2 Metal robot's joint states
SLEEP_POS = np.array([0.0, 1.3, -2.81] * 4)   # Legs fully folded
STAND_POS = np.array([0.0, 0.745, -1.5] * 4)  # Proper crouched standing pose

# --- 3. TIMELINE CONSTANTS ---
WAIT_SLEEP_DURATION = 3.0  # Seconds to wait before standing
STANDUP_DURATION    = 4.0  # Seconds it takes to smoothly transition to standing
WAIT_STAND_DURATION = 3.0  # Seconds to hold the stand before walking

# Spawn the robot in the sleep pose initially
for i in range(12):
    data.qpos[7 + i] = SLEEP_POS[i]
data.qpos[2] = 0.1493  # Prevents clipping into the floor
mujoco.mj_forward(model, data)  # Initialize physics to prevent viewer crash

# --- 4. MOTOR CONTROLLER (PD CONTROL) ---
def pd_control(data, target_pos, kp=120.0, kd=6.0):
    """
    Acts like virtual springs and dampers for the motors.
    Converts desired joint angles into actual torque (force).
    """
    for i in range(12):
        q_act  = data.qpos[7 + i]  # Current actual angle
        dq_act = data.qvel[6 + i]  # Current actual velocity
        # Torque = (Spring stiffness * Distance to target) - (Damping * Current speed)
        tau = kp * (target_pos[i] - q_act) - kd * dq_act
        # Limit the torque to the real robot's physical max limit (22 Nm)
        data.ctrl[i] = np.clip(tau, -22.0, 22.0)

def smoothstep(x):
    """S-curve math function for smooth stand-up transitions."""
    return x * x * (3 - 2 * x)

# --- 5. INVERSE KINEMATICS (IK) ENGINE ---
def solve_ik_3d(x, y, z):
    """
    Calculates the Hip, Thigh, and Calf angles needed to place the foot at (x, y, z).
    This keeps the robot body perfectly level while the legs move.
    """
    L1 = 1.0  # Length of thigh link
    L2 = 1.0  # Length of calf link

    # 1. Calculate Hip angle to swing leg sideways (Y-axis)
    hip = np.arctan2(y, -z)

    # 2. Calculate the effective 2D length the leg must reach after swinging sideways
    z_eff = -np.sqrt(y**2 + z**2)
    D_sq = x**2 + z_eff**2  # Squared distance from shoulder to target foot pos

    # Safety Check: Prevent math crashes if requested target is beyond leg length
    if D_sq > (L1 + L2)**2:
        scale = (L1 + L2 - 0.0001) / np.sqrt(D_sq)
        x *= scale; z_eff *= scale; D_sq = x**2 + z_eff**2

    # 3. Law of Cosines to find how much the knee (calf) must bend
    cos_calf = np.clip((D_sq - L1**2 - L2**2) / (2 * L1 * L2), -1.0, 1.0)
    calf = -np.arccos(cos_calf)

    # 4. Trigonometry to aim the thigh toward the target, compensating for knee bend
    thigh = np.arctan2(x, -z_eff) + np.arctan2(L2 * np.sin(-calf), L1 + L2 * np.cos(calf))

    return hip, thigh, calf

# --- 6. FOOT PATH GENERATOR ---
def get_foot_trajectory(phase, step_x, step_y, step_height, z_stand):
    """Generates the 3D arc path for the foot during a step."""
    if phase < 0.5:
        # SWING PHASE (Leg in air)
        s = phase * 2.0  # Normalize 0.0-0.5 to 0.0-1.0
        x = -(step_x / 2.0) * np.cos(s * np.pi)  # Smooth cosine acceleration forward
        y = -(step_y / 2.0) * np.cos(s * np.pi)
        z = z_stand + step_height * np.sin(s * np.pi)  # Sine wave arc for lifting foot
    else:
        # STANCE PHASE (Leg on ground pushing)
        s = (phase - 0.5) * 2.0
        x = (step_x / 2.0) * np.cos(s * np.pi)
        y = (step_y / 2.0) * np.cos(s * np.pi)
        z = z_stand  # MAGIC LINE: Z is flat, removing all bouncing!
    return x, y, z

# --- 7. TROT GAIT SCHEDULER ---
def get_trot_target_ik(t):
    """Synchronizes the 4 legs into a diagonal Trot pattern."""
    target = STAND_POS.copy()
    phase = (t * 2.0) % 1.0  # Controls speed (2 steps per second)

    # DIRECTION SETTINGS: Forward
    step_x = -0.55   # Negative pushes body Forward (axis inverted)
    step_y = 0.0     # No sideways movement
    step_height = 0.25
    z_stand = -1.46  # Mathematically matches the 0.745/-1.5 stance

    # Pair 1: Front-Left (FL) & Rear-Right (RR) move together
    x1, y1, z1 = get_foot_trajectory(phase, step_x, step_y, step_height, z_stand)
    h1, t1, c1 = solve_ik_3d(x1, y1, z1)
    target[0] = h1; target[1] = t1; target[2] = c1    # FL Joints
    target[9] = h1; target[10] = t1; target[11] = c1  # RR Joints

    # Pair 2: Front-Right (FR) & Rear-Left (RL) move opposite to Pair 1 (+0.5 phase)
    x2, y2, z2 = get_foot_trajectory((phase + 0.5) % 1.0, step_x, step_y, step_height, z_stand)
    h2, t2, c2 = solve_ik_3d(x2, y2, z2)
    target[3] = h2; target[4] = t2; target[5] = c2    # FR Joints
    target[6] = h2; target[7] = t2; target[8] = c2    # RL Joints
    return target

# --- 8. MAIN SIMULATION LOOP ---
viewer = mujoco.viewer.launch_passive(model, data)
start_time = time.time()
start_walk_pos = None

while viewer.is_running():
    t = time.time() - start_time

    # Step 1: Sleep
    if t < WAIT_SLEEP_DURATION:
        target_pos = SLEEP_POS

    # Step 2: Smooth Standup
    elif t < WAIT_SLEEP_DURATION + STANDUP_DURATION:
        alpha = smoothstep((t - WAIT_SLEEP_DURATION) / STANDUP_DURATION)
        target_pos = (1 - alpha) * SLEEP_POS + alpha * STAND_POS

    # Step 3: Hold Stand
    elif t < WAIT_SLEEP_DURATION + STANDUP_DURATION + WAIT_STAND_DURATION:
        target_pos = STAND_POS

    # Step 4: Walk Forward 3 Meters
    else:
        if start_walk_pos is None:
            start_walk_pos = data.qpos[0:2].copy()  # Record starting X,Y exactly when walking starts

        # Check if we have traveled 3 meters on the X axis
        if np.abs(data.qpos[0] - start_walk_pos[0]) < 3.0:
            walk_t = t - (WAIT_SLEEP_DURATION + STANDUP_DURATION + WAIT_STAND_DURATION)
            target_pos = get_trot_target_ik(walk_t)
        else:
            target_pos = STAND_POS  # Stop and stand once target is reached

    # Apply torques and step simulation
    pd_control(data, target_pos)
    mujoco.mj_step(model, data)
    viewer.sync()

viewer.close()

