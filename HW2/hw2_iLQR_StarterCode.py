# iLQR applied to a planar manipulator for a viapoints task (recursive formulation to find a controller)
# === STUDENT VERSION WITH INLINE HINTS ===
# Instructions:
#   - This script contains commented-out iLQR logic with inline hints for each step.
#   - Fill in the TODO sections to complete the implementation.
#   - Use provided helper functions such as f_reach, fkin, fkin0, Jkin for computations.

# For brevity, this version starts from the iLQR loop.
# Ensure you have already run the parameter initialization and helper function blocks from the full script.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb

# ===============================
# Helper functions
# ===============================

# Logarithmic map for R^2 x S^1 manifold
def logmap(f, f0):
    position_error = f[:2, :] - f0[:2, :]
    orientation_error = np.imag(np.log(np.exp(f0[-1, :] * 1j).conj().T * np.exp(f[-1, :] * 1j).T)).conj()
    error = np.vstack([position_error, orientation_error])
    return error

# Forward kinematics for end-effector (in robot coordinate system)
def fkin(x, param):
    L = np.tril(np.ones([param.nbVarX, param.nbVarX]))
    f = np.vstack([
        param.l @ np.cos(L @ x),
        param.l @ np.sin(L @ x),
        np.mod(np.sum(x, 0) + np.pi, 2 * np.pi) - np.pi
    ])  # f1,f2,f3, where f3 is the orientation (single Euler angle for planar robot)
    return f

# Forward kinematics for end-effector (in robot coordinate system)
def fkin0(x, param):
    L = np.tril(np.ones([param.nbVarX, param.nbVarX]))
    f = np.vstack([
        L @ np.diag(param.l) @ np.cos(L @ x),
        L @ np.diag(param.l) @ np.sin(L @ x)
    ])
    f = np.hstack([np.zeros([2, 1]), f])
    return f

# Jacobian with analytical computation (for single time step)
def Jkin(x, param):
    L = np.tril(np.ones([param.nbVarX, param.nbVarX]))
    J = np.vstack([
        -np.sin(L @ x).T @ np.diag(param.l) @ L,
        np.cos(L @ x).T @ np.diag(param.l) @ L,
        np.ones([1, param.nbVarX])
    ])
    return J

# Error and Jacobian for a viapoints reaching task (in object coordinate system)
def f_reach(x, param):
    f = logmap(fkin(x, param), param.Mu)
    J = np.zeros([param.nbVarF, param.nbVarX, param.nbPoints])
    for t in range(param.nbPoints):
        f[:2, t] = param.A[:, :, t].T @ f[:2, t]  # Object oriented forward kinematics
        Jtmp = Jkin(x[:, t], param)
        Jtmp[:2] = param.A[:, :, t].T @ Jtmp[:2]  # Object centered Jacobian

        J[:, :, t] = Jtmp
    return f, J


def generate_viapoints(n_points=10):
    t = np.linspace(0, 2 * np.pi, n_points)

    # Raw in [-1, 1] range
    x_raw = np.sin(t)
    y_raw = np.sin(t) * np.cos(t)

    # Normalize to [0,1]
    x_norm = (x_raw - x_raw.min()) / (x_raw.max() - x_raw.min())
    y_norm = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

    # Target bounding box
    lower = np.array([1.5, 1.0, -np.pi / 6])
    upper = np.array([3.0, 2.5, -np.pi / 3])

    # Add scaling margin to approach box edges more closely
    margin = 0.1
    x_scaled = lower[0] + (upper[0] - lower[0]) * (margin + (1 - 2 * margin) * x_norm)
    y_scaled = lower[1] + (upper[1] - lower[1]) * (margin + (1 - 2 * margin) * y_norm)

    # Orientation uses y as a proxy
    theta = lower[2] + (upper[2] - lower[2]) * y_norm

    return np.vstack([x_scaled, y_scaled, theta])


# Parameters
def param(): return None  # Lazy way to define an empty class in python


param.dt = 1e-2  # Time step length
param.nbData = 100  # Number of datapoints
param.nbIter = 60   # Maximum number of iterations for iLQR
param.nbVarX = 3  # State space dimension (x1,x2,x3)
param.nbVarU = 3  # Control space dimension (dx1,dx2,dx3)
param.nbVarF = 3  # Objective function dimension (f1,f2,f3, with f3 as orientation)
param.l = [2, 2, 1]  # Robot links lengths
param.sz = [.2, .3]  # Size of objects
param.Q = 1e0  # Tracking weighting term
param.R = 1e-6  # Control weighting term
param.nbPoints = 50  # Number of viapoints
param.Mu = generate_viapoints(param.nbPoints)  # Generate more via-points as a trajectory (e.g., 8-shape path)
param.A = np.zeros([2, 2, param.nbPoints])  # Object orientation matrices

# ===============================
# Main program
# ===============================

# Object rotation matrices
for t in range(param.nbPoints):
    orn_t = param.Mu[-1, t]
    param.A[:, :, t] = np.asarray([
        [np.cos(orn_t), -np.sin(orn_t)],
        [np.sin(orn_t), np.cos(orn_t)]
    ])

# Time occurrence of viapoints
tl = np.linspace(0, param.nbData, param.nbPoints + 1)
tl = np.rint(tl[1:]).astype(np.int64) - 1


# Transfer matrices (for linear system as single integrator)
A = np.eye(param.nbVarX)
B = np.eye(param.nbVarX, param.nbVarU) * param.dt
Su0 = np.vstack([
    np.zeros([param.nbVarX, param.nbVarX * (param.nbData - 1)]),
    np.tril(np.kron(np.ones([param.nbData - 1, param.nbData - 1]), np.eye(param.nbVarX) * param.dt))
])
Sx0 = np.kron(np.ones(param.nbData), np.identity(param.nbVarX)).T

# iLQR (recursive)
# ===============================
du = np.zeros([param.nbVarU, param.nbData - 1])
utmp = np.zeros([param.nbVarU, param.nbData - 1])
xtmp = np.zeros([param.nbVarX, param.nbData])

k = np.zeros([param.nbVarU, param.nbData - 1])
K = np.zeros([param.nbVarU, param.nbVarX, param.nbData - 1])
Luu = np.identity(param.nbVarU) * param.R  # Command cost Hessian is constant

x0 = np.array([3 * np.pi / 4, -np.pi / 2, -np.pi / 4])  # Initial state
uref = np.zeros([param.nbVarU, param.nbData - 1])  # Initial control command
xref = Su0 @ uref.flatten('F') + Sx0 @ x0  # System evolution
xref = xref.reshape([param.nbVarX, param.nbData], order='F')

for i in range(param.nbIter):
    f, J = f_reach(xref[:, tl], param)  # Compute task-space error and Jacobians at via-points

    # === Step 1: Compute gradients ===
    Lu = uref * param.R
    Lx = np.zeros([param.nbVarX, param.nbData])
    Lxx = np.zeros([param.nbVarX, param.nbVarX, param.nbData])
    for t in range(len(tl)):
        # TODO: Compute Lx
        # Lx[:, tl[t]] = ...
        # TODO: Compute Lxx
        # Lxx[:, :, tl[t]] = ...
        pass

    # === Step 2: Backward Pass ===
    Vx = Lx[:, -1]
    Vxx = Lxx[:, :, -1]
    for t in range(param.nbData - 2, -1, -1):
        # TODO: Compute Qx, Qu, Qxx, Qux, Quu
        # TODO: Compute k[:, t], K[:, :, t]
        # TODO: Update Vx and Vxx
        pass

    # === Step 3: Forward Pass with Line Search ===
    alpha = 1
    # TODO: cost0 = ...
    while True:
        xtmp[:, 0] = x0
        for t in range(param.nbData - 1):
            # TODO: Update control with feedforward and feedback terms
            # du[:,t] =
            # utmp[:,t] =
            # xtmp[:,t+1] =
            pass

        # TODO: Recompute cost and check if improvement
        # ftmp, _ = f_reach(...)
        # cost = ...
        # if cost < cost0 or alpha < 1e-3:
        #     Update uref, xref with new trajectory
        #     break
        alpha /= 2

    if np.linalg.norm(alpha * du) < 1e-2:  # Early stop condition
        break

# Continue with visualization once the above is implemented.
# Simulate reproduction
# ===============================
u = np.zeros([param.nbVarU, param.nbData - 1])
x = np.zeros([param.nbVarX, param.nbData])
x[:, 0] = x0
for t in range(param.nbData - 1):
    u[:, t] = uref[:, t] + K[:, :, t] @ (x[:, t] - xref[:, t])
    x[:, t + 1] = A @ x[:, t] + B @ u[:, t]  # System evolution

# ===============================
# Animated Plot with Fixed Size and Limits
# ===============================
fig = plt.figure(figsize=(8, 6))  # Fixed figure size
ax = plt.gca()
# ax.axis('off')
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-1.5, 3)
ax.set_ylim(0, 3.2)

# Compute overall position bounds for padding
f_all = fkin(x, param)

# Plot reference via-points and trajectory
ax.plot(param.Mu[0, :], param.Mu[1, :], 'r--', linewidth=4, label='Via-points')
traj_line, = ax.plot([], [], 'g>-.', label='Tracked trajectory')
# ax.plot(f_all[0, :], f_all[1, :], 'g--', label='Tracked trajectory')

# Animate robot configuration
for t in range(0, param.nbData, 2):
    # Clear only the previous robot arm
    for artist in ax.lines[2:]:  # Remove arm links only (not ref lines)
        artist.remove()

    f_links = fkin0(x[:, t], param)
    ax.plot(f_links[0, :], f_links[1, :], 'ko-', linewidth=4, markersize=10)
    traj_line.set_data(f_all[0, :t + 1], f_all[1, :t + 1])

    plt.pause(0.08)

plt.show()
