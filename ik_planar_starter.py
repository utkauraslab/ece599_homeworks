import numpy as np
import matplotlib.pyplot as plt

# ================================================
# Task 2: Logarithmic map on R^2 × S^1
# ================================================
def logmap(f, f0):
    """
    Compute residual vector between current pose f and target pose f0.
    Position is Euclidean; orientation uses geodesic error on S^1.
    """
    diff = np.zeros(3)
    diff[:2] = f[:2] - f0[:2]
    # TODO (Task 2): Implement orientation residual using complex exponential map
    # diff[2] =
    return diff

# ================================================
# Task 1: Forward kinematics – end-effector pose
# ================================================
def fkin(x, param):
    """
    Compute end-effector pose [x, y, theta] from joint angles.
    """
    # TODO (Task 1): Use cumulative angles (L @ x) to compute x, y, theta
    # x = sum_i l_i * cos(θ_1 + ... + θ_i)
    # y = sum_i l_i * sin(θ_1 + ... + θ_i)
    # theta = total joint angle (e.g., np.sum(x))

    return np.zeros(3)

# ================================================
# Task 5: Forward kinematics – all joint positions (for visualization)
# ================================================
def fkin0(x, param):
    """
    Compute positions of all joints including the base (origin).
    Returns: 2x(n+1) array [x_row; y_row]
    """
    # TODO (Task 5): Compute joint positions using cumulative joint angles

    return f

# ================================================
# Task 3: Analytical Jacobian (to be derived)
# ================================================
def Jkin_analytical(x, param):
    """
    Compute analytical Jacobian of end-effector pose w.r.t. joint angles.
    Returns: 3x3 Jacobian matrix
    """
    # TODO (Task 3): Derive and implement full Jacobian matrix

    return J

# ================================================
# Task 6: Numerical Jacobian (finite differences)
# ================================================
def Jkin_numerical(x, param, delta=1e-6):
    """
    Estimate the Jacobian numerically using finite differences.
    """

    return J


# ================================================
# Parameters
# ================================================
def param(): return None


param.dt = 1e-2
param.nbData = 50
param.nbVarX = 3
param.nbVarU = 3
param.nbVarF = 3
param.l = [2, 2, 1]

# Target pose (x, y, theta)
target = np.array([3, 1, -np.pi / 2])
x = -np.ones(param.nbVarX) * np.pi / param.nbVarX
x[0] += np.pi


# ================================================
# Task 4: Inverse Kinematics Loop (Analytical Jacobian)
# ================================================
plt.figure()
plt.title("IK with Analytical Jacobian")
plt.scatter(target[0], target[1], color='r', marker='x', s=100)

for t in range(param.nbData):
    fk = fkin(x, param)                # Task 1
    J = Jkin_analytical(x, param)     # Task 3  // can be replace with Jkin_numerical(x, param) for task 6

    # ================================================
    # Task 6: Compare Analytical vs Numerical Jacobians
    # ================================================
    J_ana = Jkin_analytical(x, param)
    J_num = Jkin_numerical(x, param)
    error = np.linalg.norm(J_ana - J_num)

    print(f"Jacobian difference (Frobenius norm): {error:.4e}")

    u = np.linalg.pinv(J) @ logmap(target, fk) * 0.1 / param.dt  # Task 4
    x += u * param.dt

    f_rob = fkin0(x, param)           # Task 5
    plt.plot(f_rob[0, :], f_rob[1, :], color=str(1 - t / param.nbData), linewidth=2)

plt.axis('equal')
plt.axis('off')


# ================================================
# Plot results
# ================================================
plt.show()
