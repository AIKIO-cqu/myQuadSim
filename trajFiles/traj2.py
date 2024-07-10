import numpy as np
from numpy import pi
from numpy.linalg import norm
from utils.waypoints import *


class Trajectory:

    def __init__(self, quad):
        self.ctrlType = 'xyz_pos'
        self.xyzType = 5
        self.yawType = 3
        self.averVel = 1

        # 创建航点
        t_wps, wps, y_wps, v_wp = makeWaypoints()
        self.t_wps = t_wps
        self.wps = wps
        self.y_wps = y_wps
        self.v_wp = v_wp

        # 轨迹完成状态标志，用于判断是否到达轨迹终点
        self.end_reached = 0

        # 根据控制类型，为位置控制（"xyz_pos"）初始化轨迹参数

        self.T_segment = np.diff(self.t_wps)  # 计算航点之间的时间差

        # 根据平均速度调整时间段
        distance_segment = self.wps[1:] - self.wps[:-1]
        self.T_segment = np.sqrt(
            distance_segment[:, 0] ** 2 + distance_segment[:, 1] ** 2 + distance_segment[:, 2] ** 2) / self.v_wp
        self.t_wps = np.zeros(len(self.T_segment) + 1)
        self.t_wps[1:] = np.cumsum(self.T_segment)

        # 根据xyzType的值，选择不同的轨迹生成策略，如最小速度、加速度、急动度或快停策略
        # Looking to minimize which derivative order (eg: Minimum velocity -> first order)
        self.deriv_order = int(self.xyzType - 2)

        # Calculate coefficients
        self.coeff_x = minSomethingTraj(self.wps[:, 0], self.T_segment, self.deriv_order)
        self.coeff_y = minSomethingTraj(self.wps[:, 1], self.T_segment, self.deriv_order)
        self.coeff_z = minSomethingTraj(self.wps[:, 2], self.T_segment, self.deriv_order)

        # Get initial heading
        self.current_heading = quad.psi  # 初始偏航方向current_heading

        # Initialize trajectory setpoint  初始化期望的位置、速度、加速度、推力、欧拉角、角速度和偏航速率
        self.desPos = np.zeros(3)  # Desired position (x, y, z)
        self.desVel = np.zeros(3)  # Desired velocity (xdot, ydot, zdot)
        self.desAcc = np.zeros(3)  # Desired acceleration (xdotdot, ydotdot, zdotdot)
        self.desThr = np.zeros(3)  # Desired thrust in N-E-D directions (or E-N-U, if selected)
        self.desEul = np.zeros(3)  # Desired orientation in the world frame (phi, theta, psi)
        self.desPQR = np.zeros(3)  # Desired angular velocity in the body frame (p, q, r)
        self.desYawRate = 0.  # Desired yaw speed

        # 创建一个包含所有期望轨迹状态的向量sDes
        self.sDes = np.hstack((self.desPos,
                               self.desVel,
                               self.desAcc,
                               self.desThr,
                               self.desEul,
                               self.desPQR,
                               self.desYawRate)).astype(float)
        print('轨迹初始化成功')

    def desiredState(self, t, Ts, quad):
        """根据当前时间t和时间步长Ts计算四旋翼飞行器的期望状态。
        这个方法是轨迹规划的核心，它决定了飞行器在每个时间步的目标位置、速度、加速度等"""

        self.desPos = np.zeros(3)  # Desired position (x, y, z)
        self.desVel = np.zeros(3)  # Desired velocity (xdot, ydot, zdot)
        self.desAcc = np.zeros(3)  # Desired acceleration (xdotdot, ydotdot, zdotdot)
        self.desThr = np.zeros(3)  # Desired thrust in N-E-D directions (or E-N-U, if selected)
        self.desEul = np.zeros(3)  # Desired orientation in the world frame (phi, theta, psi)
        self.desPQR = np.zeros(3)  # Desired angular velocity in the body frame (p, q, r)
        self.desYawRate = 0.  # Desired yaw speed

        def pos_waypoint_min():
            """
            生成最小速度、加速度、急动或快停轨迹
            The function takes known number of waypoints and time, then generates a
            minimum velocity, acceleration, jerk or snap trajectory which goes through each waypoint. 
            The output is the desired state associated with the next waypoint for the time t.
            """
            if not (len(self.t_wps) == self.wps.shape[0]):
                raise Exception("Time array and waypoint array not the same size.")

            nb_coeff = self.deriv_order * 2

            # Hover at t=0
            if t == 0:
                self.t_idx = 0
                self.desPos = self.wps[0, :]
            # Stay hover at the last waypoint position
            elif (t >= self.t_wps[-1]):
                self.t_idx = -1
                self.desPos = self.wps[-1, :]
            else:
                self.t_idx = np.where(t <= self.t_wps)[0][0] - 1

                # Scaled time (between 0 and duration of segment)
                scale = (t - self.t_wps[self.t_idx])

                # Which coefficients to use
                start = nb_coeff * self.t_idx
                end = nb_coeff * (self.t_idx + 1)

                # Set desired position, velocity and acceleration
                t0 = get_poly_cc(nb_coeff, 0, scale)
                self.desPos = np.array(
                    [self.coeff_x[start:end].dot(t0), self.coeff_y[start:end].dot(t0), self.coeff_z[start:end].dot(t0)])

                t1 = get_poly_cc(nb_coeff, 1, scale)
                self.desVel = np.array(
                    [self.coeff_x[start:end].dot(t1), self.coeff_y[start:end].dot(t1), self.coeff_z[start:end].dot(t1)])

                t2 = get_poly_cc(nb_coeff, 2, scale)
                self.desAcc = np.array(
                    [self.coeff_x[start:end].dot(t2), self.coeff_y[start:end].dot(t2), self.coeff_z[start:end].dot(t2)])

        def yaw_follow():
            """使飞行器的偏航与其期望速度方向一致"""
            if (t == 0) or (t >= self.t_wps[-1]):
                self.desEul[2] = self.y_wps[self.t_idx]
            else:
                # Calculate desired Yaw
                self.desEul[2] = np.arctan2(self.desVel[1], self.desVel[0])

            # Dirty hack, detect when desEul[2] switches from -pi to pi (or vice-versa) and switch manualy current_heading 
            if (np.sign(self.desEul[2]) - np.sign(self.current_heading) and abs(
                    self.desEul[2] - self.current_heading) >= 2 * pi - 0.1):
                self.current_heading = self.current_heading + np.sign(self.desEul[2]) * 2 * pi

            # Angle between current vector with the next heading vector
            delta_psi = self.desEul[2] - self.current_heading

            # Set Yaw rate
            self.desYawRate = delta_psi / Ts

            # Prepare next iteration
            self.current_heading = self.desEul[2]

        pos_waypoint_min()
        yaw_follow()

        self.sDes = np.hstack((self.desPos, self.desVel, self.desAcc, self.desThr, self.desEul, self.desPQR,
                               self.desYawRate)).astype(float)

        return self.sDes


def get_poly_cc(n, k, t):
    """ This is a helper function to get the coeffitient of coefficient for n-th
        order polynomial with k-th derivative at time t.
    """
    assert (n > 0 and k >= 0), "order and derivative must be positive."

    cc = np.ones(n)
    D = np.linspace(n - 1, 0, n)

    for i in range(n):
        for j in range(k):
            cc[i] = cc[i] * D[i]
            D[i] = D[i] - 1
            if D[i] == -1:
                D[i] = 0

    for i, c in enumerate(cc):
        cc[i] = c * np.power(t, D[i])

    return cc


def minSomethingTraj(waypoints, times, order):
    """ This function takes a list of desired waypoint i.e. [x0, x1, x2...xN] and
    time, returns a [M*N,1] coeffitients matrix for the N+1 waypoints (N segments), 
    where M is the number of coefficients per segment and is equal to (order)*2. If one 
    desires to create a minimum velocity, order = 1. Minimum snap would be order = 4. 

    1.The Problem
    Generate a full trajectory across N+1 waypoint is made of N polynomial line segment.
    Each segment is defined as a (2*order-1)-th order polynomial defined as follow:
    Minimum velocity:     Pi = ai_0 + ai1*t
    Minimum acceleration: Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3
    Minimum jerk:         Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3 + ai4*t^4 + ai5*t^5
    Minimum snap:         Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3 + ai4*t^4 + ai5*t^5 + ai6*t^6 + ai7*t^7

    Each polynomial has M unknown coefficients, thus we will have M*N unknown to
    solve in total, so we need to come up with M*N constraints.

    2.The constraints
    In general, the constraints is a set of condition which define the initial
    and final state, continuity between each piecewise function. This includes
    specifying continuity in higher derivatives of the trajectory at the
    intermediate waypoints.

    3.Matrix Design
    Since we have M*N unknown coefficients to solve, and if we are given M*N
    equations(constraints), then the problem becomes solving a linear equation.

    A * Coeff = B

    Let's look at B matrix first, B matrix is simple because it is just some constants
    on the right hand side of the equation. There are M*N constraints,
    so B matrix will be [M*N, 1].

    Coeff is the final output matrix consists of M*N elements. 
    Since B matrix is only one column, Coeff matrix must be [M*N, 1].

    A matrix is tricky, we then can think of A matrix as a coeffient-coeffient matrix.
    We are no longer looking at a particular polynomial Pi, but rather P1, P2...PN
    as a whole. Since now our Coeff matrix is [M*N, 1], and B is [M*N, 1], thus
    A matrix must have the form [M*N, M*N].

    A = [A10 A11 ... A1M A20 A21 ... A2M ... AN0 AN1 ... ANM
        ...
        ]

    Each element in a row represents the coefficient of coeffient aij under
    a certain constraint, where aij is the jth coeffient of Pi with i = 1...N, j = 0...(M-1).
    """

    n = len(waypoints) - 1
    nb_coeff = order * 2

    # initialize A, and B matrix
    A = np.zeros([nb_coeff * n, nb_coeff * n])
    B = np.zeros(nb_coeff * n)

    # populate B matrix.
    for i in range(n):
        B[i] = waypoints[i]
        B[i + n] = waypoints[i + 1]

    # Constraint 1 - Starting position for every segment
    for i in range(n):
        A[i][nb_coeff * i:nb_coeff * (i + 1)] = get_poly_cc(nb_coeff, 0, 0)

    # Constraint 2 - Ending position for every segment
    for i in range(n):
        A[i + n][nb_coeff * i:nb_coeff * (i + 1)] = get_poly_cc(nb_coeff, 0, times[i])

    # Constraint 3 - Starting position derivatives (up to order) are null
    for k in range(1, order):
        A[2 * n + k - 1][:nb_coeff] = get_poly_cc(nb_coeff, k, 0)

    # Constraint 4 - Ending position derivatives (up to order) are null
    for k in range(1, order):
        A[2 * n + (order - 1) + k - 1][-nb_coeff:] = get_poly_cc(nb_coeff, k, times[i])

    # Constraint 5 - All derivatives are continuous at each waypint transition
    for i in range(n - 1):
        for k in range(1, nb_coeff - 1):
            A[2 * n + 2 * (order - 1) + i * 2 * (order - 1) + k - 1][
            i * nb_coeff: (i * nb_coeff + nb_coeff * 2)] = np.concatenate(
                (get_poly_cc(nb_coeff, k, times[i]), -get_poly_cc(nb_coeff, k, 0)))

    # solve for the coefficients
    Coeff = np.linalg.solve(A, B)
    return Coeff


# Minimum acceleration/jerk/snap Trajectory, but with null velocity, accel and jerk at each waypoint
def minSomethingTraj_stop(waypoints, times, order):
    """ This function takes a list of desired waypoint i.e. [x0, x1, x2...xN] and
    time, returns a [M*N,1] coeffitients matrix for the N+1 waypoints (N segments), 
    where M is the number of coefficients per segment and is equal to (order)*2. If one 
    desires to create a minimum acceleration, order = 2. Minimum snap would be order = 4. 

    1.The Problem
    Generate a full trajectory across N+1 waypoint is made of N polynomial line segment.
    Each segment is defined as a (2*order-1)-th order polynomial defined as follow:
    Minimum velocity:     Pi = ai_0 + ai1*t
    Minimum acceleration: Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3
    Minimum jerk:         Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3 + ai4*t^4 + ai5*t^5
    Minimum snap:         Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3 + ai4*t^4 + ai5*t^5 + ai6*t^6 + ai7*t^7

    Each polynomial has M unknown coefficients, thus we will have M*N unknown to
    solve in total, so we need to come up with M*N constraints.

    Unlike the function minSomethingTraj, where continuous equations for velocity, jerk and snap are generated, 
    this function generates trajectories with null velocities, accelerations and jerks at each waypoints. 
    This will make the drone stop for an instant at each waypoint.
    """

    n = len(waypoints) - 1
    nb_coeff = order * 2

    # initialize A, and B matrix
    A = np.zeros([nb_coeff * n, nb_coeff * n])
    B = np.zeros(nb_coeff * n)

    # populate B matrix.
    for i in range(n):
        B[i] = waypoints[i]
        B[i + n] = waypoints[i + 1]

    # Constraint 1 - Starting position for every segment
    for i in range(n):
        A[i][nb_coeff * i:nb_coeff * (i + 1)] = get_poly_cc(nb_coeff, 0, 0)

    # Constraint 2 - Ending position for every segment
    for i in range(n):
        A[i + n][nb_coeff * i:nb_coeff * (i + 1)] = get_poly_cc(nb_coeff, 0, times[i])

    # Constraint 3 - Starting position derivatives (up to order) for each segment are null
    for i in range(n):
        for k in range(1, order):
            A[2 * n + k - 1 + i * (order - 1)][nb_coeff * i:nb_coeff * (i + 1)] = get_poly_cc(nb_coeff, k, 0)

    # Constraint 4 - Ending position derivatives (up to order) for each segment are null
    for i in range(n):
        for k in range(1, order):
            A[2 * n + (order - 1) * n + k - 1 + i * (order - 1)][nb_coeff * i:nb_coeff * (i + 1)] = get_poly_cc(
                nb_coeff, k, times[i])

    # solve for the coefficients
    Coeff = np.linalg.solve(A, B)
    return Coeff


# Minimum acceleration/jerk/snap Trajectory, but with null velocity only at each waypoint
def minSomethingTraj_faststop(waypoints, times, order):
    """ This function takes a list of desired waypoint i.e. [x0, x1, x2...xN] and
    time, returns a [M*N,1] coeffitients matrix for the N+1 waypoints (N segments), 
    where M is the number of coefficients per segment and is equal to (order)*2. If one 
    desires to create a minimum acceleration, order = 2. Minimum snap would be order = 4. 

    1.The Problem
    Generate a full trajectory across N+1 waypoint is made of N polynomial line segment.
    Each segment is defined as a (2*order-1)-th order polynomial defined as follow:
    Minimum velocity:     Pi = ai_0 + ai1*t
    Minimum acceleration: Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3
    Minimum jerk:         Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3 + ai4*t^4 + ai5*t^5
    Minimum snap:         Pi = ai_0 + ai1*t + ai2*t^2 + ai3*t^3 + ai4*t^4 + ai5*t^5 + ai6*t^6 + ai7*t^7

    Each polynomial has M unknown coefficients, thus we will have M*N unknown to
    solve in total, so we need to come up with M*N constraints.

    Unlike the function minSomethingTraj, where continuous equations for velocity, jerk and snap are generated, 
    and unlike the function minSomethingTraj_stop, where velocities, accelerations and jerks are equal to 0 at each waypoint,
    this function generates trajectories with only null velocities. Accelerations and above derivatives are continuous. 
    This will make the drone stop for an instant at each waypoint, and then leave in the same direction it came from.
    """

    n = len(waypoints) - 1
    nb_coeff = order * 2

    # initialize A, and B matrix
    A = np.zeros([nb_coeff * n, nb_coeff * n])
    B = np.zeros(nb_coeff * n)

    # populate B matrix.
    for i in range(n):
        B[i] = waypoints[i]
        B[i + n] = waypoints[i + 1]

    # Constraint 1 - Starting position for every segment
    for i in range(n):
        # print(i)
        A[i][nb_coeff * i:nb_coeff * (i + 1)] = get_poly_cc(nb_coeff, 0, 0)

    # Constraint 2 - Ending position for every segment
    for i in range(n):
        # print(i+n)
        A[i + n][nb_coeff * i:nb_coeff * (i + 1)] = get_poly_cc(nb_coeff, 0, times[i])

    # Constraint 3 - Starting velocity for every segment is null
    for i in range(n):
        # print(i+2*n)
        A[i + 2 * n][nb_coeff * i:nb_coeff * (i + 1)] = get_poly_cc(nb_coeff, 1, 0)

    # Constraint 4 - Ending velocity for every segment is null
    for i in range(n):
        # print(i+3*n)
        A[i + 3 * n][nb_coeff * i:nb_coeff * (i + 1)] = get_poly_cc(nb_coeff, 1, times[i])

    # Constraint 5 - Starting position derivatives (above velocity and up to order) are null
    for k in range(2, order):
        # print(4*n + k-2)
        A[4 * n + k - 2][:nb_coeff] = get_poly_cc(nb_coeff, k, 0)

    # Constraint 6 - Ending position derivatives (above velocity and up to order) are null
    for k in range(2, order):
        # print(4*n+(order-2) + k-2)
        A[4 * n + k - 2 + (order - 2)][-nb_coeff:] = get_poly_cc(nb_coeff, k, times[i])

    # Constraint 7 - All derivatives above velocity are continuous at each waypint transition
    for i in range(n - 1):
        for k in range(2, nb_coeff - 2):
            # print(4*n+2*(order-2)+k-2+i*(nb_coeff-4))
            A[4 * n + 2 * (order - 2) + k - 2 + i * (nb_coeff - 4)][
            i * nb_coeff: (i * nb_coeff + nb_coeff * 2)] = np.concatenate(
                (get_poly_cc(nb_coeff, k, times[i]), -get_poly_cc(nb_coeff, k, 0)))

    # solve for the coefficients
    Coeff = np.linalg.solve(A, B)
    return Coeff
