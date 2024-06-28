import numpy as np
import casadi as ca
import math


def shift(u, x_n):
    u_end = np.concatenate((u[1:], u[-1:]))
    x_n = np.concatenate((x_n[1:], x_n[-1:]))
    return u_end, x_n


class AltitudeMPC:
    def __init__(self, quad, T=0.02, N=30, Q=np.diag([40.0, 1.0]), R=np.diag([1.0])):
        self.quad = quad
        self.T = T  # 时间步
        self.N = N  # 预测区间长度

        # 权重矩阵
        self.Q = Q
        self.R = R

        # 初始化预测出来的状态和控制
        self.next_states = np.zeros((self.N + 1, 2))
        self.u0 = np.zeros((self.N, 1))

        # 设置控制器
        self.setupController()

    def setupController(self):
        # 创建一个 CasADi 的优化问题实例
        self.opti = ca.Opti()

        # 控制变量: 推力
        self.opt_controls = self.opti.variable(self.N, 1)
        thrust = self.opt_controls

        # 状态变量: 高度，高度方向的速度
        self.opt_states = self.opti.variable(self.N + 1, 2)
        z = self.opt_states[:, 0]
        dz = self.opt_states[:, 1]

        # 动力学模型，获取z方向的速度、加速度
        f = lambda x_, u_: ca.vertcat(*[
            x_[1],
            self.quad.g - u_ / self.quad.mq,
        ])

        # 优化器的参数：期望控制输入和期望状态
        self.opt_u_ref = self.opti.parameter(self.N, 1)
        self.opt_x_ref = self.opti.parameter(self.N + 1, 2)

        # 约束条件：mpc的预测结果，根据公式: x_t+1 = x_t + dx_t * T 计算下一时刻的状态
        self.opti.subject_to(self.opt_states[0, :] == self.opt_x_ref[0, :])
        for i in range(self.N):
            x_next = self.opt_states[i, :] + f(self.opt_states[i, :], self.opt_controls[i, :]).T * self.T
            self.opti.subject_to(self.opt_states[i + 1, :] == x_next)

        # cost function
        obj = 0
        for i in range(self.N):
            state_error_ = self.opt_states[i, :] - self.opt_x_ref[i + 1, :]
            control_error_ = self.opt_controls[i, :] - self.opt_u_ref[i, :]
            obj = obj + ca.mtimes([state_error_, self.Q, state_error_.T]) \
                  + ca.mtimes([control_error_, self.R, control_error_.T])
        self.opti.minimize(obj)

        # 边界约束
        self.opti.subject_to(self.opti.bounded(-math.inf, z, self.quad.max_z))
        self.opti.subject_to(self.opti.bounded(self.quad.min_dz, dz, self.quad.max_dz))

        self.opti.subject_to(self.opti.bounded(self.quad.min_thrust, thrust, self.quad.max_thrust))

        opts_setting = {'ipopt.max_iter': 2000,
                        'ipopt.print_level': 0,
                        'print_time': 0,
                        'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        self.opti.solver('ipopt', opts_setting)

    def solve(self, next_trajectories, next_controls):
        # 设置优化器参数，此处仅更新x的初始状态(x0)
        self.opti.set_value(self.opt_x_ref, next_trajectories)
        self.opti.set_value(self.opt_u_ref, next_controls)

        # 为优化目标提供的初始猜测
        # self.next_states:上一次优化的最终状态; self.u0:上一次优化的控制输入
        self.opti.set_initial(self.opt_states, self.next_states)
        self.opti.set_initial(self.opt_controls, self.u0.reshape(self.N, 1))

        # 调用优化求解器来解决优化问题
        sol = self.opti.solve()

        # 获得控制输入
        u_res = sol.value(self.opt_controls)
        x_m = sol.value(self.opt_states)
        self.u0, self.next_states = shift(u_res, x_m)
        return u_res


class PositionMPC:
    def __init__(self, quad, T=0.02, N=30, Q=np.diag([40.0, 40.0, 1.0, 1.0]), R=np.diag([1.0, 1.0])):
        self.quad = quad
        self.T = T  # 时间步
        self.N = N  # 预测区间长度

        # 权重矩阵
        self.Q = Q
        self.R = R

        # 初始化预测出来的状态和控制
        self.next_states = np.zeros((self.N + 1, 4))
        self.u0 = np.zeros((self.N, 2))

        # 设置控制器
        self.setupController()

    def setupController(self):
        # 创建一个 CasADi 的优化问题实例
        self.opti = ca.Opti()

        # 控制变量: 俯仰角，滚动角
        self.opt_controls = self.opti.variable(self.N, 2)
        phid = self.opt_controls[:, 0]
        thed = self.opt_controls[:, 1]

        # 状态变量: 位置(x,y) 速度(dx,dy)
        self.opt_states = self.opti.variable(self.N + 1, 4)
        x = self.opt_states[:, 0]
        y = self.opt_states[:, 1]

        dx = self.opt_states[:, 2]
        dy = self.opt_states[:, 3]

        # 动力学模型(t_是推力)，获取xy方向速度、xy方向加速度
        f = lambda x_, u_, t_: ca.vertcat(*[
            x_[2], x_[3],  # dx, dy
            ca.sin(u_[1]) * t_ / self.quad.mq,  # ddx
            -ca.sin(u_[0]) * t_ / self.quad.mq,  # ddy
        ])

        # 优化器的参数：推力、期望控制输入和期望状态
        self.thrust = self.opti.parameter(self.N, 1)
        self.opt_u_ref = self.opti.parameter(self.N, 2)
        self.opt_x_ref = self.opti.parameter(self.N + 1, 4)

        # 约束条件：mpc的预测结果，根据公式: x_t+1 = x_t + dx_t * T 计算下一时刻的状态
        self.opti.subject_to(self.opt_states[0, :] == self.opt_x_ref[0, :])
        for i in range(self.N):
            x_next = self.opt_states[i, :] + f(self.opt_states[i, :], self.opt_controls[i, :],
                                               self.thrust[i, :]).T * self.T
            self.opti.subject_to(self.opt_states[i + 1, :] == x_next)

        # cost function
        obj = 0
        for i in range(self.N):
            state_error_ = self.opt_states[i, :] - self.opt_x_ref[i + 1, :]
            control_error_ = self.opt_controls[i, :] - self.opt_u_ref[i, :]
            obj = obj + ca.mtimes([state_error_, self.Q, state_error_.T]) \
                  + ca.mtimes([control_error_, self.R, control_error_.T])
        self.opti.minimize(obj)

        # 边界约束
        self.opti.subject_to(self.opti.bounded(self.quad.min_dx, dx, self.quad.max_dx))
        self.opti.subject_to(self.opti.bounded(self.quad.min_dy, dy, self.quad.max_dy))

        self.opti.subject_to(self.opti.bounded(self.quad.min_phi, phid, self.quad.max_phi))
        self.opti.subject_to(self.opti.bounded(self.quad.min_the, thed, self.quad.max_the))

        opts_setting = {'ipopt.max_iter': 2000,
                        'ipopt.print_level': 0,
                        'print_time': 0,
                        'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        self.opti.solver('ipopt', opts_setting)

    def solve(self, next_trajectories, next_controls, thrust):
        # 设置优化器参数，此处仅更新x的初始状态(x0)
        self.opti.set_value(self.opt_x_ref, next_trajectories)
        self.opti.set_value(self.opt_u_ref, next_controls)
        self.opti.set_value(self.thrust, thrust)

        # 为优化目标提供的初始猜测
        # self.next_states:上一次优化的最终状态; self.u0:上一次优化的控制输入
        self.opti.set_initial(self.opt_states, self.next_states)
        self.opti.set_initial(self.opt_controls, self.u0.reshape(self.N, 2))

        # 调用优化求解器来解决优化问题
        sol = self.opti.solve()

        # 获得控制输入
        u_res = sol.value(self.opt_controls)
        x_m = sol.value(self.opt_states)
        self.u0, self.next_states = shift(u_res, x_m)
        return u_res[:, 0], u_res[:, 1]


class AttitudeMPC:
    def __init__(self, quad, T=0.02, N=30, Q=np.diag([40.0, 40.0, 40.0, 1.0, 1.0, 1.0]), R=np.diag([1.0, 1.0, 1.0])):
        self.quad = quad
        self.T = T  # 时间步
        self.N = N  # 预测区间长度

        # 权重矩阵
        self.Q = Q
        self.R = R

        # 初始化预测出来的状态和控制
        self.next_states = np.zeros((self.N + 1, 6))
        self.u0 = np.zeros((self.N, 3))

        # 设置控制器
        self.setupController()

    def setupController(self):
        # 创建一个 CasADi 的优化问题实例
        self.opti = ca.Opti()

        # 控制变量: xyz方向的扭矩
        self.opt_controls = self.opti.variable(self.N, 3)
        tau_phi = self.opt_controls[:, 0]
        tau_the = self.opt_controls[:, 1]
        tau_psi = self.opt_controls[:, 2]

        # 状态变量: xyz欧拉角、角速度
        self.opt_states = self.opti.variable(self.N + 1, 6)
        phi = self.opt_states[:, 0]
        the = self.opt_states[:, 1]
        psi = self.opt_states[:, 2]

        dphi = self.opt_states[:, 3]
        dthe = self.opt_states[:, 4]
        dpsi = self.opt_states[:, 5]

        # 动力学模型，获取xyz角速度和角加速度
        f = lambda x_, u_: ca.vertcat(*[
            x_[3], x_[4], x_[5],  # dotphi, dotthe, dotpsi
            (x_[4] * x_[5] * (self.quad.Iy - self.quad.Iz) + self.quad.la * u_[0]) / self.quad.Ix,  # ddotphi
            (x_[3] * x_[5] * (self.quad.Iz - self.quad.Ix) + self.quad.la * u_[1]) / self.quad.Iy,  # ddotthe
            (x_[3] * x_[4] * (self.quad.Ix - self.quad.Iy) + u_[2]) / self.quad.Iz,  # ddotpsi
        ])

        # 优化器的参数：期望控制输入和期望状态
        self.opt_u_ref = self.opti.parameter(self.N, 3)
        self.opt_x_ref = self.opti.parameter(self.N + 1, 6)

        # 约束条件：mpc的预测结果，根据公式: x_t+1 = x_t + dx_t * T 计算下一时刻的状态
        self.opti.subject_to(self.opt_states[0, :] == self.opt_x_ref[0, :])
        for i in range(self.N):
            x_next = self.opt_states[i, :] + f(self.opt_states[i, :], self.opt_controls[i, :]).T * self.T
            self.opti.subject_to(self.opt_states[i + 1, :] == x_next)

        # cost function
        obj = 0
        for i in range(self.N):
            state_error_ = self.opt_states[i, :] - self.opt_x_ref[i + 1, :]
            control_error_ = self.opt_controls[i, :] - self.opt_u_ref[i, :]
            obj = obj + ca.mtimes([state_error_, self.Q, state_error_.T]) \
                  + ca.mtimes([control_error_, self.R, control_error_.T])
        self.opti.minimize(obj)

        # 边界约束
        self.opti.subject_to(self.opti.bounded(self.quad.min_phi, phi, self.quad.max_phi))
        self.opti.subject_to(self.opti.bounded(self.quad.min_the, the, self.quad.max_the))

        self.opti.subject_to(self.opti.bounded(self.quad.min_dphi, dphi, self.quad.max_dphi))
        self.opti.subject_to(self.opti.bounded(self.quad.min_dthe, dthe, self.quad.max_dthe))
        self.opti.subject_to(self.opti.bounded(self.quad.min_dpsi, dpsi, self.quad.max_dpsi))

        self.opti.subject_to(self.opti.bounded(self.quad.min_tau_phi, tau_phi, self.quad.max_tau_phi))
        self.opti.subject_to(self.opti.bounded(self.quad.min_tau_the, tau_the, self.quad.max_tau_the))
        self.opti.subject_to(self.opti.bounded(self.quad.min_tau_psi, tau_psi, self.quad.max_tau_psi))

        opts_setting = {'ipopt.max_iter': 2000,
                        'ipopt.print_level': 0,
                        'print_time': 0,
                        'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        self.opti.solver('ipopt', opts_setting)

    def solve(self, next_trajectories, next_controls):
        # 设置优化器参数，此处仅更新x的初始状态(x0)
        self.opti.set_value(self.opt_x_ref, next_trajectories)
        self.opti.set_value(self.opt_u_ref, next_controls)

        # 为优化目标提供的初始猜测
        # self.next_states:上一次优化的最终状态; self.u0:上一次优化的控制输入
        self.opti.set_initial(self.opt_states, self.next_states)
        self.opti.set_initial(self.opt_controls, self.u0.reshape(self.N, 3))

        # 调用优化求解器来解决优化问题
        sol = self.opti.solve()

        # 获得控制输入
        u_res = sol.value(self.opt_controls)
        x_m = sol.value(self.opt_states)
        self.u0, self.next_states = shift(u_res, x_m)
        return u_res[:, 0], u_res[:, 1], u_res[:, 2]
