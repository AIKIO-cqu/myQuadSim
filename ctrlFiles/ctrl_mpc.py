import numpy as np
import casadi as ca
import math
from numpy.linalg import norm
from utils.mixer import *


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
            self.quad.params['g'] - u_ / self.quad.params['m'],
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
        self.opti.subject_to(self.opti.bounded(-math.inf, z, self.quad.params['max_z']))
        self.opti.subject_to(self.opti.bounded(self.quad.params['min_dz'], dz, self.quad.params['max_dz']))

        self.opti.subject_to(self.opti.bounded(self.quad.params['minThr'], thrust, self.quad.params['maxThr']))

        opts_setting = {'ipopt.max_iter': 2000,
                        'ipopt.print_level': 0,
                        'print_time': 0,
                        'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        self.opti.solver('ipopt', opts_setting)

    def solve(self, traj, quad, idx, N_):
        """首先获取z方向的期望状态和控制，然后优化控制"""
        z_ref_ = traj.z_ref[idx:(idx + N_)]
        length = len(z_ref_)
        if length < N_:  # 如果当前获取的参考轨迹长度小于预测范围N_，则需要进行扩展
            z_ex = np.ones(N_ - length) * z_ref_[-1]
            z_ref_ = np.concatenate((z_ref_, z_ex), axis=None)

        dz_ref_ = np.diff(z_ref_)
        dz_ref_ = np.concatenate((quad.vel[2], dz_ref_), axis=None)

        ddz_ref_ = np.diff(dz_ref_)
        ddz_ref_ = np.concatenate((ddz_ref_[0], ddz_ref_), axis=None)

        thrust_ref_ = (quad.params['g'] - ddz_ref_) * quad.params['m']  # 动力学模型

        x_ = np.array([z_ref_, dz_ref_]).T
        x_ = np.concatenate((np.array([[quad.pos[2], quad.vel[2]]]), x_), axis=0)
        u_ = np.array([thrust_ref_]).T
        # x_: (N_ + 1, 2) 高度、高度速度
        # u_: (N_, 1) 推力

        next_trajectories = x_
        next_controls = u_

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
        phi = self.opt_controls[:, 0]
        the = self.opt_controls[:, 1]

        # 状态变量: 位置(x,y) 速度(dx,dy)
        self.opt_states = self.opti.variable(self.N + 1, 4)
        x = self.opt_states[:, 0]
        y = self.opt_states[:, 1]

        dx = self.opt_states[:, 2]
        dy = self.opt_states[:, 3]

        # 动力学模型(t_是推力)，获取xy方向速度、xy方向加速度
        f = lambda x_, u_, t_: ca.vertcat(*[
            x_[2], x_[3],  # dx, dy
            ca.sin(u_[1]) * t_ / self.quad.params['m'],  # ddx
            -ca.sin(u_[0]) * t_ / self.quad.params['m'],  # ddy
        ])

        # 优化器的参数：推力、期望控制输入和期望状态
        self.thrusts = self.opti.parameter(self.N, 1)
        self.opt_u_ref = self.opti.parameter(self.N, 2)
        self.opt_x_ref = self.opti.parameter(self.N + 1, 4)

        # 约束条件：mpc的预测结果，根据公式: x_t+1 = x_t + dx_t * T 计算下一时刻的状态
        self.opti.subject_to(self.opt_states[0, :] == self.opt_x_ref[0, :])
        for i in range(self.N):
            x_next = self.opt_states[i, :] + f(self.opt_states[i, :], self.opt_controls[i, :],
                                               self.thrusts[i, :]).T * self.T
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
        self.opti.subject_to(self.opti.bounded(self.quad.params['min_dx'], dx, self.quad.params['max_dx']))
        self.opti.subject_to(self.opti.bounded(self.quad.params['min_dy'], dy, self.quad.params['max_dy']))

        self.opti.subject_to(self.opti.bounded(self.quad.params['min_phi'], phi, self.quad.params['max_phi']))
        self.opti.subject_to(self.opti.bounded(self.quad.params['min_the'], the, self.quad.params['max_the']))

        opts_setting = {'ipopt.max_iter': 2000,
                        'ipopt.print_level': 0,
                        'print_time': 0,
                        'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        self.opti.solver('ipopt', opts_setting)

    def solve(self, traj, quad, idx, N_, thrusts):
        """首先获取xy平面的期望状态和控制，然后优化控制"""
        x_ref_ = traj.x_ref[idx:(idx + N_)]
        y_ref_ = traj.y_ref[idx:(idx + N_)]
        length = len(x_ref_)
        if length < N_:
            x_ex = np.ones(N_ - length) * x_ref_[-1]
            x_ref_ = np.concatenate((x_ref_, x_ex), axis=None)

            y_ex = np.ones(N_ - length) * y_ref_[-1]
            y_ref_ = np.concatenate((y_ref_, y_ex), axis=None)

        dx_ref_ = np.diff(x_ref_)
        dx_ref_ = np.concatenate((quad.vel[0], dx_ref_), axis=None)
        dy_ref_ = np.diff(y_ref_)
        dy_ref_ = np.concatenate((quad.vel[1], dy_ref_), axis=None)

        ddx_ref_ = np.diff(dx_ref_)
        ddx_ref_ = np.concatenate((ddx_ref_[0], ddx_ref_), axis=None)
        ddy_ref_ = np.diff(dy_ref_)
        ddy_ref_ = np.concatenate((ddy_ref_[0], ddy_ref_), axis=None)

        the_ref_ = np.arcsin(ddx_ref_ * quad.params['m'] / thrusts)
        phi_ref_ = -np.arcsin(ddy_ref_ * quad.params['m'] / thrusts)

        x_ = np.array([x_ref_, y_ref_, dx_ref_, dy_ref_]).T
        x_ = np.concatenate((np.array([[quad.pos[0], quad.pos[1], quad.vel[0], quad.vel[1]]]), x_), axis=0)
        u_ = np.array([phi_ref_, the_ref_]).T

        # x_: (N_ + 1, 4) x位置，y位置，x速度，y速度
        # u_: (N_, 2) 俯仰角，滚动角

        next_trajectories = x_
        next_controls = u_

        # 设置优化器参数，此处仅更新x的初始状态(x0)
        self.opti.set_value(self.opt_x_ref, next_trajectories)
        self.opti.set_value(self.opt_u_ref, next_controls)
        self.opti.set_value(self.thrusts, thrusts)

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
            (x_[4] * x_[5] * (self.quad.params['Iy'] - self.quad.params['Iz']) + self.quad.params['dxm'] * u_[0]) /
            self.quad.params['Ix'],  # ddotphi
            (x_[3] * x_[5] * (self.quad.params['Iz'] - self.quad.params['Ix']) + self.quad.params['dxm'] * u_[1]) /
            self.quad.params['Iy'],  # ddotthe
            (x_[3] * x_[4] * (self.quad.params['Ix'] - self.quad.params['Iy']) + u_[2]) / self.quad.params['Iz'],
            # ddotpsi
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
        self.opti.subject_to(self.opti.bounded(self.quad.params['min_phi'], phi, self.quad.params['max_phi']))
        self.opti.subject_to(self.opti.bounded(self.quad.params['min_the'], the, self.quad.params['max_the']))

        self.opti.subject_to(self.opti.bounded(self.quad.params['min_dphi'], dphi, self.quad.params['max_dphi']))
        self.opti.subject_to(self.opti.bounded(self.quad.params['min_dthe'], dthe, self.quad.params['max_dthe']))
        self.opti.subject_to(self.opti.bounded(self.quad.params['min_dpsi'], dpsi, self.quad.params['max_dpsi']))

        self.opti.subject_to(
            self.opti.bounded(self.quad.params['min_tau_phi'], tau_phi, self.quad.params['max_tau_phi']))
        self.opti.subject_to(
            self.opti.bounded(self.quad.params['min_tau_the'], tau_the, self.quad.params['max_tau_the']))
        self.opti.subject_to(
            self.opti.bounded(self.quad.params['min_tau_psi'], tau_psi, self.quad.params['max_tau_psi']))

        opts_setting = {'ipopt.max_iter': 2000,
                        'ipopt.print_level': 0,
                        'print_time': 0,
                        'ipopt.acceptable_tol': 1e-8,
                        'ipopt.acceptable_obj_change_tol': 1e-6}

        self.opti.solver('ipopt', opts_setting)

    def solve(self, traj, quad, idx, N_, phis, thes):
        """首先获取无人机姿态的期望状态和控制，然后优化控制"""
        phi_ref_ = phis
        the_ref_ = thes

        psi_ref_ = traj.psi_ref[idx:(idx + N_)]
        length = len(psi_ref_)
        if length < N_:
            psi_ex = np.ones(N_ - length) * psi_ref_[-1]
            psi_ref_ = np.concatenate((psi_ref_, psi_ex), axis=None)

        dphi_ref_ = np.diff(phi_ref_)
        dphi_ref_ = np.concatenate((quad.omega[0], dphi_ref_), axis=None)
        dthe_ref_ = np.diff(the_ref_)
        dthe_ref_ = np.concatenate((quad.omega[1], dthe_ref_), axis=None)
        dpsi_ref_ = np.diff(psi_ref_)
        dpsi_ref_ = np.concatenate((quad.omega[2], dpsi_ref_), axis=None)

        ddphi_ref_ = np.diff(dphi_ref_)
        ddphi_ref_ = np.concatenate((ddphi_ref_[0], ddphi_ref_), axis=None)
        ddthe_ref_ = np.diff(dthe_ref_)
        ddthe_ref_ = np.concatenate((ddthe_ref_[0], ddthe_ref_), axis=None)
        ddpsi_ref_ = np.diff(dpsi_ref_)
        ddpsi_ref_ = np.concatenate((ddpsi_ref_[0], ddpsi_ref_), axis=None)

        tau_phi_ref_ = (quad.params['Ix'] * ddphi_ref_ - dthe_ref_ * dpsi_ref_ * (
                quad.params['Iy'] - quad.params['Iz'])) / quad.params['dxm']
        tau_the_ref_ = (quad.params['Iy'] * ddthe_ref_ - dphi_ref_ * dpsi_ref_ * (
                quad.params['Iz'] - quad.params['Ix'])) / quad.params['dxm']
        tau_psi_ref_ = quad.params['Iz'] * ddpsi_ref_ - dphi_ref_ * dthe_ref_ * (quad.params['Ix'] - quad.params['Iy'])

        x_ = np.array([phi_ref_, the_ref_, psi_ref_, dphi_ref_, dthe_ref_, dpsi_ref_]).T
        x_ = np.concatenate(
            (np.array([[quad.ori[0], quad.ori[1], quad.ori[2], quad.omega[0], quad.omega[1], quad.omega[2]]]), x_),
            axis=0)
        u_ = np.array([tau_phi_ref_, tau_the_ref_, tau_psi_ref_]).T

        # x_: (N_ + 1, 6) xyz方向的角度，xyz方向的角速度
        # u_: (N_, 3) xyz方向的扭矩

        next_trajectories = x_
        next_controls = u_

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


class Control:
    def __init__(self, quad, Ts, N):
        self.al = AltitudeMPC(quad, T=Ts, N=N)
        self.po = PositionMPC(quad, T=Ts, N=N)
        self.at = AttitudeMPC(quad, T=Ts, N=N)

    def controller(self, traj, quad, i, N, Ts):
        # Solve altitude -> thrust
        thrusts = self.al.solve(traj, quad, i, N)

        # Solve position -> phi, the
        phis, thes = self.po.solve(traj, quad, i, N, thrusts)

        # Solve attitude -> tau_phi, tau_the, tau_psi
        tau_phis, tau_thes, tau_psis = self.at.solve(traj, quad, i, N, phis, thes)

        return thrusts, tau_phis, tau_thes, tau_psis
