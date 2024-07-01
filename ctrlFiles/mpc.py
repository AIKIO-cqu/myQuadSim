import casadi as ca
import math
import numpy as np


class MPC:
    def __init__(self, quad, T=0.02, N=30, Q=np.diag([40.0, 1.0]), R=np.diag([1.0])):
        self.quad = quad
        self.T = T  # 时间步
        self.N = N  # 预测区间长度

        # 权重矩阵
        self.Q = Q
        self.R = R

        # 初始化预测出来的状态和控制
        self.next_states = np.zeros((self.N + 1, 12))
        self.u0 = np.zeros((self.N, 6))

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
