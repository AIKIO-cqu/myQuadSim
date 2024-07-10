import numpy as np
import torch
from utils.mixer import *

# state inds
POS = slice(0, 3)
VEL = slice(3, 6)
QUAT = slice(6, 10)
OMEGA = slice(10, 13)
# action inds
THRUST = 0
ANGVEL = slice(1, 4)


class MPPIConfig:
    lam = 0.05  # temparature
    H = 40  # horizon
    N = 4096  # number of samples
    K_delay = 1
    sim_K_delay = 1

    sample_std = [0.25, 0.7, 0.7,
                  0.7]  # standard deviation for sampling = [thrust (unit = hovering thrust), omega (unit = rad/s)]
    gamma_mean = 1.0  # learning rate
    gamma_Sigma = 0.  # learning rate
    omega_gain = 40.  # gain of the low-level controller
    discount = 0.99  # discount factor in MPPI
    a_min = [0., -5., -5., -2.]  # bounds of sampling action = [thrust, omega (unit = rad/s)]
    a_max = [0., 5., 5., 2.]

    # reward functions
    alpha_p = 5.0
    alpha_w = 0.0
    alpha_a = 0.0
    alpha_R = 3.0
    alpha_v = 0.0
    alpha_z = 0.0
    alpha_yaw = 0.0

    noise_measurement_std = np.zeros(10)
    noise_measurement_std[:3] = 0.005
    noise_measurement_std[3:6] = 0.005
    noise_measurement_std[6:10] = 0.01


class MPPIController:
    def __init__(self, quad):
        self.config = quad.params
        self.mppi_config = MPPIConfig()
        self.thrust_hover = self.config['m'] * self.config['g']
        self.dt = 0.005  # 这个这里先手动设置，需要跟Ts保持一致
        self.t_H = np.arange(0, self.mppi_config.H * self.dt, self.dt)
        # self.t_H = np.arange(t, t + self.mppi_config.H * self.dt, self.dt)
        self.u = torch.zeros((self.mppi_config.N, self.mppi_config.H))
        self.angvel = torch.zeros((self.mppi_config.N, self.mppi_config.H, 3))
        self.a_mean = torch.zeros((self.mppi_config.H, 4))
        self.a_mean[:, 0] = self.thrust_hover
        sample_std = self.mppi_config.sample_std
        self.sampling_std = torch.tensor([
            sample_std[0] * self.thrust_hover,
            sample_std[1],
            sample_std[2],
            sample_std[3]
        ])
        self.a_min = torch.tensor(self.mppi_config.a_min)
        self.a_max = torch.tensor(self.mppi_config.a_max)
        self.a_max[0] = self.mppi_config.a_max[0] * 4

    def sample(self):
        return torch.normal(mean=self.a_mean.repeat(self.mppi_config.N, 1, 1), std=self.sampling_std)

    def qmultiply_loop(self, quat, rotquat, states, H):
        """
        quat：初始化姿态
        rotquat：旋转四元数序列，表示在每个时间步的旋转变化。
        states：状态矩阵，将在每个时间步更新四元数部分。
        H：时间步数，即 rotquat 和 states 的时间维度。
        """
        # print("rotquat, quat", rotquat[:, 0], quat)
        for h in range(H):
            states[:, h, 6:10] = self.quat_mult(rotquat[:, h], quat)
            quat = states[:, h, 6:10]
        return states

    def quat_mult(self, q1, q2):
        """
        进行四元数乘法
        q1 的形状为 (N, 4) 或 (H, 4)
        q2 的形状为 (N, 4) 或 (4,)
        """
        if isinstance(q2, np.ndarray):
            q2 = torch.tensor(q2, dtype=torch.float32)
        if len(q2.shape) == 1:
            q2 = q2.unsqueeze(0).expand(q1.shape[0], -1)

        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack((w, x, y, z), dim=1)

    def rollout(self, startstate, actions):
        startstate = torch.tensor(startstate)
        N, H, _ = actions.shape
        xdim = startstate.shape[0]
        e3 = torch.tensor([0, 0, 1])
        dt = self.dt

        self.u = self.u + (actions[:, :, 0] / self.config['m'] - self.u)
        self.angvel = self.angvel + (actions[:, :, 1:4] - self.angvel)

        u = self.u
        angvel = self.angvel

        # 初始化状态矩阵
        states = torch.zeros((N, H, xdim))

        # 计算四元数
        # 角度乘以时间步长
        dang = angvel * dt
        angnorm = torch.linalg.norm(dang, axis=2)
        hangnorm = 0.5 * angnorm
        axisfactor = torch.where(angnorm < 1e-8, 0.5 + angnorm ** 2 / 48, torch.sin(hangnorm) / angnorm).unsqueeze(2)

        # 计算四元数
        rotquat = torch.cat((torch.cos(hangnorm).unsqueeze(2), axisfactor * dang), dim=2)
        # print("74", startstate[6:10])
        quat = startstate[6:10]

        states = self.qmultiply_loop(quat, rotquat, states, H)
        # print("101", u, states[:, :, 6:10], self.config['mB'])
        # print("102", u.unsqueeze(2).shape, self.z_from_q(states[:, :, 6:10]).shape)

        accel = u.unsqueeze(2) * self.z_from_q(states[:, :, 6:10]) - self.config['m'] * e3.view(1, 1, 3)

        states[:, :, 3:6] = startstate[3:6] + torch.cumsum(accel * dt, dim=1)
        states[:, :, 0:3] = startstate[0:3] + torch.cumsum(states[:, :, 3:6] * dt, dim=1)

        return states

    def z_from_q(self, q):
        """
        将四元数转换为 z 轴方向向量。
        输入 q 的形状为 (N, H, 4)。
        输出的形状为 (N, H, 3)。
        """
        # 计算每个分量的数组
        z_x = 2 * (q[:, :, 1] * q[:, :, 3] + q[:, :, 0] * q[:, :, 2])
        z_y = 2 * (q[:, :, 2] * q[:, :, 3] - q[:, :, 0] * q[:, :, 1])
        z_z = 1 - 2 * (q[:, :, 1] ** 2 + q[:, :, 2] ** 2)

        # 使用 torch.stack 将它们组合成形状为 (N, H, 3) 的数组
        return torch.stack((z_x, z_y, z_z), dim=2)

    def policy(self, state, Ts, traj, i):
        # print("92", self.a_mean)
        self.a_mean[:-1, :] = self.a_mean[1:, :].clone()

        actions = self.sample()

        states = self.rollout(state, actions)

        # 这个的作用是什么？
        # state_ref = torch.as_tensor(traj.futureDesiredStates(self.t_H + Ts), dtype=torch.float32).unsqueeze(0)
        state_ref = torch.from_numpy(traj.ref[i:i + self.mppi_config.H - 1]).unsqueeze(0)
        while state_ref.shape[1] < self.mppi_config.H:
            num_to_add = self.mppi_config.H - state_ref.shape[1]
            last_element = state_ref[:, -1, :].unsqueeze(0)
            repeat_last_element = last_element.expand(-1, num_to_add, -1)
            state_ref = torch.cat((state_ref, repeat_last_element), dim=1)

        poserr = states[:, :, 0:3] - state_ref[:, :, 0:3]
        # cost = self.mppi_config.alpha_p * torch.sum(torch.linalg.norm(poserr, dim=2), dim=1) + \
        #        self.mppi_config.alpha_R * torch.sum(self.qdistance(states[:, :, 6:10], state_ref[:, :, 6:10]), dim=1)
        cost = self.mppi_config.alpha_p * torch.sum(torch.linalg.norm(poserr, dim=2), dim=1)

        cost *= self.dt
        cost -= torch.min(cost)
        weight = torch.softmax(-cost / self.mppi_config.lam, dim=0)
        self.a_mean = torch.sum(actions * weight.view(self.mppi_config.N, 1, 1), dim=0)

        a_final = self.a_mean[0, :]
        a_final[0] = a_final[0] / self.config['m']
        return a_final

    def qdistance(self, q1, q2):
        return 1 - torch.sum(q1 * q2, axis=2) ** 2


# 控制类
class Control:
    def __init__(self, quad):
        self.sDesCalc = np.zeros(16)  # 用于存储计算出的期望状态
        self.w_cmd = np.ones(4) * quad.params["w_hover"]  # 初始化电机命令（w_cmd）为一个数组，其所有元素都等于四旋翼悬停时的电机速度
        self.thr_int = np.zeros(3)  # 初始化一个三维零数组，用于存储积分项（thr_int），这通常用于PID控制器中的积分部分，以消除稳态误差
        self.pos_sp = np.zeros(3)  # 期望的位置
        self.vel_sp = np.zeros(3)  # 速度
        self.acc_sp = np.zeros(3)  # 加速度
        self.thrust_sp = np.zeros(3)  # 推力
        self.eul_sp = np.zeros(3)  # 欧拉角
        self.pqr_sp = np.zeros(3)  # 角速度
        self.yawFF = np.zeros(3)  # 偏航前馈控制量
        self.mppi_controller = MPPIController(quad)

    def controller(self, traj, quad, Ts, i):
        current_state = np.concatenate((quad.pos, quad.vel, quad.quat, quad.omega))
        action = self.mppi_controller.policy(current_state, Ts, traj, i)
        self.thrust_sp = action[0]
        self.rate_sp = action[1:4]
        self.w_cmd = mixerFM(quad, self.thrust_sp, self.rate_sp)
        return self.w_cmd
