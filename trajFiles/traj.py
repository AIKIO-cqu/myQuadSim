import math
import numpy as np


class Trajectory:
    def __init__(self, sim_time=20.0, dt=0.005):
        self.wps = np.loadtxt('data/data_wps.txt', dtype=float, delimiter='\t')
        self.desTraj = np.loadtxt('data/data_desTraj.txt', dtype=float, delimiter='\t')
        # self.desTraj[0:3] -> desPos, 期望的位置
        # self.desTraj[3:6] -> desVel, 速度
        # self.desTraj[6:9] -> desAcc, 加速度
        # self.desTraj[9:12] -> desThr, 推力
        # self.desTraj[12:15] -> desEul, 欧拉角
        # self.desTraj[15:18] -> desPQR, 角速度
        # self.desTraj[18] -> desYawRate, 偏航速率

        self.sim_time = sim_time
        self.dt = dt
        self.ref = self.desiredTrajectory()

        self.x_ref = np.array(self.ref)[:, 0]
        self.y_ref = np.array(self.ref)[:, 1]
        self.z_ref = np.array(self.ref)[:, 2]
        self.phi_ref = np.array(self.ref)[:, 3]
        self.the_ref = np.array(self.ref)[:, 4]
        self.psi_ref = np.array(self.ref)[:, 5]

        self.des_pos = np.array(self.ref)[:, 0:3]
        self.des_ori = np.array(self.ref)[:, 3:6]

        print('轨迹对象初始化成功')

    def desiredTrajectory(self):  # MPC
        ref = []
        for i in range(int(self.sim_time / self.dt)):
            x = self.desTraj[i + 1, 0]
            y = self.desTraj[i + 1, 1]
            z = self.desTraj[i + 1, 2]
            phi = self.desTraj[i + 1, 12]
            the = self.desTraj[i + 1, 13]
            psi = self.desTraj[i + 1, 14]
            ref.append([x, y, z, phi, the, psi])
        return ref

    def desired_altitude_mpc(self, quad, idx, N_):
        """获取z方向的期望状态和控制"""
        z_ref_ = self.z_ref[idx:(idx + N_)]
        length = len(z_ref_)
        if length < N_:  # 如果当前获取的参考轨迹长度小于预测范围N_，则需要进行扩展
            z_ex = np.ones(N_ - length) * z_ref_[-1]
            z_ref_ = np.concatenate((z_ref_, z_ex), axis=None)

        dz_ref_ = np.diff(z_ref_)
        dz_ref_ = np.concatenate((quad.vel[2], dz_ref_), axis=None)

        ddz_ref_ = np.diff(dz_ref_)
        ddz_ref_ = np.concatenate((ddz_ref_[0], ddz_ref_), axis=None)

        thrust_ref_ = (quad.g - ddz_ref_) * quad.mq  # 动力学模型

        x_ = np.array([z_ref_, dz_ref_]).T
        x_ = np.concatenate((np.array([[quad.pos[2], quad.vel[2]]]), x_), axis=0)
        u_ = np.array([thrust_ref_]).T
        # x_: (N_ + 1, 2) 高度、高度速度
        # u_: (N_, 1) 推力
        return x_, u_

    def desired_position_mpc(self, quad, idx, N_, thrust):
        """获取xy平面的期望状态和控制"""
        x_ref_ = self.x_ref[idx:(idx + N_)]
        y_ref_ = self.y_ref[idx:(idx + N_)]
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

        the_ref_ = np.arcsin(ddx_ref_ * quad.mq / thrust)
        phi_ref_ = -np.arcsin(ddy_ref_ * quad.mq / thrust)

        x_ = np.array([x_ref_, y_ref_, dx_ref_, dy_ref_]).T
        x_ = np.concatenate((np.array([[quad.pos[0], quad.pos[1], quad.vel[0], quad.vel[1]]]), x_), axis=0)
        u_ = np.array([phi_ref_, the_ref_]).T

        # x_: (N_ + 1, 4) x位置，y位置，x速度，y速度
        # u_: (N_, 2) 俯仰角，滚动角
        return x_, u_

    def desired_attitude_mpc(self, quad, idx, N_, phid, thed):
        """获取无人机姿态的期望状态和控制"""
        phi_ref_ = phid
        the_ref_ = thed

        psi_ref_ = self.psi_ref[idx:(idx + N_)]
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

        tau_phi_ref_ = (quad.Ix * ddphi_ref_ - dthe_ref_ * dpsi_ref_ * (quad.Iy - quad.Iz)) / quad.la
        tau_the_ref_ = (quad.Iy * ddthe_ref_ - dphi_ref_ * dpsi_ref_ * (quad.Iz - quad.Ix)) / quad.la
        tau_psi_ref_ = quad.Iz * ddpsi_ref_ - dphi_ref_ * dthe_ref_ * (quad.Ix - quad.Iy)

        x_ = np.array([phi_ref_, the_ref_, psi_ref_, dphi_ref_, dthe_ref_, dpsi_ref_]).T
        x_ = np.concatenate(
            (np.array([[quad.ori[0], quad.ori[1], quad.ori[2], quad.omega[0], quad.omega[1], quad.omega[2]]]), x_), axis=0)
        u_ = np.array([tau_phi_ref_, tau_the_ref_, tau_psi_ref_]).T

        # x_: (N_ + 1, 6) xyz方向的角度，xyz方向的角速度
        # u_: (N_, 3) xyz方向的扭矩
        return x_, u_
