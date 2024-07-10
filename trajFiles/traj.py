import math
import numpy as np


class Trajectory:
    def __init__(self, sim_time=20.0, dt=0.005):
        self.sim_time = sim_time
        self.dt = dt

        self.desiredTrajectory()
        # self.ref = self.desiredTrajectory()
        self.ref = self.generateTrajectory()

        self.x_ref = self.ref[:, 0]
        self.y_ref = self.ref[:, 1]
        self.z_ref = self.ref[:, 2]
        self.psi_ref = self.ref[:, 3]

        self.des_pos = self.ref[:, 0:3]
        self.des_psi = self.ref[:, 3]

        print('轨迹对象初始化成功')

    def desiredTrajectory(self):
        self.wps = np.loadtxt('data/data_wps.txt', dtype=float, delimiter='\t')
        self.desTraj = np.loadtxt('data/data_desTraj.txt', dtype=float, delimiter='\t')
        # self.desTraj[0:3] -> desPos, 期望的位置
        # self.desTraj[3:6] -> desVel, 速度
        # self.desTraj[6:9] -> desAcc, 加速度
        # self.desTraj[9:12] -> desThr, 推力
        # self.desTraj[12:15] -> desEul, 欧拉角
        # self.desTraj[15:18] -> desPQR, 角速度
        # self.desTraj[18] -> desYawRate, 偏航速率
        ref = []
        for i in range(int(self.sim_time / self.dt)):
            x = self.desTraj[i, 0]
            y = self.desTraj[i, 1]
            z = self.desTraj[i, 2]
            psi = self.desTraj[i, 14]
            ref.append([x, y, z, psi])
        ref = np.array(ref)
        return ref

    def generateTrajectory(self):
        ref = []
        for i in range(int(self.sim_time / self.dt)):
            t = i * self.dt
            x = 5 * math.sin(2 * math.pi * t / 10)
            y = 5 * math.cos(2 * math.pi * t / 10) - 5
            z = -0.5 * t
            yaw = 2 * math.pi * t / 10
            ref.append([x, y, z, yaw])
        ref = np.array(ref)
        return ref
