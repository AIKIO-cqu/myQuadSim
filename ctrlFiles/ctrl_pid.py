import numpy as np
from numpy import pi
from numpy import sin, cos, tan, sqrt
from numpy.linalg import norm
from utils.mixer import *
from utils.quaternionFunctions import *
from utils.rotationConversion import *

# 弧度和角度之间的转换系数
rad2deg = 180.0 / pi
deg2rad = pi / 180.0

# Set PID Gains and Max Values
# ---------------------------

# Position P gains 位置的P增益
Py = 1.0
Px = Py
Pz = 1.0

pos_P_gain = np.array([Px, Py, Pz])

# Velocity P-D gains 速度的PID增益
Pxdot = 5.0
Dxdot = 0.5
Ixdot = 5.0

Pydot = Pxdot
Dydot = Dxdot
Iydot = Ixdot

Pzdot = 4.0
Dzdot = 0.5
Izdot = 5.0

vel_P_gain = np.array([Pxdot, Pydot, Pzdot])
vel_D_gain = np.array([Dxdot, Dydot, Dzdot])
vel_I_gain = np.array([Ixdot, Iydot, Izdot])

# Attitude P gains 姿态的P增益
Pphi = 8.0
Ptheta = Pphi
Ppsi = 1.5
PpsiStrong = 8

att_P_gain = np.array([Pphi, Ptheta, Ppsi])

# Rate P-D gains 角速度的P-D增益
Pp = 1.5
Dp = 0.04

Pq = Pp
Dq = Dp

Pr = 1.0
Dr = 0.1

rate_P_gain = np.array([Pp, Pq, Pr])
rate_D_gain = np.array([Dp, Dq, Dr])

# Max Velocities 最大速度定义
uMax = 5.0
vMax = 5.0
wMax = 5.0

velMax = np.array([uMax, vMax, wMax])
velMaxAll = 5.0

saturateVel_separetely = False

# Max tilt 最大倾斜角度
tiltMax = 50.0 * deg2rad

# Max Rate 最大角速度定义
pMax = 200.0 * deg2rad
qMax = 200.0 * deg2rad
rMax = 150.0 * deg2rad

rateMax = np.array([pMax, qMax, rMax])


class Control:

    def __init__(self, quad):
        self.w_cmd = np.ones(4) * quad.params["w_hover"]  # 初始化电机命令（w_cmd）为一个数组，其所有元素都等于四旋翼悬停时的电机速度
        self.thr_int = np.zeros(3)  # 初始化一个三维零数组，用于存储积分项（thr_int），这通常用于PID控制器中的积分部分，以消除稳态误差
        self.setYawWeight()  # 设置偏航控制权重，这有助于平衡偏航控制与其他轴的控制
        self.pos_sp = np.zeros(3)  # 期望的位置
        self.vel_sp = np.zeros(3)  # 速度
        self.acc_sp = np.zeros(3)  # 加速度
        self.thrust_sp = np.zeros(3)  # 推力
        self.psi_sp = 0.  # 欧拉角
        self.yawFF = np.zeros(3)  # 偏航前馈控制量

    def controller(self, quad, sDes, Ts):

        # 获取期望轨迹
        self.pos_sp[:] = sDes[0:3]
        self.psi_sp = sDes[3]

        self.z_pos_control(quad)
        self.xy_pos_control(quad)
        self.saturateVel()
        self.z_vel_control(quad, Ts)
        self.xy_vel_control(quad, Ts)
        self.thrustToAttitude()
        self.attitude_control(quad)
        self.rate_control(quad)

        # Mixer 电机混合器 -> 根据计算出的推力和角速度控制输入，计算四个电机的转速命令w_cmd
        self.w_cmd = mixerFM(quad, norm(self.thrust_sp), self.rateCtrl)

        return self.w_cmd

    def z_pos_control(self, quad):
        # z方向的位置误差
        pos_z_error = self.pos_sp[2] - quad.pos[2]
        # 比例增益P
        self.vel_sp[2] += pos_P_gain[2] * pos_z_error

    def xy_pos_control(self, quad):
        # XY平面内的位置误差
        pos_xy_error = (self.pos_sp[0:2] - quad.pos[0:2])
        # 比例增益P
        self.vel_sp[0:2] += pos_P_gain[0:2] * pos_xy_error

    def saturateVel(self):
        """用于对四旋翼飞行器的速度设定点进行饱和处理，确保速度不会超过预设的最大值"""
        if (saturateVel_separetely):
            # 分别饱和每个速度轴
            self.vel_sp = np.clip(self.vel_sp, -velMax, velMax)
        else:
            # 总速度饱和
            totalVel_sp = norm(self.vel_sp)
            if (totalVel_sp > velMaxAll):
                self.vel_sp = self.vel_sp / totalVel_sp * velMaxAll

    def z_vel_control(self, quad, Ts):
        # z方向的速度误差
        vel_z_error = self.vel_sp[2] - quad.vel[2]
        # z方向的期望推力
        thrust_z_sp = (vel_P_gain[2] * vel_z_error -
                       vel_D_gain[2] * quad.vel_dot[2] +
                       quad.params["m"] * (self.acc_sp[2] - quad.params["g"]) +
                       self.thr_int[2])

        # 推力范围
        uMax = -quad.params["minThr"]
        uMin = -quad.params["maxThr"]

        # 判断是否需要停止积分项的累加，以防止积分饱和
        stop_int_D = (thrust_z_sp >= uMax and vel_z_error >= 0.0) or (thrust_z_sp <= uMin and vel_z_error <= 0.0)

        # 如果不停止积分，进行积分项的更新
        if not (stop_int_D):
            self.thr_int[2] += vel_I_gain[2] * vel_z_error * Ts * quad.params["useIntergral"]
            # 限制积分项的大小
            self.thr_int[2] = min(abs(self.thr_int[2]), quad.params["maxThr"]) * np.sign(self.thr_int[2])

        # 期望推力限制在允许的范围内
        self.thrust_sp[2] = np.clip(thrust_z_sp, uMin, uMax)

    def xy_vel_control(self, quad, Ts):
        # xy方向的速度误差
        vel_xy_error = self.vel_sp[0:2] - quad.vel[0:2]
        # xy方向的期望推力
        thrust_xy_sp = (vel_P_gain[0:2] * vel_xy_error -
                        vel_D_gain[0:2] * quad.vel_dot[0:2] +
                        quad.params["m"] * (self.acc_sp[0:2]) +
                        self.thr_int[0:2])

        # 推力饱和：基于最大倾斜角度和当前Z轴推力，计算水平方向上允许的最大推力，并进行饱和处理
        thrust_max_xy_tilt = abs(self.thrust_sp[2]) * np.tan(tiltMax)
        thrust_max_xy = sqrt(quad.params["maxThr"] ** 2 - self.thrust_sp[2] ** 2)
        thrust_max_xy = min(thrust_max_xy, thrust_max_xy_tilt)

        # 限制推力的范围
        self.thrust_sp[0:2] = thrust_xy_sp
        if (np.dot(self.thrust_sp[0:2].T, self.thrust_sp[0:2]) > thrust_max_xy ** 2):
            mag = norm(self.thrust_sp[0:2])
            self.thrust_sp[0:2] = thrust_xy_sp / mag * thrust_max_xy

        # 积分项更新：使用跟踪防积分饱和（tracking Anti-Windup）策略更新积分项
        arw_gain = 2.0 / vel_P_gain[0:2]
        vel_err_lim = vel_xy_error - (thrust_xy_sp - self.thrust_sp[0:2]) * arw_gain
        self.thr_int[0:2] += vel_I_gain[0:2] * vel_err_lim * Ts * quad.params["useIntergral"]

    def thrustToAttitude(self):
        """根据计算出的期望推力和期望偏航角来确定四旋翼飞行器期望的姿态"""

        yaw_sp = self.psi_sp  # 期望偏航角提取
        body_z = -vectNormalize(self.thrust_sp)
        y_C = np.array([-sin(yaw_sp), cos(yaw_sp), 0.0])
        body_x = np.cross(y_C, body_z)
        body_x = vectNormalize(body_x)
        body_y = np.cross(body_z, body_x)

        # 期望旋转矩阵构建
        R_sp = np.array([body_x, body_y, body_z]).T

        # 将期望的旋转矩阵转换为四元数
        self.qd_full = RotToQuat(R_sp)

    def attitude_control(self, quad):
        """根据当前的姿态误差计算出必要的角速度控制命令，以调整飞行器的姿态至期望值"""

        # 计算当前推力方向e_z 与期望的推力方向e_z_d
        e_z = quad.dcm[:, 2]
        e_z_d = -vectNormalize(self.thrust_sp)

        # 计算四元数误差
        qe_red = np.zeros(4)
        qe_red[0] = np.dot(e_z, e_z_d) + sqrt(norm(e_z) ** 2 * norm(e_z_d) ** 2)  # 四元数误差的实部
        qe_red[1:4] = np.cross(e_z, e_z_d)  # 四元数误差的虚部
        qe_red = vectNormalize(qe_red)  # 标准化四元数误差向量，确保它是一个单位四元数

        # 计算期望的简化四元数
        self.qd_red = quatMultiply(qe_red, quad.quat)

        # 混合期望四元数
        q_mix = quatMultiply(inverse(self.qd_red), self.qd_full)
        q_mix = q_mix * np.sign(q_mix[0])
        q_mix[0] = np.clip(q_mix[0], -1.0, 1.0)
        q_mix[3] = np.clip(q_mix[3], -1.0, 1.0)

        # 计算最终期望四元数
        self.qd = quatMultiply(self.qd_red, np.array([cos(self.yaw_w * np.arccos(q_mix[0])),
                                                      0,
                                                      0,
                                                      sin(self.yaw_w * np.arcsin(q_mix[3]))]))

        # 计算四元数误差
        self.qe = quatMultiply(inverse(quad.quat), self.qd)

        # 根据四元数误差计算期望角速度
        self.rate_sp = (2.0 * np.sign(self.qe[0]) * self.qe[1:4]) * att_P_gain

        # Limit yawFF 限制偏航前馈角速度
        self.yawFF = np.clip(self.yawFF, -rateMax[2], rateMax[2])

        # Add Yaw rate feed-forward 添加偏航角速度前馈
        self.rate_sp += quat2Dcm(inverse(quad.quat))[:, 2] * self.yawFF

        # Limit rate setpoint 限制期望角速度
        self.rate_sp = np.clip(self.rate_sp, -rateMax, rateMax)

    def rate_control(self, quad):
        # 角速度误差
        rate_error = self.rate_sp - quad.omega
        # 根据角速度误差和角速度导数（即角加速度quad.omega_dot）使用PID控制律来计算角速度的控制输入
        self.rateCtrl = (rate_P_gain * rate_error -
                         rate_D_gain * quad.omega_dot)

    def setYawWeight(self):
        """确保四旋翼飞行器的姿态控制系统中，偏航控制不会对俯仰和横滚控制产生过大的影响"""
        # 计算偏航(Yaw)控制增益的权重(self.yaw_w)
        roll_pitch_gain = 0.5 * (att_P_gain[0] + att_P_gain[1])
        self.yaw_w = np.clip(att_P_gain[2] / roll_pitch_gain, 0.0, 1.0)

        att_P_gain[2] = roll_pitch_gain
