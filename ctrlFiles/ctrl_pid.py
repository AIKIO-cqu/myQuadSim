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
        self.sDesCalc = np.zeros(16)  # 用于存储计算出的期望状态
        self.w_cmd = np.ones(4) * quad.params["w_hover"]  # 初始化电机命令（w_cmd）为一个数组，其所有元素都等于四旋翼悬停时的电机速度
        self.thr_int = np.zeros(3)  # 初始化一个三维零数组，用于存储积分项（thr_int），这通常用于PID控制器中的积分部分，以消除稳态误差
        self.setYawWeight()  # 设置偏航控制权重，这有助于平衡偏航控制与其他轴的控制
        self.pos_sp = np.zeros(3)  # 期望的位置
        self.vel_sp = np.zeros(3)  # 速度
        self.acc_sp = np.zeros(3)  # 加速度
        self.thrust_sp = np.zeros(3)  # 推力
        self.eul_sp = np.zeros(3)  # 欧拉角
        self.pqr_sp = np.zeros(3)  # 角速度
        self.yawFF = np.zeros(3)  # 偏航前馈控制量

    def controller(self, quad, sDes, Ts):

        # Desired State (Create a copy, hence the [:])
        # ---------------------------
        self.pos_sp[:] = sDes[0:3]
        self.vel_sp[:] = sDes[3:6]
        self.acc_sp[:] = sDes[6:9]
        self.thrust_sp[:] = sDes[9:12]
        self.eul_sp[:] = sDes[12:15]
        self.pqr_sp[:] = sDes[15:18]
        self.yawFF[:] = sDes[18]

        self.z_pos_control(quad, Ts)  # 垂直位置控制
        self.xy_pos_control(quad, Ts)  # 水平位置控制
        self.saturateVel()
        self.z_vel_control(quad, Ts)
        self.xy_vel_control(quad, Ts)
        self.thrustToAttitude(quad, Ts)
        self.attitude_control(quad, Ts)
        self.rate_control(quad, Ts)

        # Mixer 电机混合器 -> 根据计算出的推力和角速度控制输入，计算四个电机的转速命令w_cmd
        # ---------------------------
        self.w_cmd = mixerFM(quad, norm(self.thrust_sp), self.rateCtrl)

        # Add calculated Desired States 期望状态向量更新
        # ---------------------------
        self.sDesCalc[0:3] = self.pos_sp
        self.sDesCalc[3:6] = self.vel_sp
        self.sDesCalc[6:9] = self.thrust_sp
        self.sDesCalc[9:13] = self.qd
        self.sDesCalc[13:16] = self.rate_sp

    def z_pos_control(self, quad, Ts):

        # Z Position Control
        # ---------------------------
        # 计算垂直方向的位置误差pos_z_error = 期望的Z轴位置self.pos_sp[2]与四旋翼当前Z轴位置quad.pos[2]之差
        pos_z_error = self.pos_sp[2] - quad.pos[2]
        # 使用比例增益pos_P_gain[2]调整垂直速度设定点self.vel_sp[2]，以减少Z轴的位置误差
        self.vel_sp[2] += pos_P_gain[2] * pos_z_error

    # 这些都是通过PID中的P控制的
    def xy_pos_control(self, quad, Ts):

        # XY Position Control
        # ---------------------------
        # 计算XY平面内的位置误差pos_xy_error = 期望位置self.pos_sp[0:2]与四旋翼当前位置quad.pos[0:2]之差
        pos_xy_error = (self.pos_sp[0:2] - quad.pos[0:2])
        # 使用比例增益pos_P_gain[0:2]调整水平方向的速度设定点self.vel_sp[0:2]，以减少位置误差
        self.vel_sp[0:2] += pos_P_gain[0:2] * pos_xy_error

    def saturateVel(self):
        """用于对四旋翼飞行器的速度设定点进行饱和处理，确保速度不会超过预设的最大值"""
        # Saturate Velocity Setpoint
        # ---------------------------
        # Either saturate each velocity axis separately, or total velocity (prefered)
        # self.vel_sp：速度设定点，一个三维向量，表示期望的x、y、z方向速度
        if (saturateVel_separetely):
            # 分别饱和每个速度轴
            self.vel_sp = np.clip(self.vel_sp, -velMax, velMax)
        else:
            # 总速度饱和
            totalVel_sp = norm(self.vel_sp)
            if (totalVel_sp > velMaxAll):
                self.vel_sp = self.vel_sp / totalVel_sp * velMaxAll

    # 速度控制
    def z_vel_control(self, quad, Ts):

        # Z Velocity Control (Thrust in D-direction)
        # ---------------------------
        # Hover thrust (m*g) is sent as a Feed-Forward term, in order to
        # allow hover when the position and velocity error are nul

        vel_z_error = self.vel_sp[2] - quad.vel[2]

        thrust_z_sp = vel_P_gain[2] * vel_z_error - vel_D_gain[2] * quad.vel_dot[2] + quad.params["mB"] * (
                self.acc_sp[2] - quad.params["g"]) + self.thr_int[2]

        # Get thrust limits
        # The Thrust limits are negated and swapped due to NED-frame
        uMax = -quad.params["minThr"]
        uMin = -quad.params["maxThr"]

        # Apply Anti-Windup in D-direction
        stop_int_D = (thrust_z_sp >= uMax and vel_z_error >= 0.0) or (thrust_z_sp <= uMin and vel_z_error <= 0.0)

        # Calculate integral part
        if not (stop_int_D):
            self.thr_int[2] += vel_I_gain[2] * vel_z_error * Ts * quad.params["useIntergral"]
            # Limit thrust integral
            self.thr_int[2] = min(abs(self.thr_int[2]), quad.params["maxThr"]) * np.sign(self.thr_int[2])

        # Saturate thrust setpoint in D-direction
        self.thrust_sp[2] = np.clip(thrust_z_sp, uMin, uMax)

    def xy_vel_control(self, quad, Ts):

        # XY Velocity Control (Thrust in NE-direction)
        # ---------------------------
        # 速度误差计算：计算期望速度self.vel_sp[0:2]与当前速度quad.vel[0:2]之间的差值
        vel_xy_error = self.vel_sp[0:2] - quad.vel[0:2]
        # 根据速度误差、加速度和积分项计算推力thrust_xy_sp
        thrust_xy_sp = vel_P_gain[0:2] * vel_xy_error - vel_D_gain[0:2] * quad.vel_dot[0:2] + quad.params["mB"] * (
            self.acc_sp[0:2]) + self.thr_int[0:2]

        # Max allowed thrust in NE based on tilt and excess thrust
        # 推力饱和：基于最大倾斜角度和当前Z轴推力，计算水平方向上允许的最大推力，并进行饱和处理
        thrust_max_xy_tilt = abs(self.thrust_sp[2]) * np.tan(tiltMax)
        thrust_max_xy = sqrt(quad.params["maxThr"] ** 2 - self.thrust_sp[2] ** 2)
        thrust_max_xy = min(thrust_max_xy, thrust_max_xy_tilt)

        # Saturate thrust in NE-direction
        self.thrust_sp[0:2] = thrust_xy_sp
        if (np.dot(self.thrust_sp[0:2].T, self.thrust_sp[0:2]) > thrust_max_xy ** 2):
            mag = norm(self.thrust_sp[0:2])
            self.thrust_sp[0:2] = thrust_xy_sp / mag * thrust_max_xy

        # Use tracking Anti-Windup for NE-direction: during saturation, the integrator is used to unsaturate the output
        # see Anti-Reset Windup for PID controllers, L.Rundqwist, 1990
        # 积分项更新：使用跟踪防积分饱和（tracking Anti-Windup）策略更新积分项
        arw_gain = 2.0 / vel_P_gain[0:2]
        vel_err_lim = vel_xy_error - (thrust_xy_sp - self.thrust_sp[0:2]) * arw_gain
        self.thr_int[0:2] += vel_I_gain[0:2] * vel_err_lim * Ts * quad.params["useIntergral"]

    # 姿态控制 基于四元数的比例微分（PD）控制算法进行
    def thrustToAttitude(self, quad, Ts):
        """根据计算出的推力设定点（self.thrust_sp）和期望的偏航角（self.eul_sp[2]）来确定四旋翼飞行器期望的姿态"""
        # Create Full Desired Quaternion Based on Thrust Setpoint and Desired Yaw Angle
        # ---------------------------
        yaw_sp = self.eul_sp[2]  # 期望偏航角提取

        # Desired body_z axis direction 期望z轴方向计算
        body_z = -vectNormalize(self.thrust_sp)

        # Vector of desired Yaw direction in XY plane, rotated by pi/2 (fake body_y axis) 期望的偏航方向向量
        y_C = np.array([-sin(yaw_sp), cos(yaw_sp), 0.0])

        # Desired body_x axis direction 期望x轴方向计算
        body_x = np.cross(y_C, body_z)
        body_x = vectNormalize(body_x)

        # Desired body_y axis direction 期望y轴方向计算
        body_y = np.cross(body_z, body_x)

        # Desired rotation matrix 期望旋转矩阵构建
        R_sp = np.array([body_x, body_y, body_z]).T

        # Full desired quaternion (full because it considers the desired Yaw angle)
        # 将期望的旋转矩阵转换为四元数，这个四元数包含了完整的期望姿态，包括偏航角
        self.qd_full = RotToQuat(R_sp)

    # 角速度控制
    def attitude_control(self, quad, Ts):
        """根据当前的姿态误差计算出必要的角速度控制命令，以调整飞行器的姿态至期望值"""
        # Current thrust orientation e_z and desired thrust orientation e_z_d
        # 计算当前推力方向e_z 与期望的推力方向e_z_d
        e_z = quad.dcm[:, 2]
        e_z_d = -vectNormalize(self.thrust_sp)

        # Quaternion error between the 2 vectors 计算四元数误差
        qe_red = np.zeros(4)
        qe_red[0] = np.dot(e_z, e_z_d) + sqrt(norm(e_z) ** 2 * norm(e_z_d) ** 2)
        qe_red[1:4] = np.cross(e_z, e_z_d)
        qe_red = vectNormalize(qe_red)

        # Reduced desired quaternion (reduced because it doesn't consider the desired Yaw angle) 计算期望的简化四元数
        self.qd_red = quatMultiply(qe_red, quad.quat)

        # Mixed desired quaternion (between reduced and full) and resulting desired quaternion qd 混合期望四元数
        q_mix = quatMultiply(inverse(self.qd_red), self.qd_full)
        q_mix = q_mix * np.sign(q_mix[0])
        q_mix[0] = np.clip(q_mix[0], -1.0, 1.0)
        q_mix[3] = np.clip(q_mix[3], -1.0, 1.0)
        # 计算最终期望四元数
        self.qd = quatMultiply(self.qd_red, np.array(
            [cos(self.yaw_w * np.arccos(q_mix[0])), 0, 0, sin(self.yaw_w * np.arcsin(q_mix[3]))]))

        # Resulting error quaternion 计算四元数误差
        self.qe = quatMultiply(inverse(quad.quat), self.qd)

        # Create rate setpoint from quaternion error 根据四元数误差创建角速度设定点
        self.rate_sp = (2.0 * np.sign(self.qe[0]) * self.qe[1:4]) * att_P_gain

        # Limit yawFF 限制偏航前馈
        self.yawFF = np.clip(self.yawFF, -rateMax[2], rateMax[2])

        # Add Yaw rate feed-forward 添加偏航角速度前馈
        self.rate_sp += quat2Dcm(inverse(quad.quat))[:, 2] * self.yawFF

        # Limit rate setpoint 限制角速度设定点
        self.rate_sp = np.clip(self.rate_sp, -rateMax, rateMax)

    def rate_control(self, quad, Ts):

        # Rate Control
        # ---------------------------
        rate_error = self.rate_sp - quad.omega  # 计算角速度误差

        # 根据角速度误差和角速度的导数(quad.omega_dot)，使用PID控制律计算出角速度的控制输入(self.rateCtrl)
        # 这里使用了比例增益(rate_P_gain)和微分增益(rate_D_gain)
        self.rateCtrl = rate_P_gain * rate_error - rate_D_gain * quad.omega_dot  # Be sure it is right sign for the D part

    def setYawWeight(self):
        """用于平衡偏航控制与其他轴(俯仰和横滚)的控制"""
        # Calculate weight of the Yaw control gain 计算偏航(Yaw)控制增益的权重(self.yaw_w)
        roll_pitch_gain = 0.5 * (att_P_gain[0] + att_P_gain[1])
        self.yaw_w = np.clip(att_P_gain[2] / roll_pitch_gain, 0.0, 1.0)

        att_P_gain[2] = roll_pitch_gain
