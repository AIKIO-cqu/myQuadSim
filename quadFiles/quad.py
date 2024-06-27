import numpy as np
from numpy.linalg import inv
from scipy.integrate import ode
from numpy import sin, cos, tan, pi, sign

from utils.rotationConversion import *


class Quadcopter:
    def __init__(self):
        # -----------无人机参数-----------
        self.params = self.sys_params()  # 初始化无人机系统参数

        # -----------初始控制命令-----------
        ini_hover = self.init_cmd(self.params)  # 获取初始命令（悬停）
        self.params['w_cmd'] = ini_hover[0]  # 悬停时的前馈控制命令
        self.params['w_hover'] = ini_hover[1]  # 悬停时的电机转速
        self.params['thr_hover'] = ini_hover[2]  # 悬停时的推力
        self.params['tor_hover'] = ini_hover[3]  # 悬停时的电机扭矩
        self.thr = np.ones(4) * ini_hover[2]
        self.tor = np.ones(4) * ini_hover[3]

        # -----------无人机初始状态-----------
        self.state = self.init_state(self.params)
        self.pos = self.state[0:3]  # 位置
        self.quat = self.state[3:7]  # 四元数
        self.vel = self.state[7:10]  # 速度
        self.omega = self.state[10:13]  # 角速度
        self.wMotor = np.array([self.state[13],  # 电机转速
                                self.state[15],
                                self.state[17],
                                self.state[19]])
        self.vel_dot = np.zeros(3)  # 速度导数
        self.omega_dot = np.zeros(3)  # 角速度导数
        self.acc = np.zeros(3)  # 加速度
        self.extend_state()  # 计算旋转矩阵和欧拉角
        self.forces()  # 根据电机转速计算电机产生的推力和扭矩

        # -----------设置积分器-----------
        self.integrator = ode(self.state_dot).set_integrator(
            name='dopri5',
            first_step='0.00005',
            atol='10e-6',
            rtol='10e-6'
        )
        self.integrator.set_initial_value(self.state, 0)

        print('无人机初始化完成')

    def sys_params(self):
        mB = 1.2  # 电机质量
        g = 9.81  # 重力加速度
        dxm = 0.16  # 无人机翼臂长度
        dym = 0.16  # 无人机翼臂长度
        dzm = 0.05  # 无人机高度
        IB = np.array([[0.0123, 0, 0],  # 转动惯量矩阵
                       [0, 0.0123, 0],
                       [0, 0, 0.0123]])
        IRzz = 2.7e-5  # 单个电机的转动惯量

        params = {}
        params['mB'] = mB
        params['g'] = g
        params['dxm'] = dxm
        params['dym'] = dym
        params['dzm'] = dzm
        params['IB'] = IB
        params['invI'] = inv(IB)  # 转动惯量的逆矩阵
        params['IRzz'] = IRzz

        params["useIntergral"] = bool(False)  # 是否在线性速度控制中使用积分增益
        params["Cd"] = 0.1  # 阻力系数
        params["kTh"] = 1.076e-5  # 推力系数
        params["kTo"] = 1.632e-7  # 扭矩系数
        params["mixerFM"] = self.makeMixerFM(params)  # 矩阵：根据电机转速计算推力和力矩
        params["mixerFMinv"] = inv(params["mixerFM"])
        params["minThr"] = 0.1 * 4  # 总推力的最小值
        params["maxThr"] = 9.18 * 4  # 总推力的最大值
        params["minWmotor"] = 75  # 电机转速的最小值
        params["maxWmotor"] = 925  # 电机转速的最大值
        params["tau"] = 0.015  # 电机动力学的二阶系统参数
        params["kp"] = 1.0  # 电机动力学的二阶系统参数
        params["damp"] = 1.0  # 电机动力学的二阶系统参数

        params["motorc1"] = 8.49  # 电机控制参数：用于将控制命令转换为电机转速
        params["motorc0"] = 74.7  # 电机控制参数：用于将控制命令转换为电机转速
        params["motordeadband"] = 1  # 电机的死区

        return params

    def makeMixerFM(self, params):
        dxm = params["dxm"]
        dym = params["dym"]
        kTh = params["kTh"]
        kTo = params["kTo"]

        # 默认是NED
        mixerFM = np.array([[kTh, kTh, kTh, kTh],
                            [dym * kTh, -dym * kTh, -dym * kTh, dym * kTh],
                            [dxm * kTh, dxm * kTh, -dxm * kTh, -dxm * kTh],
                            [-kTo, kTo, -kTo, kTo]])

        return mixerFM

    def init_cmd(self, params):
        mB = params["mB"]
        g = params["g"]
        kTh = params["kTh"]
        kTo = params["kTo"]
        c1 = params["motorc1"]
        c0 = params["motorc0"]

        """
        w = cmd*c1 + c0
        m*g/4 = kTh*w^2
        torque = kTo*w^2
        """
        thr_hover = mB * g / 4.0  # 悬停所需的推力
        w_hover = np.sqrt(thr_hover / kTh)  # 悬停时的电机转速
        tor_hover = kTo * w_hover * w_hover  # 悬停时的电机扭矩
        cmd_hover = (w_hover - c0) / c1  # 悬停时的控制命令

        return [cmd_hover, w_hover, thr_hover, tor_hover]

    def init_state(self, params):
        x0 = 0.  # m
        y0 = 0.  # m
        z0 = 0.  # m
        phi0 = 0.  # rad
        theta0 = 0.  # rad
        psi0 = 0.  # rad

        quat = YPRToQuat(psi0, theta0, phi0)  # 将欧拉角转换为四元数

        s = np.zeros(21)  # 状态向量，包括位置、四元数、速度、角速度和电机转速及其加速度
        s[0] = x0  # x
        s[1] = y0  # y
        s[2] = z0  # z
        s[3] = quat[0]  # q0
        s[4] = quat[1]  # q1
        s[5] = quat[2]  # q2
        s[6] = quat[3]  # q3
        s[7] = 0.  # xdot
        s[8] = 0.  # ydot
        s[9] = 0.  # zdot
        s[10] = 0.  # p
        s[11] = 0.  # q
        s[12] = 0.  # r

        w_hover = params["w_hover"]  # 悬停时的电机转速
        wdot_hover = 0.  # 悬停时的电机加速度
        # 4个电机
        s[13] = w_hover
        s[14] = wdot_hover
        s[15] = w_hover
        s[16] = wdot_hover
        s[17] = w_hover
        s[18] = wdot_hover
        s[19] = w_hover
        s[20] = wdot_hover

        return s

    def extend_state(self):
        self.dcm = quat2Dcm(self.quat)  # 将四元数转化为旋转矩阵
        YRR = quatToYPR_ZYX(self.quat)  # 将四元数转化为欧拉角
        # YPR = [Yaw, pitch, roll] = [psi, theta, phi]
        self.euler = YRR[::-1]
        self.psi = YRR[0]
        self.theta = YRR[1]
        self.phi = YRR[2]
        self.ori = np.array([self.euler[2], self.euler[1], self.euler[0]])

    def forces(self):
        self.thr = self.params['kTh'] * self.wMotor * self.wMotor
        self.tor = self.params['kTo'] * self.wMotor * self.wMotor

    def state_dot(self, t, state, cmd, wind):
        """计算无人机状态的导数"""
        # step1. 获取无人机系统参数
        mB = self.params["mB"]
        g = self.params["g"]
        dxm = self.params["dxm"]
        dym = self.params["dym"]
        IB = self.params["IB"]
        IBxx = IB[0, 0]
        IByy = IB[1, 1]
        IBzz = IB[2, 2]
        Cd = self.params["Cd"]
        kTh = self.params["kTh"]
        kTo = self.params["kTo"]
        tau = self.params["tau"]
        kp = self.params["kp"]
        damp = self.params["damp"]
        minWmotor = self.params["minWmotor"]
        maxWmotor = self.params["maxWmotor"]
        IRzz = self.params["IRzz"]
        uP = 0  # 默认是0

        # step2. 获取状态信息
        x = state[0]
        y = state[1]
        z = state[2]
        q0 = state[3]
        q1 = state[4]
        q2 = state[5]
        q3 = state[6]
        xdot = state[7]
        ydot = state[8]
        zdot = state[9]
        p = state[10]
        q = state[11]
        r = state[12]
        wM1 = state[13]
        wdotM1 = state[14]
        wM2 = state[15]
        wdotM2 = state[16]
        wM3 = state[17]
        wdotM3 = state[18]
        wM4 = state[19]
        wdotM4 = state[20]

        # step3. 根据命令cmd计算每个电机的角加速度
        uMotor = cmd
        wddotM1 = (-2.0 * damp * tau * wdotM1 - wM1 + kp * uMotor[0]) / (tau ** 2)
        wddotM2 = (-2.0 * damp * tau * wdotM2 - wM2 + kp * uMotor[1]) / (tau ** 2)
        wddotM3 = (-2.0 * damp * tau * wdotM3 - wM3 + kp * uMotor[2]) / (tau ** 2)
        wddotM4 = (-2.0 * damp * tau * wdotM4 - wM4 + kp * uMotor[3]) / (tau ** 2)

        # step4. 根据电机的角速度wMotor和推力系数kTh、扭矩系数kTo计算总推力thrust和扭矩torque
        wMotor = np.array([wM1, wM2, wM3, wM4])
        wMotor = np.clip(wMotor, minWmotor, maxWmotor)
        thrust = kTh * wMotor * wMotor
        torque = kTo * wMotor * wMotor

        ThrM1 = thrust[0]
        ThrM2 = thrust[1]
        ThrM3 = thrust[2]
        ThrM4 = thrust[3]
        TorM1 = torque[0]
        TorM2 = torque[1]
        TorM3 = torque[2]
        TorM4 = torque[3]

        # step5. 根据wind对象提供的信息计算风对飞行器的影响
        [velW, qW1, qW2] = wind.randomWind(t)

        # step6. 计算状态导数
        DynamicsDot = np.array([
            [xdot],
            [ydot],
            [zdot],
            [-0.5 * p * q1 - 0.5 * q * q2 - 0.5 * q3 * r],
            [0.5 * p * q0 - 0.5 * q * q3 + 0.5 * q2 * r],
            [0.5 * p * q3 + 0.5 * q * q0 - 0.5 * q1 * r],
            [-0.5 * p * q2 + 0.5 * q * q1 + 0.5 * q0 * r],
            [(Cd * sign(velW * cos(qW1) * cos(qW2) - xdot) * (velW * cos(qW1) * cos(qW2) - xdot) ** 2 - 2 * (
                    q0 * q2 + q1 * q3) * (ThrM1 + ThrM2 + ThrM3 + ThrM4)) / mB],
            [(Cd * sign(velW * sin(qW1) * cos(qW2) - ydot) * (velW * sin(qW1) * cos(qW2) - ydot) ** 2 + 2 * (
                    q0 * q1 - q2 * q3) * (ThrM1 + ThrM2 + ThrM3 + ThrM4)) / mB],
            [(-Cd * sign(velW * sin(qW2) + zdot) * (velW * sin(qW2) + zdot) ** 2 - (ThrM1 + ThrM2 + ThrM3 + ThrM4) * (
                    q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2) + g * mB) / mB],
            [((IByy - IBzz) * q * r - uP * IRzz * (wM1 - wM2 + wM3 - wM4) * q + (
                    ThrM1 - ThrM2 - ThrM3 + ThrM4) * dym) / IBxx],
            # uP activates or deactivates the use of gyroscopic precession.
            [((IBzz - IBxx) * p * r + uP * IRzz * (wM1 - wM2 + wM3 - wM4) * p + (
                    ThrM1 + ThrM2 - ThrM3 - ThrM4) * dxm) / IByy],
            # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
            [((IBxx - IByy) * p * q - TorM1 + TorM2 - TorM3 + TorM4) / IBzz]
        ])

        # step7. 将计算得到的状态导数存储在向量sdot中
        sdot = np.zeros([21])
        sdot[0] = DynamicsDot[0]
        sdot[1] = DynamicsDot[1]
        sdot[2] = DynamicsDot[2]
        sdot[3] = DynamicsDot[3]
        sdot[4] = DynamicsDot[4]
        sdot[5] = DynamicsDot[5]
        sdot[6] = DynamicsDot[6]
        sdot[7] = DynamicsDot[7]
        sdot[8] = DynamicsDot[8]
        sdot[9] = DynamicsDot[9]
        sdot[10] = DynamicsDot[10]
        sdot[11] = DynamicsDot[11]
        sdot[12] = DynamicsDot[12]
        sdot[13] = wdotM1
        sdot[14] = wddotM1
        sdot[15] = wdotM2
        sdot[16] = wddotM2
        sdot[17] = wdotM3
        sdot[18] = wddotM3
        sdot[19] = wdotM4
        sdot[20] = wddotM4

        self.acc = sdot[7:10]

        return sdot

    def update(self, t, Ts, cmd, wind):
        # 先记录无人机当前的速度和角速度
        prev_vel = self.vel
        prev_omega = self.omega

        # 设置积分器参数
        self.integrator.set_f_params(cmd, wind)
        # 使用积分器从当前时间t积分到t+Ts，以计算下一个时间步的状态
        self.state = self.integrator.integrate(t, t + Ts)
        # self.state += Ts * self.state_dot(t, self.state, cmd, wind)

        # 更新后的状态
        self.pos = self.state[0:3]
        self.quat = self.state[3:7]
        self.vel = self.state[7:10]
        self.omega = self.state[10:13]
        self.wMotor = np.array([self.state[13], self.state[15], self.state[17], self.state[19]])

        # 计算状态导数
        self.vel_dot = (self.vel - prev_vel) / Ts
        self.omega_dot = (self.omega - prev_omega) / Ts

        # 扩展状态和力的更新
        self.extend_state()  # 计算旋转矩阵和欧拉角
        self.forces()  # 根据电机转速计算电机产生的推力和扭矩

    # def set_params_forUC(self):
    #     self.ori = np.array([self.phi, self.theta, self.psi])  # 无人机当前的偏航角、俯仰角和滚转角
    #     self.dori = self.omega  # 无人机当前绕X、Y、Z轴的角速度
    #     self.mq = self.params['mB']  # 无人机的质量
    #     self.g = self.params['g']  # 重力加速度
    #     self.Ix = self.params['IB'][0, 0]  # self.Ix、self.Iy、self.Iz: 分别是绕X、Y、Z轴的转动惯量
    #     self.Iy = self.params['IB'][1, 1]
    #     self.Iz = self.params['IB'][2, 2]
    #     self.la = self.params['dxm']  # 旋翼臂长，影响扭矩对角动量的影响
    #     self.path = [np.append(self.pos, self.ori)]
    #     self.dpos = self.vel
    #     self.min_thrust=self.params['minThr']
    #     self.max_thrust=self.params['maxThr']
