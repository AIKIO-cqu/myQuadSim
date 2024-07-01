import time
import numpy as np
from matplotlib import pyplot as plt
from quadFiles.quad import Quadrotor
from trajFiles.traj import Trajectory
from utils.animation import sameAxisAnimation
from ctrlFiles.ctrl_mpc import Control
from utils.plotting import errorPlotting
from utils.windModel import Wind


def main():
    start_time = time.time()
    # 定义初始化的参数
    Ti = 0  # 初始时间
    Ts = 0.005  # 时间间隔
    Tf = 20.0  # 总时间
    N = 50  # 预测区间

    # 定义无人机、轨迹、控制器
    quad = Quadrotor()
    traj = Trajectory(Tf, Ts)
    ctrl = Control(quad, Ts, N)
    wind = Wind('None', 2.0, 90, -15)  # 风

    numTimeStep = int(Tf / Ts)
    t_all = np.zeros(numTimeStep)
    pos_all = np.zeros([numTimeStep, 3])
    quat_all = np.zeros([numTimeStep, 4])
    pos_err_all = np.zeros([numTimeStep, 3])
    psi_err_all = np.zeros([numTimeStep, 1])

    # 仿真循环
    t = Ti  # 将模拟的当前时间t设置为初始时间Ti
    for i in range(numTimeStep):
        print(i)
        thrusts, tau_phis, tau_thes, tau_psis = ctrl.controller(traj, quad, i, N, Ts)
        quad.update_mpc(thrusts[0], tau_phis[0], tau_thes[0], tau_psis[0], Ts)

        t_all[i] = t
        pos_all[i] = quad.pos
        quat_all[i] = quad.quat
        pos_err_all[i] = quad.pos - traj.des_pos[i]
        psi_err_all[i] = quad.psi - traj.des_psi[i]

        t += Ts
        i += 1

    end_time = time.time()
    print("Simulated {:.2f}s in {:.6f}s.".format(Tf, end_time - start_time))

    # 动画
    sameAxisAnimation(t_all,
                      pos_all,
                      quat_all,
                      traj.ref,
                      Ts,
                      ifsave=False)

    # 误差图表
    errorPlotting(t_all, pos_err_all, psi_err_all)


if __name__ == '__main__':
    main()
