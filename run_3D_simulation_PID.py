import time
import numpy as np
import matplotlib.pyplot as plt
from trajFiles.traj import Trajectory
from ctrlFiles.ctrl_pid import Control
from quadFiles.quad import Quadrotor
from utils.plotting import errorPlotting
from utils.windModel import Wind
from utils.animation import sameAxisAnimation


def main():
    start_time = time.time()
    Ti = 0  # Ti 代表模拟的初始时间，这里设置为0，意味着模拟从时间的起点开始
    Ts = 0.005  # Ts 是模拟的时间步长，单位是秒。这里设置为0.005秒，意味着每次迭代模拟将前进0.005秒的时间
    Tf = 20  # Tf 代表模拟的总时间，单位是秒。这里设置为20秒，意味着整个模拟将持续20秒

    quad = Quadrotor()  # 四旋翼无人机
    traj = Trajectory(Tf, Ts)  # 轨迹
    ctrl = Control(quad)  # 控制器
    wind = Wind('None', 2.0, 90, -15)  # 风

    numTimeStep = int(Tf / Ts)  # 总的时间步数 numTimeStep
    t_all = np.zeros(numTimeStep)
    pos_all = np.zeros([numTimeStep, 3])
    quat_all = np.zeros([numTimeStep, 4])
    pos_err_all = np.zeros([numTimeStep, 3])
    psi_err_all = np.zeros([numTimeStep, 1])

    # 仿真循环
    t = Ti  # 将模拟的当前时间t设置为初始时间Ti
    for i in range(numTimeStep):
        cmd = ctrl.controller(traj, quad, Ts, i)
        quad.update_pid(t, Ts, cmd, wind)

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
                      traj.wps,
                      pos_all,
                      quat_all,
                      traj.ref,
                      Ts,
                      ifsave=False)

    # 误差图表
    errorPlotting(t_all, pos_err_all, psi_err_all)


if __name__ == "__main__":
    main()
