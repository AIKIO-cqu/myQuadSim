import numpy as np
import matplotlib.pyplot as plt
import time

from trajFiles.traj import Trajectory
from ctrlFiles.ctrl_pid import Control
from quadFiles.quad import Quadcopter
from utils.windModel import Wind
from utils.animation import sameAxisAnimation


def main():
    start_time = time.time()
    Ti = 0  # Ti 代表模拟的初始时间，这里设置为0，意味着模拟从时间的起点开始
    Ts = 0.005  # Ts 是模拟的时间步长，单位是秒。这里设置为0.005秒，意味着每次迭代模拟将前进0.005秒的时间
    Tf = 20  # Tf 代表模拟的总时间，单位是秒。这里设置为20秒，意味着整个模拟将持续20秒
    ifsave = 0  # ifsave是一个标志变量，用于控制是否保存模拟的结果

    quad = Quadcopter()  # 四旋翼无人机
    traj = Trajectory()  # 轨迹
    ctrl = Control(quad)  # 控制器
    wind = Wind('None', 2.0, 90, -15)  # 风

    sDes = traj.desTraj[0]  # 轨迹对象初始时刻（时间0）的期望状态（sDes）

    ctrl.controller(quad, sDes, Ts)  # 根据当前四旋翼的状态和期望状态来计算并生成初始控制命令

    numTimeStep = int(Tf / Ts)  # 总的时间步数 numTimeStep
    t_all = np.zeros(numTimeStep)
    pos_all = np.zeros([numTimeStep, 3])
    quat_all = np.zeros([numTimeStep, 4])
    pos_err_all = np.zeros([numTimeStep, 3])
    ori_err_all = np.zeros([numTimeStep, 3])

    # Run Simulation
    t = Ti  # 将模拟的当前时间t设置为初始时间Ti
    i = 0  # 初始化时间步的索引i
    while i - Tf / Ts < 0.0:
        quad.update(t, Ts, ctrl.w_cmd, wind)
        sDes = traj.desTraj[i + 1]
        ctrl.controller(quad, sDes, Ts)

        t_all[i] = t
        pos_all[i] = quad.pos
        quat_all[i] = quad.quat
        pos_err_all[i] = quad.pos - traj.des_pos[i]
        ori_err_all[i] = quad.ori - traj.des_ori[i]

        t += Ts
        i += 1

    end_time = time.time()
    print("Simulated {:.2f}s in {:.6f}s.".format(Tf, end_time - start_time))

    # 创建一个动画
    ani = sameAxisAnimation(t_all,
                            traj.wps,
                            pos_all,
                            quat_all,
                            traj.desTraj,
                            Ts,
                            ifsave=False)

    # 显示位置误差
    plt.figure(figsize=(10, 6))
    plt.plot(t_all, pos_err_all[:, 0], label='x pos_error', color='red')
    plt.plot(t_all, pos_err_all[:, 1], label='y pos_error', color='blue')
    plt.plot(t_all, pos_err_all[:, 2], label='z pos_error', color='green')
    plt.ylabel('pos error')
    plt.xlabel('Time')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.show()

    # 显示角度误差
    plt.figure(figsize=(10, 6))
    plt.plot(t_all, ori_err_all[:, 0], label='x ori_error(phi)', color='red')
    plt.plot(t_all, ori_err_all[:, 1], label='y ori_error(the)', color='blue')
    plt.plot(t_all, ori_err_all[:, 2], label='z ori_error(psi)', color='green')
    plt.ylabel('ori error')
    plt.xlabel('Time')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
