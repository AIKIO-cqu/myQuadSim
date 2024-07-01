import time
import numpy as np
from matplotlib import pyplot as plt
from quadFiles.quad import Quadrotor
from trajFiles.traj import Trajectory
from utils.animation import sameAxisAnimation
from ctrlFiles.ctrl_mpc import AltitudeMPC, PositionMPC, AttitudeMPC
from utils.plotting import errorPlotting


def main():
    start_time = time.time()
    # 定义初始化的参数
    Ti = 0  # 初始时间
    Ts = 0.005  # 时间间隔
    Tf = 20  # 总时间
    N = 50  # 预测区间

    # 定义无人机、轨迹、控制器
    quad = Quadrotor()
    traj = Trajectory(Tf, Ts)

    al = AltitudeMPC(quad, T=Ts, N=N)
    po = PositionMPC(quad, T=Ts, N=N)
    at = AttitudeMPC(quad, T=Ts, N=N)

    numTimeStep = int(Tf / Ts)
    t_all = np.zeros(numTimeStep)
    pos_all = np.zeros([numTimeStep, 3])
    quat_all = np.zeros([numTimeStep, 4])
    pos_err_all = np.zeros([numTimeStep, 3])
    psi_err_all = np.zeros([numTimeStep, 1])

    # 仿真循环
    for i in range(numTimeStep):
        print(i)

        # Solve altitude -> thrust
        thrusts = al.solve(traj, quad, i, N)

        # Solve position -> phid, thed
        phids, theds = po.solve(traj, quad, i, N, thrusts)

        # Solve attitude -> tau_phi, tau_the, tau_psi
        tau_phis, tau_thes, tau_psis = at.solve(traj, quad, i, N, phids, theds)

        quad.update_mpc(thrusts[0], tau_phis[0], tau_thes[0], tau_psis[0], Ts)

        # Store values
        t_all[i] = i * Ts
        pos_all[i] = quad.pos
        quat_all[i] = quad.quat
        pos_err_all[i] = quad.pos - traj.des_pos[i]
        psi_err_all[i] = quad.psi - traj.des_psi[i]

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
