import time
import numpy as np
from matplotlib import pyplot as plt
from quadFiles.quad2 import Quadrotor
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

    # his_thrust = []
    # his_tau_phi = []
    # his_tau_the = []
    # his_tau_psi = []
    # his_time = []

    numTimeStep = int(Tf / Ts)
    t_all = np.zeros(numTimeStep)
    pos_all = np.zeros([numTimeStep, 3])
    quat_all = np.zeros([numTimeStep, 4])
    pos_err_all = np.zeros([numTimeStep, 3])
    ori_err_all = np.zeros([numTimeStep, 3])

    # 仿真循环
    i = 0
    while i - Tf / Ts < 0.0:
        # print(i)

        # Solve altitude -> thrust
        next_al_trajectories, next_al_controls = traj.desired_altitude_mpc(quad, i, N)
        thrusts = al.solve(next_al_trajectories, next_al_controls)

        # Solve position -> phid, thed
        next_po_trajectories, next_po_controls = traj.desired_position_mpc(quad, i, N, thrusts)
        phids, theds = po.solve(next_po_trajectories, next_po_controls, thrusts)

        # Solve attitude -> tau_phi, tau_the, tau_psi
        next_at_trajectories, next_at_controls = traj.desired_attitude_mpc(quad, i, N, phids, theds)
        tau_phis, tau_thes, tau_psis = at.solve(next_at_trajectories, next_at_controls)

        quad.updateConfiguration(thrusts[0], tau_phis[0], tau_thes[0], tau_psis[0], Ts)

        # Store values
        t_all[i] = i * Ts
        pos_all[i] = quad.pos
        quat_all[i] = quad.quat
        pos_err_all[i] = quad.pos - traj.des_pos[i]
        ori_err_all[i] = quad.ori - traj.des_ori[i]
        # his_thrust.append(thrusts[0])
        # his_tau_phi.append(tau_phis[0])
        # his_tau_the.append(tau_thes[0])
        # his_tau_psi.append(tau_psis[0])
        # his_time.append(i * Ts)

        i += 1

    end_time = time.time()
    print("Simulated {:.2f}s in {:.6f}s.".format(Tf, end_time - start_time))

    # 动画
    sameAxisAnimation(t_all,
                      traj.wps,
                      pos_all,
                      quat_all,
                      traj.desTraj,
                      Ts,
                      ifsave=False)

    # 误差图表
    errorPlotting(t_all, pos_err_all, ori_err_all)


if __name__ == '__main__':
    main()
