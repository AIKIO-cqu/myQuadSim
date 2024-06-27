import matplotlib.pyplot as plt


def errorPlotting(t_all, pos_err_all, ori_err_all):
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
