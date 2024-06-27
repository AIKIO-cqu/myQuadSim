import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

from utils.rotationConversion import *

numFrames = 8


def sameAxisAnimation(t_all, waypoints, pos_all, quat_all, sDes_tr_all, Ts, ifsave=False):
    # 从pos_all数组中提取所有时间步的x，y，z坐标
    x = pos_all[:, 0]
    y = pos_all[:, 1]
    z = pos_all[:, 2]

    # 提取期望轨迹的所有时间步的x，y，z坐标
    xDes = sDes_tr_all[:, 0]
    yDes = sDes_tr_all[:, 1]
    zDes = sDes_tr_all[:, 2]

    # 提取所有航点的x，y，z坐标
    x_wp = waypoints[:, 0]
    y_wp = waypoints[:, 1]
    z_wp = waypoints[:, 2]

    z = -z  # 如果坐标系是"NED"，这行代码将所有的z坐标取反，使其符合"NED"坐标系中z轴向下的约定
    zDes = -zDes  # 取反期望轨迹中的z坐标
    z_wp = -z_wp  # 取反所有航点的z坐标

    # 创建一个新的matplotlib图形对象fig。这是一个容器，用于存放所有的绘图元素
    fig = plt.figure()
    """ 这里创建了一个mpl_toolkits.mplot3d.axes3d.Axes3D的实例ax，它是一个三维坐标轴，附加到前面创建的图形对象fig上，
        参数auto_add_to_figure=False表示这个坐标轴对象不会自动添加到图形中，因此需要手动添加 """
    ax = p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)
    line1, = ax.plot([], [], [], lw=2, color='red')  # 四旋翼的一个电机臂
    line2, = ax.plot([], [], [], lw=2, color='blue')  # 四旋翼另一个电机臂
    line3, = ax.plot([], [], [], '--', lw=1, color='blue')  # 四旋翼的飞行轨迹

    # Setting the axes properties 设置三维坐标轴的显示属性
    extraEachSide = 0.5  # 定义了在自动计算的范围之外额外添加的空间，以确保图形不会紧贴坐标轴
    maxRange = 0.5 * np.array(
        [x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() + extraEachSide  # 计算x、y、z坐标轴的最大显示范围
    mid_x = 0.5 * (x.max() + x.min())  # 分别计算x、y、z坐标轴的中点
    mid_y = 0.5 * (y.max() + y.min())
    mid_z = 0.5 * (z.max() + z.min())

    # 设置坐标轴的范围和标签
    ax.set_xlim3d([mid_x - maxRange, mid_x + maxRange])
    ax.set_xlabel('X')

    ax.set_ylim3d([mid_y + maxRange, mid_y - maxRange])

    ax.set_ylabel('Y')
    ax.set_zlim3d([mid_z - maxRange, mid_z + maxRange])
    ax.set_zlabel('Altitude')

    # 在图形的左上角添加一个文本对象，用于显示当前模拟时间
    titleTime = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    # 分别用于存储 轨迹类型 和 偏航轨迹类型 的字符串
    trajType = 'Minimum Jerk Trajectory'
    yawTrajType = 'Follow'

    # 使用ax.scatter在坐标轴上显示 航点
    ax.scatter(x_wp, y_wp, z_wp, color='green', alpha=1, marker='o', s=25)
    # 使用ax.plot在坐标轴上显示 期望轨迹线
    ax.plot(xDes, yDes, zDes, ':', lw=1.3, color='green')

    # 分别在图形右上角添加文本对象，显示轨迹类型和偏航类型
    titleType1 = ax.text2D(0.95, 0.95, trajType, transform=ax.transAxes, horizontalalignment='right')
    titleType2 = ax.text2D(0.95, 0.91, 'Yaw: ' + yawTrajType, transform=ax.transAxes, horizontalalignment='right')

    def updateLines(i):
        """核心函数，定义了动画每一帧的更新逻辑"""
        time = t_all[i * numFrames]  # 从t_all数组中提取当前帧对应的时间点
        pos = pos_all[i * numFrames]  # 从pos_all数组中提取当前帧的四旋翼位置

        # 分别提取当前位置的x、y、z坐标
        x = pos[0]
        y = pos[1]
        z = pos[2]

        # 提取从模拟开始到当前帧的所有x、y、z坐标，用于绘制四旋翼的飞行轨迹
        x_from0 = pos_all[0:i * numFrames, 0]
        y_from0 = pos_all[0:i * numFrames, 1]
        z_from0 = pos_all[0:i * numFrames, 2]

        # 从params字典中提取四旋翼的尺寸参数，这些参数可能用于计算电机的位置
        # dxm = params["dxm"]#0.16
        # dym = params["dym"]#0.16
        # dzm = params["dzm"]#0.05
        dxm = 0.16
        dym = 0.16
        dzm = 0.05

        # 提取当前帧的四旋翼姿态四元数
        quat = quat_all[i * numFrames]

        # 根据配置文件中的坐标系方向（"NED"或"ENU"），调整z坐标和四元数的符号
        z = -z
        z_from0 = -z_from0
        quat = np.array([quat[0], -quat[1], -quat[2], quat[3]])

        # 使用utils.quat2Dcm函数将四元数转换为方向余弦矩阵（DCM），这个矩阵用于后续计算电机的位置
        R = quat2Dcm(quat)

        # 定义电机相对于四旋翼中心的位置
        motorPoints = np.array(
            [[dxm, -dym, dzm], [0, 0, 0], [dxm, dym, dzm], [-dxm, dym, dzm], [0, 0, 0], [-dxm, -dym, dzm]])
        # 使用方向余弦矩阵R和电机位置motorPoints计算电机在全局坐标系中的位置
        motorPoints = np.dot(R, np.transpose(motorPoints))

        # 将电机的全局坐标系位置与四旋翼的位置相加，得到最终位置
        motorPoints[0, :] += x
        motorPoints[1, :] += y
        motorPoints[2, :] += z

        # line1, line2更新表示四旋翼电机位置的线条数据
        line1.set_data(motorPoints[0, 0:3], motorPoints[1, 0:3])
        line1.set_3d_properties(motorPoints[2, 0:3])
        line2.set_data(motorPoints[0, 3:6], motorPoints[1, 3:6])
        line2.set_3d_properties(motorPoints[2, 3:6])
        # line3更新表示四旋翼飞行轨迹的线条数据
        line3.set_data(x_from0, y_from0)
        line3.set_3d_properties(z_from0)
        # 更新图形顶部的时间显示，显示当前模拟时间
        titleTime.set_text(u"Time = {:.2f} s".format(time))

        return line1, line2

    def ini_plot():
        """用于初始化动画中的线对象"""
        line1.set_data(np.empty([1]), np.empty([1]))
        line1.set_3d_properties(np.empty([1]))
        line2.set_data(np.empty([1]), np.empty([1]))
        line2.set_3d_properties(np.empty([1]))
        line3.set_data(np.empty([1]), np.empty([1]))
        line3.set_3d_properties(np.empty([1]))

        return line1, line2, line3

    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, updateLines, init_func=ini_plot, frames=len(t_all[0:-2:numFrames]),
                                       interval=(Ts * 1000 * numFrames), blit=False)

    if (ifsave):
        line_ani.save('animation_{0:.0f}_{1:.0f}.gif'.format(5, 3), dpi=80, writer='imagemagick',
                      fps=25)

    plt.show()
    return line_ani
