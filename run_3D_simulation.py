from random import random

import numpy as np


def move_to_pose(start_pos, goal_pos):
    x = start_pos[0]
    y = start_pos[1]
    z = start_pos[2]
    direction_vector = (goal_pos - start_pos) / np.linalg.norm(goal_pos - start_pos)

    x_diff = goal_pos[0] - start_pos[0]
    y_diff = goal_pos[1] - start_pos[1]
    z_diff = goal_pos[2] - start_pos[2]
    distance = np.sqrt(x_diff ** 2 + y_diff ** 2 + z_diff ** 2)

    count = 0
    while distance > 0.001:
        count += 1
        v = 0.01 * distance

        x = x + v * direction_vector[0]
        y = y + v * direction_vector[1]
        z = z + v * direction_vector[2]

        x_diff = goal_pos[0] - x
        y_diff = goal_pos[1] - y
        z_diff = goal_pos[2] - z
        distance = np.sqrt(x_diff ** 2 + y_diff ** 2 + z_diff ** 2)
        print(count, distance)


def main():
    # 将一个点移动到另一个点
    x_start = random() * 20.0
    y_start = random() * 20.0
    z_start = random() * 20.0
    start_pos = np.array([x_start, y_start, z_start])
    x_goal = random() * 20.0
    y_goal = random() * 20.0
    z_goal = random() * 20.0
    goal_pos = np.array([x_goal, y_goal, z_goal])

    print('start pos:', start_pos)
    print('goal  pos:', goal_pos)

    move_to_pose(start_pos, goal_pos)


if __name__ == '__main__':
    main()
