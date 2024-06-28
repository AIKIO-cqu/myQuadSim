import numpy as np


def mixerFM(quad, thr, moment):
    t = np.array([thr, moment[0], moment[1], moment[2]])
    w_cmd = np.sqrt(np.clip(np.dot(quad.params["mixerFMinv"], t),
                            quad.params["minWmotor"] ** 2,
                            quad.params["maxWmotor"] ** 2))

    return w_cmd
