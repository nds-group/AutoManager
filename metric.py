import numpy as np
import pandas as pd
import tensorflow as tf


def metric_model(metric_l, metric_inputs):
    """
    Targeted metric (must use only tf functions)
    """
    overprov_start = 0.02
    level_return2zero = 0.10
    reference_percent = -0.10
    M = 0.4 * len(metric_inputs[0])
    neglim = 0.20
    negloss = 0.2
    C = metric_inputs[1]
    met_f = None
    eps = tf.convert_to_tensor(0.01)
    eps = tf.dtypes.cast(eps, tf.float64)
    full_alloc = None
    athmax = 0.1
    ra = metric_inputs[2][0]
    rb = metric_inputs[2][1]
    for ind, array in enumerate(metric_l):
        R = metric_inputs[0][ind] / 10
        K = 3 * R
        ath = R / ((level_return2zero - overprov_start))
        error = tf.subtract(array[:, 1], array[:, 0])
        cost = error
        z = (ra * error + rb)
        cost = (K * (tf.tanh(z) + 1) / 2) - R
        cost = tf.where(array[:, 1] > array[:, 0] + overprov_start,  ath * array[:, 1] -
                        R * (array[:, 0] + level_return2zero) / (level_return2zero - overprov_start), cost)
        cost = tf.where(array[:, 1] < 0, cost - cost, cost)
        cost = tf.where(array[:, 1] < -neglim, cost - cost + negloss, cost)
        if met_f is None:
            full_alloc = tf.clip_by_value(
                array[:, 1] * metric_inputs[0][ind], clip_value_min=0, clip_value_max=1000)
            met_f = cost
        else:
            full_alloc += tf.clip_by_value(array[:, 1] * metric_inputs[0][ind],
                                           clip_value_min=0, clip_value_max=1000)
            met_f += cost
    met_f = tf.where(full_alloc > C,  met_f - met_f + M, met_f)
    return met_f
