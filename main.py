import numpy as np
from matplotlib import plt
from random import random
from settings import settings


def generate_walker():
    x_traj = np.zeros(settings.N)

    for i in range(1, settings.N):
        if random.random() > settings.p:
            x_traj[i] = x_traj[i-1] - settings.Delta_x
        else:
            x_traj[i] = x_traj[i-1] + settings.Delta_x

    return x_traj


def plot_traj(x_traj, first_steps):
    plt.plot(settings.N[:first_steps], x_traj[:first_steps])