import numpy as np
import matplotlib.pyplot as plt
import random
from settings import settings


def generate_walker(cfg):
    x_traj = np.zeros(cfg.N)
    steps_arr = np.zeros(cfg.N) # array of steps with values of -1 or +1 (left and right)

    for i in range(1, cfg.N):
        if random.random() > cfg.p:
            x_traj[i] = x_traj[i-1] - cfg.Delta_x
            steps_arr[i] = -1
        else:
            x_traj[i] = x_traj[i-1] + cfg.Delta_x
            steps_arr[i] = 1

    return x_traj, steps_arr


def plot_traj(cfg, all_trajs, first_walkers=5):
    t = np.arange(cfg.N)
    for i in range(first_walkers):
        plt.plot(t, all_trajs[i], label=f'walker {i}')
    plt.xlabel(r'number of step $n$')
    plt.ylabel(r'position $X_n$')
    plt.legend()


def main():
    # part 1)a)
    # default instance is the configuration of the variable settings of exercise 1a)
    cfg_part_a = settings.Config()
    # bc we used @dataclass we can now redefine easily for example q and p like this:
    # cfg_part_a = settings.Config(q=0.1, p=0.9)

    x_trajectories = np.zeros((cfg_part_a.M, cfg_part_a.N))
    steps_arrays =  np.zeros((cfg_part_a.M, cfg_part_a.N))

    for i in range(cfg_part_a.M):
        x_trajectories[i], steps_arrays[i] = generate_walker(cfg_part_a)
    plot_traj(cfg_part_a, x_trajectories, first_walkers=10)
    plt.savefig(cfg_part_a.out_dir / f"part_1a.pdf")
    # plt.show()
    plt.clf()   # clear figure for next plot

    # part b)<
    n_plus, n_minus = np.zeros(10), np.zeros(10)
    X_mean = 0

    for i in range(10):
        n_plus[i] = (steps_arrays[i] == 1).sum()
        n_minus[i] = (steps_arrays[i] == -1).sum()

    X_n_arr = [x_trajectories[i][-1] for i in range(10)]  # taking the n-the (last) value of the first 10 walkers
    X_mean = np.mean(X_n_arr)
    X_var = np.var(X_n_arr)

    print(f'n_plus = {n_plus}; n_minus = {n_minus}')
    print(f'X_n_arr = {X_n_arr}')
    print(f'<X> = {X_mean:.1f}')
    print(f'Var(X_n) = {X_var:.1f}')
    


if __name__ == '__main__':
    main()