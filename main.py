import numpy as np
import matplotlib.pyplot as plt
import random
from settings import settings

import logging                  # package for writing log file
from utilities import utils     # used to define



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
    plt.xlabel(r'step number $n$')
    plt.ylabel(r'position $X_n$')
    plt.legend()

def do_histogram(X_n_arr):
    logging.info(f"Plotting histogram with bin size of {int(np.sqrt(len(X_n_arr)))}")
    histogram = plt.hist(X_n_arr, bins=int(35), density=True, color='skyblue', edgecolor='black')
    # plt.title("Histogram for X_n with n=N=1000")
    plt.xlabel(r'$X_N$')
    plt.ylabel(r'probability $p(X_N)$')
    # plt.legend()
    # plt.show()
    return histogram



def main():
    # init stuff
    out_dir = utils.create_output_directory()
    utils.setup_logging(out_dir)
    # utils.setup_logging(f'Created output directory: {out_dir}')
    plots_dir = utils.create_plots_directory()

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
    plt.savefig(plots_dir / f"part_1a.pdf")
    # plt.show()
    plt.clf()   # clear figure for next plot


    # part 1b)
    logging.info("\n\nPart 1b)\n")
    n_plus, n_minus = np.zeros(10), np.zeros(10)

    for i in range(10):
        n_plus[i] = (steps_arrays[i] == 1).sum()
        n_minus[i] = (steps_arrays[i] == -1).sum()

    X_n_arr = [x_trajectories[i][-1] for i in range(10)]  # taking the n-the (last) value of the first 10 walkers
    X_mean = np.mean(X_n_arr)
    X_var = np.var(X_n_arr)

    logging.info(f'n_plus = {n_plus}; n_minus = {n_minus}')
    logging.info(f'X_n_arr = {X_n_arr}')
    logging.info(f'<X> = {X_mean:.1f}')
    logging.info(f'Var(X_n) = {X_var:.1f}\n\n')
    

    # part 1c)
    logging.info("\n\nPart 1c)\n")
    X_M5000_arr = [x_trajectories[i][-1] for i in range(cfg_part_a.M)] 
    X_M5000_var_of_mean = np.var(X_M5000_arr) / cfg_part_a.M
    X_M5000_std_of_mean = np.sqrt(X_M5000_var_of_mean)

    logging.info(f'Var(X_M5000) = {X_M5000_var_of_mean:.1f}')
    logging.info(f'Std(X_M5000) = {X_M5000_std_of_mean:.1f}')

    # part 1d)
    logging.info("\n\nPart 1d)\n")
    do_histogram(X_M5000_arr)
    plt.savefig(plots_dir / f"part_1d.pdf")
    







if __name__ == '__main__':
    main()