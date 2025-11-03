import numpy as np
import matplotlib.pyplot as plt
import random
from math import factorial
from scipy.stats import poisson
from settings import settings
import numba

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


def generate_stat_walker(cfg_Ex2):
    X_arr = np.zeros(cfg_Ex2.N)
    X_arr[0] = cfg_Ex2.X_0

    for n in range(1, cfg_Ex2.N):
        r_n = random.random() - 0.5
        X_arr[n] = X_arr[n-1] + r_n

    return X_arr


def plot_traj(cfg, all_trajs, first_walkers=5):
    t = np.arange(cfg.N)
    for i in range(first_walkers):
        plt.plot(t, all_trajs[i], label=f'walker {i}')
    plt.xlabel(r'step number $n$')
    plt.ylabel(r'position $X_n$')
    plt.legend()

def do_histogram(X_n_arr):
    # numb_of_bins = int(np.sqrt(len(X_n_arr)))
    # numb_of_bins = int(45)
    min_val = int(np.min(X_n_arr)) - 1
    max_val = int(np.max(X_n_arr)) + 3    
    bins = np.arange(min_val, max_val, 4) 

    logging.info(f"Plotting histogram with bins of {len(bins)}")
    histogram = plt.hist(X_n_arr, bins=bins, density=True, color='skyblue', edgecolor='black')
    plt.xlabel(r'$X_n$')
    plt.ylabel(r'probability density $f(X_n)$')

    return histogram

def gauss(x_arr, mu, var):
    return 1/np.sqrt(2*np.pi*var) * np.exp(- (x_arr - mu)**2/ (2*var))



def main(run_task="all"):
    # run_task kann "all", "1", "2", ... sein

    # init stuff
    out_dir = utils.create_output_directory()
    utils.setup_logging(out_dir)
    # utils.setup_logging(f'Created output directory: {out_dir}')
    plots_dir = utils.create_plots_directory()


    if run_task in ("all", "1"):
        # configuration
        cfg_part_a = settings.Config()

        # part 1)a)
        # default instance is the configuration of the variable settings of exercise 1a)
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
        hist_d = do_histogram(X_M5000_arr)
        
        Mean_theo =  cfg_part_a.N * (cfg_part_a.p * 1 - cfg_part_a.q )
        Var_theo = cfg_part_a.N * (4*cfg_part_a.p * cfg_part_a.q)         # 
        logging.info(f"True theoretiacal mean is: mu = E[X_n]= {Mean_theo:.2f}")
        logging.info(f"True theoretiacal variance is: var = {Var_theo:.2f}\n")

        X_arr_hist = np.arange(-3*np.sqrt(Var_theo) + Mean_theo, +3*np.sqrt(Var_theo) + Mean_theo)
        plt.plot(X_arr_hist, gauss(X_arr_hist, Mean_theo, Var_theo), color="orange", label="Theoretical Gaussian")
        
        plt.legend()
        plt.savefig(plots_dir / f"part_1d.pdf")
        plt.clf()   # clear figure for next plot

        # part 1e)
        logging.info("\n\nPart 1e)\n")
        # Config
        p = 0.005
        N = 1000
        M = 5000
        cfg_part_e = settings.Config(p=p, N=N, M=M)

        x_trajectories = np.zeros((cfg_part_e.M, cfg_part_e.N))
        
        # simulate walkers
        x_trajectories = np.zeros((M, N))
        for i in range(M):
            x_trajectories[i], _ = generate_walker(cfg_part_e)

        # Endpositionen aller Walker
        X_N_arr = x_trajectories[:, -1]

        # Theoretischer Poisson-Parameter für rechte Schritte
        lambda_theo = N * p

        # Bins für Histogramm (X_N nimmt nur gerade Werte an)
        min_val = int(np.min(X_N_arr)) - 1
        max_val = int(np.max(X_N_arr)) + 3
        bins = np.arange(min_val, max_val, 2)

        # Histogramm der Simulation (normalisierte Wahrscheinlichkeiten)
        plt.hist(
            X_N_arr,
            bins=bins,
            density=False,
            weights=np.ones_like(X_N_arr) / len(X_N_arr),
            label="Simulation $P(X_N)$",
            align='left',
            rwidth=0.8,
            color='skyblue',
            edgecolor='black'
        )

        # X-Werte für theoretische Poisson-PMF (nur ganze Schritte)
        X_values = bins[:-1]

        # Anzahl der rechten Schritte n_R
        n_R = (X_values + N) // 2  # Integer-Division
        P_poisson = poisson.pmf(n_R, mu=lambda_theo)

        # Plot Poisson
        plt.plot(X_values, P_poisson, 'o', color="red", label=f"Theorie Poisson (lambda={lambda_theo})")

        plt.xlabel("$X_N$")
        plt.ylabel("Probability $P(X_N)$")
        plt.legend()
        plt.savefig(plots_dir / f"part_1e1.pdf")
        plt.clf()   # clear figure for next plot



        #part 1e) - 2nd part
        p = 0.005
        N = 100
        M = 5000
        cfg_part_e_2 = settings.Config(p=p, N=N, M=M)


        x_trajectories = np.zeros((cfg_part_e_2.M, cfg_part_e_2.N))
        
        # simulate walkers
        x_trajectories = np.zeros((M, N))
        for i in range(M):
            x_trajectories[i], _ = generate_walker(cfg_part_e_2)

        # Endpositionen aller Walker
        X_N_arr = x_trajectories[:, -1]

        # Theoretischer Poisson-Parameter für rechte Schritte
        lambda_theo = N * p

        # Bins für Histogramm (X_N nimmt nur gerade Werte an)
        min_val = int(np.min(X_N_arr)) - 1
        max_val = int(np.max(X_N_arr)) + 3
        bins = np.arange(min_val, max_val, 2)

        # Histogramm der Simulation (normalisierte Wahrscheinlichkeiten)
        plt.hist(
            X_N_arr,
            bins=bins,
            density=False,
            weights=np.ones_like(X_N_arr) / len(X_N_arr),
            label="Simulation $P(X_N)$",
            align='left',
            rwidth=0.8,
            color='skyblue',
            edgecolor='black'
        )

        # X-Werte für theoretische Poisson-PMF (nur ganze Schritte)
        X_values = bins[:-1]

        # Anzahl der rechten Schritte n_R
        n_R = (X_values + N) // 2  # Integer-Division
        P_poisson = poisson.pmf(n_R, mu=lambda_theo)

        # Plot Poisson
        plt.plot(X_values, P_poisson, 'o', color="red", label=f"Theorie Poisson (lambda={lambda_theo})")

        plt.xlabel("$X_N$")
        plt.ylabel("Probability $P(X_N)$")
        plt.legend()
        plt.savefig(plots_dir / f"part_1e2.pdf")
        plt.clf()   # clear figure for next plot



##############    # Exercise 2      #################
    if run_task in ("all", "2"):
        
        logging.info("\nStarting exercise 2")
    
        # part (1)
        
        logging.info("\nPart (1)\n\n")
        # cfg_2_1 = settings.Config_Ex2(M=500) # has default values of k=0.1, N=1000, M=500 000
        cfg_2_1 = settings.Config_Ex2() # has default values of k=0.1, N=1000, M=500 000


        x_trajectories = np.zeros((cfg_2_1.M, cfg_2_1.N))
        for m in range(cfg_2_1.M):
            x_trajectories[m] = generate_stat_walker(cfg_2_1)

        means_X_n = np.mean(x_trajectories, axis=0)
        vars_X_n = np.var(x_trajectories, axis=0)

        logging.info(means_X_n[-20:-1])

        plt.figure("2_1_expectation_val")
        plt.plot(np.arange(cfg_2_1.N), means_X_n, '.', label="Mean random walk positions\nof M=500 000 walkers for\nall n=1,2,...,1000 steps")
        plt.xlabel(r"$n$")
        plt.ylabel(r"$\langle X_n \rangle$")
        plt.legend()
        plt.savefig(plots_dir / f"Ex_2_1_expectation_val.pdf")

        plt.figure("2_1_var")
        plt.plot(np.arange(cfg_2_1.N), vars_X_n, '.', label="Variance of random walk positions\nof M=500 000 walkers for\nall n=1,2,...,1000 steps")
        plt.xlabel(r"$n$")
        plt.ylabel(r"$\mathrm{Var}[X_n]$")
        plt.legend()
        plt.savefig(plots_dir / f"Ex_2_1_var.pdf")


        # part (2)
        logging.info("\nPart (2)\n\n")

        n_arr = [200, 500, 1000]

        x_traj_selected = [x_trajectories[:,n-1] for n in n_arr]  # -1 because we start at 0

        # logging.info(np.shape(x_traj_selected))

        # bins for histograms 
        bins = np.linspace(-40, 40, 81)  # or based on the largest variance (n=1000)

        # Histogramm der Simulation (normalisierte Wahrscheinlichkeiten)
        plt.figure("2_2_n200")
        plt.xlabel(r"$X_{200}$")
        plt.ylabel(r"PDF $P(X_{200})$")
        plt.hist(
            x_traj_selected[0],
            bins=bins,
            density=True,
            align='left',
            rwidth=0.8,
            color='blue',
            edgecolor='black',
            label = "n=200"
        )
        plt.legend()
        # plt.savefig(plots_dir / f"Ex_2_2_n200.pdf")
        # plt.show()

        plt.xlabel(r"$X_{500}$")
        plt.ylabel(r"PDF $P(X_{500})$")
        plt.hist(
            x_traj_selected[1],
            bins=bins,
            density=True,
            align='left',
            rwidth=0.8,
            color='red',
            edgecolor='black',
            label = "n=500"
        )
        # plt.show()
        plt.legend()
        # plt.savefig(plots_dir / f"Ex_2_2_n500.pdf")

        plt.xlabel(r"$X_{1000}$")
        plt.ylabel(r"PDF $P(X_{1000})$")
        plt.hist(
            x_traj_selected[2],
            bins=bins,
            density=True,
            align='left',
            rwidth=0.8,
            color='green',
            edgecolor='black',
            label = "n=1000"
        )
        # plt.show()        
        plt.legend()
        plt.savefig(plots_dir / f"Ex_2_2_all.pdf")
        


if __name__ == '__main__':
    main(run_task="all")