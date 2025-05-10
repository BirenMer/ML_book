from gradient_descent import do_gragient_descent
from ada_grad import do_adagrad
from adam import do_adam
from line_search_gradient_descent import do_line_search_gradient_descent
from mini_batch_gradient_descent import do_mini_batch_gradient_descent
from momentum_based_gradient_descent import do_momentum_based_gradient_descent
from nesterov_accelerated_gardient_descent import do_nesterov_accelerated_gradient_descent
from rmsprop import do_rmsprop
from stochastic_gradient_descent import do_stochastic_gradient_descent
from stochastic_momentum_based_gradient_descent import do_stochastic_momentum_based_gradient_descent
from stochastic_nestrov_accelerated_gradient_descent import do_stochastic_nesterov_accelerated_gradient_descent
from trajectory_plotting_utils import plot_all_trajectories

import matplotlib.pyplot as plt

def main():

    gd_trajectory=do_gragient_descent()
    momentum_trajectory=do_momentum_based_gradient_descent()
    nesterov_trajectory=do_nesterov_accelerated_gradient_descent()
    sgd_trajectory=do_stochastic_gradient_descent()
    mini_batch_trajectory=do_mini_batch_gradient_descent()
    line_search_trajectory=do_line_search_gradient_descent()
    adagrad_trajectory=do_adagrad()
    rmsprop_trajectory=do_rmsprop()
    adam_trajectory=do_adam()
    stochastic_momentum_trajectory=do_stochastic_momentum_based_gradient_descent()
    stochastic_nesterov_trajectory=do_stochastic_nesterov_accelerated_gradient_descent()

    trajectories = {
        "Vanilla GD": gd_trajectory,
        "Momentum GD": momentum_trajectory,
        "Nesterov": nesterov_trajectory,
        "SGD": sgd_trajectory,
        "Mini-Batch GD": mini_batch_trajectory,
        "Line Search GD": line_search_trajectory,
        "Adagrad": adagrad_trajectory,
        "RMSprop": rmsprop_trajectory,
        "Adam": adam_trajectory,
        "Stochastic Momentum GD": stochastic_momentum_trajectory,
        "Stochastic Nesterov": stochastic_nesterov_trajectory,
    }

    # Ensure no other plots are shown before the combined plot
    plt.close('all')

    plot_all_trajectories(trajectories)

if __name__ == "__main__":
    main()
