from general_utils import error
import matplotlib.pyplot as plt
import numpy as np

def plot_contour(trajectory, label='Gradient Descent'):
    """
    Draw the fixed contour of general_utils.error(w,b) over [-4,4]×[-4,4]
    and overlay the given trajectory.

    Parameters
    ----------
    trajectory : array‑like of shape (n_steps, 2)
        Sequence of (w,b) points to plot.
    label : str, default='Gradient Descent'
        Legend label for the trajectory line.
    """
    # build fixed grid
    w_vals = np.linspace(-4, 4, 100)
    b_vals = np.linspace(-4, 4, 100)
    W, B = np.meshgrid(w_vals, b_vals)
    
    # compute error surface
    Z = np.vectorize(error)(W, B)
    
    # plot contour
    plt.figure(figsize=(8, 6))
    cf = plt.contourf(W, B, Z, levels=100, cmap='coolwarm')
    cbar = plt.colorbar(cf,orientation='horizontal')
    cbar.set_label("Error Value")
    
    # overlay trajectory
    traj = np.asarray(trajectory)
    plt.plot(traj[:, 0], traj[:, 1], 'ro-', markersize=4, alpha=0.7, linewidth=1.2, label="Trajectory")
    plt.scatter(traj[0, 0], traj[0, 1], color='blue', s=60, edgecolors='black', label='Start')
    plt.scatter(traj[-1, 0], traj[-1, 1], color='green', s=60, edgecolors='black', label='End')
    
    # labels and legend
    plt.xlabel('w', fontsize=12, color='red')
    plt.ylabel('b', fontsize=12, color='red')
    plt.title(f'{label} Convergence Trajectory', fontsize=14)
    plt.legend()
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.tight_layout()
    plt.show()

def plot_all_trajectories_one_by_one(trajectories_dict):
    """
    Plot the error contour and overlay trajectories from multiple optimizers.

    Parameters
    ----------
    trajectories_dict : dict
        A dictionary where keys are optimizer labels (str) and values are
        trajectory arrays of shape (n_steps, 2).
    """
    # Build fixed grid
    w_vals = np.linspace(-4, 4, 100)
    b_vals = np.linspace(-4, 4, 100)
    W, B = np.meshgrid(w_vals, b_vals)
    
    # Compute error surface
    Z = np.vectorize(error)(W, B)

    # Set up plot
    plt.figure(figsize=(10, 8))
    cf = plt.contourf(W, B, Z, levels=100, cmap='coolwarm')
    cbar = plt.colorbar(cf, orientation='horizontal')
    cbar.set_label("Error Value")

    # Define color cycle
    colors = plt.cm.get_cmap('tab10', len(trajectories_dict))

    # Plot each trajectory
    for i, (label, trajectory) in enumerate(trajectories_dict.items()):
        traj = np.asarray(trajectory)
        plt.plot(traj[:, 0], traj[:, 1], marker='o', markersize=3,
                 alpha=0.7, linewidth=1.5, label=label,
                 color=colors(i))
        plt.scatter(traj[0, 0], traj[0, 1], color=colors(i), s=50, edgecolors='black', marker='s')  # Start
        plt.scatter(traj[-1, 0], traj[-1, 1], color=colors(i), s=50, edgecolors='black', marker='X')  # End

    # Labels and legend
    plt.xlabel('w', fontsize=12, color='red')
    plt.ylabel('b', fontsize=12, color='red')
    plt.title('Optimizer Trajectories Comparison', fontsize=14)
    plt.legend()
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.tight_layout()
    plt.show()

def plot_all_trajectories(trajectories):
    """
    Draw the error contour once, then overlay multiple optimizer trajectories.

    Parameters
    ----------
    trajectories : dict
        keys   → label for each optimizer (str)
        values → array‑like of shape (n_steps, 2) giving the (w,b) points
    """
    # 1) build fixed error‑surface grid
    w_vals = np.linspace(-4, 4, 100)
    b_vals = np.linspace(-4, 4, 100)
    W, B = np.meshgrid(w_vals, b_vals)
    Z = np.vectorize(error)(W, B)

    # 2) plot contour
    plt.figure(figsize=(10, 8))
    cf = plt.contourf(W, B, Z, levels=100, cmap='coolwarm')
    cbar = plt.colorbar(cf, orientation='horizontal')
    cbar.set_label("Error Value")

    # 3) overlay each trajectory in its own color
    #    plt.rcParams['axes.prop_cycle'] defines a default color cycle
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for idx, (label, traj) in enumerate(trajectories.items()):
        traj = np.asarray(traj)
        color = color_cycle[idx % len(color_cycle)]
        # path
        plt.plot(traj[:,0], traj[:,1],
                 marker='o', markersize=3, linewidth=1.5,
                 alpha=0.8, label=label, color=color)
        # start / end markers
        plt.scatter(traj[0,0], traj[0,1], marker='s', s=50,
                    edgecolors='black', color=color)
        plt.scatter(traj[-1,0], traj[-1,1], marker='X', s=50,
                    edgecolors='black', color=color)

    # 4) labels, legend, limits
    plt.xlabel('w', fontsize=12, color='red')
    plt.ylabel('b', fontsize=12, color='red')
    plt.title('Comparison of Optimizer Convergence Trajectories', fontsize=14)
    plt.legend(loc='lower left', fontsize=10)
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.tight_layout()
    plt.show()

