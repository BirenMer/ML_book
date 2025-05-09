from general_utils import error
import matplotlib.pyplot as plt
import numpy as np

def plot_contour(trajectory, label='Gradient Descent Path'):
    """
    Draw the fixed contour of general_utils.error(w,b) over [-4,4]×[-4,4]
    and overlay the given trajectory.

    Parameters
    ----------
    trajectory : array‑like of shape (n_steps, 2)
        Sequence of (w,b) points to plot.
    label : str, default='Gradient Descent Path'
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
    plt.plot(traj[:, 0], traj[:, 1], 'ro-', markersize=4, alpha=0.7, linewidth=1.2, label=label)
    plt.scatter(traj[0, 0], traj[0, 1], color='blue', s=60, edgecolors='black', label='Start')
    plt.scatter(traj[-1, 0], traj[-1, 1], color='green', s=60, edgecolors='black', label='End')
    
    # labels and legend
    plt.xlabel('w', fontsize=12, color='red')
    plt.ylabel('b', fontsize=12, color='red')
    plt.title('Error Contour with Trajectory', fontsize=14)
    plt.legend()
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.tight_layout()
    plt.show()
