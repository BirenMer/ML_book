import numpy as np
import matplotlib.pyplot as plt

def plot_error_contour(error_function,trajectory,algo_name):
    """Function for creating a contour plot and trajectory of the optimization algorithm."""
    # Note Currently generating static data to maintain similarity
    # Generate grid data for w and b
    w_vals = np.linspace(-4, 4, 100)
    b_vals = np.linspace(-4, 4, 100)

    W, B = np.meshgrid(w_vals, b_vals)

    # Compute the error at each point in the grid
    Z = np.array([[error_function(w, b) for w in w_vals] for b in b_vals])

    w_traj, b_traj = zip(*trajectory)

    # Plot the contour plot
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(W, B, Z, levels=30, cmap="coolwarm")
    plt.colorbar(contour, label="Error Value")
    
    # Plot gradient descent trajectory
    plt.plot(w_traj, b_traj, 'o-', color="black", markersize=4, label="Gradient Descent Path")

    plt.xlabel("w", fontsize=14, color="red")
    plt.ylabel("b", fontsize=14, color="red")
    plt.title(f"Error Contour Plot with {algo_name} Path", fontsize=14)
    plt.legend()
    plt.show()