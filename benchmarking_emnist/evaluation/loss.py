from matplotlib.pylab import plt
import numpy as np

from evaluations_utils import create_directory_if_not_exists, round_to_nearest_multiple


def plot_loss(model_name: str, loss_per_epoch: list, figure_evaluation_dir):
    """
    Plots the loss per epoch for a given model.

    Args:
        model_name (str): The name of the model.
        loss_per_epoch (list): List of loss values for each epoch.
        figure_evaluation_dir (str): Directory where the figure will be saved.

    Returns:
        None

    This function plots the loss values per epoch and saves the figure.
    The figure is saved in the specified directory with the format '<model_name>_loss.png'.

    Example Usage:
        >>> plot_loss("MyModel", [0.1, 0.05, 0.02, 0.01], "/path/to/save")

    """
    # Create dir if it doesn't exist
    create_directory_if_not_exists(figure_evaluation_dir)

    # Calculate optimal x-ticks
    rounded_max_x = round_to_nearest_multiple(len(loss_per_epoch))
    tick_interval = int(rounded_max_x / 5)
    if tick_interval == 0:
        tick_interval = 1

    if rounded_max_x < len(loss_per_epoch):
        stop_x_tick = rounded_max_x + tick_interval + 1
    else:
        stop_x_tick = rounded_max_x + 1

    epochs = range(1, len(loss_per_epoch) + 1)

    # Update fig size
    plt.figure(figsize=(12, 6), dpi=100)

    # Add data
    plt.plot(epochs, loss_per_epoch, label="Loss after each epoch")

    # Add information
    plt.title(f"Loss per Epoch for {model_name} model")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(np.arange(0, stop_x_tick, tick_interval))
    plt.legend(loc="best")

    # Save the plot
    plot_path = f"{figure_evaluation_dir}/{model_name}_loss.png"
    plt.savefig(plot_path)
    plt.clf()


num_epochs = 100
loss_values = np.logspace(0, -3, num=num_epochs) + np.random.normal(0, 0.05, num_epochs)

plot_loss(
    "GRNN",
    loss_values,
    "/Users/simon/Documents/neural_networks_ocr_project/neural_networks_seminar/benchmarking_emnist/models/crnn",
)
