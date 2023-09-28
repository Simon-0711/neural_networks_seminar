from matplotlib.pylab import plt
import numpy as np

from .evaluations_utils import create_directory_if_not_exists, round_to_nearest_multiple


def plot_loss(
    model_name: str,
    train_loss_per_epoch: list,
    val_loss_per_epoch: list,
    figure_evaluation_dir=None,
):
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

    if figure_evaluation_dir is None:
        figure_evaluation_dir = (
            f"neural_networks_seminar/benchmarking_emnist/models/{model_name}/"
        )
    # Create dir if it doesn't exist
    create_directory_if_not_exists(figure_evaluation_dir)

    # Calculate optimal x-ticks
    rounded_max_x = round_to_nearest_multiple(len(train_loss_per_epoch))
    tick_interval = int(rounded_max_x / 5)
    if tick_interval == 0:
        tick_interval = 1

    if rounded_max_x < len(train_loss_per_epoch):
        stop_x_tick = rounded_max_x + tick_interval + 1
    else:
        stop_x_tick = rounded_max_x + 1

    epochs = range(1, len(train_loss_per_epoch) + 1)

    # Update fig size
    plt.figure(figsize=(12, 6), dpi=100)

    # Add data
    plt.plot(epochs, train_loss_per_epoch, label="Train Loss")
    plt.plot(epochs, val_loss_per_epoch, label="Validation Loss")

    # Add information
    plt.title(f"Training and Validation Loss after Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(np.arange(0, stop_x_tick, tick_interval))
    plt.legend(loc="best")

    # Save the plot
    plot_path = f"{figure_evaluation_dir}{model_name}_loss.png"
    plt.savefig(plot_path)
    plt.clf()


def plot_multiple_losses(
    model_names: list,
    train_losses_list: list,
    val_losses_list: list,
    figure_evaluation_dir=None,
):
    """
    Plots the training and validation loss per epoch for multiple models.

    Args:
        model_names (list): List of model names.
        train_losses_list (list): List of lists containing training loss values for each epoch.
        val_losses_list (list): List of lists containing validation loss values for each epoch.
        figure_evaluation_dir (str): Directory where the figures will be saved.

    Returns:
        None

    This function generates separate plots for training and validation losses for each model.
    The figures are saved in the specified directory with the format '<model_name>_loss.png'.

    Example Usage:
        >>> plot_multiple_losses(
                ["Model1", "Model2"],
                [[0.1, 0.05, 0.02, 0.01], [0.12, 0.08, 0.05, 0.03]],
                [[0.2, 0.15, 0.1, 0.08], [0.22, 0.18, 0.14, 0.12]],
                "/path/to/save"
            )

    """

    if figure_evaluation_dir is None:
        figure_evaluation_dir = (
            f"neural_networks_seminar/benchmarking_emnist/models/{model_name}/"
        )
    # Create dir if it doesn't exist
    create_directory_if_not_exists(figure_evaluation_dir)

    # Calculate optimal x-ticks
    rounded_max_x = round_to_nearest_multiple(len(train_losses_list[0]))
    tick_interval = int(rounded_max_x / 5)
    if tick_interval == 0:
        tick_interval = 1

    if rounded_max_x < len(train_losses_list[0]):
        stop_x_tick = rounded_max_x + tick_interval + 1
    else:
        stop_x_tick = rounded_max_x + 1

    epochs = range(1, len(train_losses_list[0]) + 1)

    # Update fig size
    plt.figure(figsize=(12, 6), dpi=100)

    for model_name, train_loss_per_epoch, val_loss_per_epoch in zip(
        model_names, train_losses_list, val_losses_list
    ):
        # Add data
        plt.plot(
            epochs,
            train_loss_per_epoch,
            label=f"Model number {model_name}",
        )

    # Add information
    plt.title(f"Training Loss after Epoch for Top 5 Grid Search Models")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(np.arange(0, stop_x_tick, tick_interval))
    plt.legend(loc="best")

    # Save the plot
    plot_path = f"{figure_evaluation_dir}top_5_models_training_losses.png"
    plt.savefig(plot_path)
    plt.clf()

    # Update fig size
    plt.figure(figsize=(12, 6), dpi=100)

    for model_name, train_loss_per_epoch, val_loss_per_epoch in zip(
        model_names, train_losses_list, val_losses_list
    ):
        # Add data
        plt.plot(
            epochs,
            val_loss_per_epoch,
            label=f"Model number {model_name}",
        )

    # Add information
    plt.title(f"Validation Loss after Epoch for Top 5 Grid Search Models")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(np.arange(0, stop_x_tick, tick_interval))
    plt.legend(loc="best")

    # Save the plot
    plot_path = f"{figure_evaluation_dir}top_5_models_validation losses.png"
    plt.savefig(plot_path)
    plt.clf()
