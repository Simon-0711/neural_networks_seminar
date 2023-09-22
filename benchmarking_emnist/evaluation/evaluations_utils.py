import os
import math


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def round_to_nearest_multiple(number):
    """
    Round a number to the nearest multiple of 10, 100, 1000, etc.

    Args:
        number (int): The number to be rounded.

    Returns:
        int: The rounded number.

    If the input number is less than 10, it remains unchanged.

    Examples:
        >>> round_to_nearest_multiple(12)
        10
        >>> round_to_nearest_multiple(235)
        240
        >>> round_to_nearest_multiple(2348)
        2400
        >>> round_to_nearest_multiple(27)
        30
        >>> round_to_nearest_multiple(2735)
        2700
        >>> round_to_nearest_multiple(9859)
        9900
    """
    if number < 10:
        return number  # Numbers less than 10 remain unchanged

    order_of_magnitude = int((10 ** (int(math.log10(number)))) / 10)
    rounded_number = round(number / order_of_magnitude) * order_of_magnitude
    return rounded_number
