import argparse


def batch_size_arg(value):
    if value == 'auto':
        return value
    try:
        size = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("batch size must be a positive integer or 'auto'")
    if size < 1:
        raise argparse.ArgumentTypeError("batch size must be a positive integer or 'auto'")
    return size
