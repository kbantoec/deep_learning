import matplotlib.pyplot as plt
import numpy as np


def blue_red_plot(x, y, ylabel: str, save: bool = False, imgname: str = None):
    plt.figure(figsize=(6, 4))
    plt.plot(x, label='Training', c='b')
    plt.plot(y, label='Validation', c='r')
    plt.ylabel(ylabel)
    plt.xlabel('Iteration')
    plt.grid(c='k', ls=':')
    plt.legend()
    plt.tight_layout()
    if save:
        if imgname is not None:
            plt.savefig(f'out/plots/{imgname}.png', dpi=600, transparent=True)
        else:
            print("Please, pass an image name as argument to the 'imgname' parameter.")
    plt.show()


def plot_accuracy(train_acc, validation_acc, save: bool = False, imgname: str = None):
    blue_red_plot(train_acc, validation_acc, 'Accuracy', save, imgname)


def plot_loss(train_loss, validation_loss, save: bool = False, imgname: str = None):
    blue_red_plot(train_loss, validation_loss, 'Loss', save, imgname)


def encode_to_bits(x) -> int:
    return int(np.sign(x)) if x > 0 else int(0)


def get_label(x: np.ndarray, i1: int, i2: int, i3: int) -> int:
    """Creates the label applied to three consecutive values by passing them
    to a XOR function. The labels are either 1 or 0, meaning this is used
    for binary classification.

    Parameters
    ----------
    x : numpy.ndarray
        Time series sequence.

    i1 : int
        Index number 1.

    i2 : int
        Index number 2.

    i3 : int
        Index number 3.

    Returns
    -------
    int
        Either 1 or 0.
    """
    x1: int = encode_to_bits(x[i1])
    x2: int = encode_to_bits(x[i2])
    x3: int = encode_to_bits(x[i3])
    xor: int = x1 ^ x2 ^ x3
    return 0 if xor == 1 else 1


def generate_distance(T: int = 10, short: bool = True) -> tuple:
    """Generate random data points (X) from a standard normal distribution
     and their respective target labels (Y) according to three consecutive
     data points passed in a XOR function.

    Parameters
    ----------
    T : int
       Time steps of a sequence.

    short : bool
       Whether we want to build a short-distance dataset or not. If not, then it
       will generate a long-distance dataset. By short- or long-distance, we mean
       a pattern that is located at the end or the start of a time series sequence,
       respectively.

    Returns
    -------
    tuple
        A `tuple` of numpy NDarrays for `X` and `Y`, and `int` for `n`. Where `n`
        is the number of samples (of time series sequences) generated.
    """
    arr = np.array([-4, 10, 5])
    assert get_label(arr, 0, 1, 2) == 1, "There is a problem with 'get_label' function."

    x_store: list = []
    y_store: list = []

    for t in range(5000):
        x: np.ndarray = np.random.randn(T)
        x_store.append(x)
        if short:
            y = get_label(x, -1, -2, -3)
        else:
            y = get_label(x, 0, 1, 2)
        y_store.append(y)

    n: int = len(x_store)
    return np.array(x_store), np.array(y_store), n
