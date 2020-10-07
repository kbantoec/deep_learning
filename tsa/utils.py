import matplotlib.pyplot as plt


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
