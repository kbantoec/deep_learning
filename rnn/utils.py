from cycler import cycler
from itertools import cycle, product, chain, count
from pathlib import Path
from string import ascii_uppercase
import sys
import time
from typing import Union, Optional

import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, SimpleRNN, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

colors = {'green': '#59C899', 
          'blue': '#6370F1', 
          'orange': '#F3A467', 
          'purple': '#A16AF2', 
          'light_blue': '#5BCCC7', 
          'red': '#DF6046'}

cmap = cycle(colors.values())

mpl_cycler = cycler(color=colors.values())

target_pred_cycler = (cycler(color=['k', '#ff6e00']) + cycler(linestyle=[':', '-']))
loss_cycler = (cycler(color=['b', 'r']) + cycler(linestyle=['-', '-']))


def check_dir(mypath: Path):
    if not mypath.exists():
        mypath.mkdir()
        print(f"Directory created: {mypath.resolve()!r}")
    else:
        print(f"This directory exists: {mypath.resolve()!r}")


def print_nb_info():
    print(sys.version)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"TensorFlow GPU: {tf.config.list_physical_devices('GPU')}")


def print_elapsed_time(start_time: float):
    print(f"Elapsed time: {int(time.time() - start_time)} seconds")


class SineModel:
    np.random.seed(0)

    def __init__(self,
                 noise: bool = True,
                 activation: Union[str, None] = None,
                 T: int = 10,
                 fig_ext: Optional[str] = None,
                 fig_path: Optional[Path] = None):
        self.__noise: bool = noise
        self.series: np.ndarray = np.sin(0.1 * np.arange(200)) + np.random.randn(200) * 0.1 if noise else np.sin(0.1 * np.arange(200))
        self.activation: Optional[str] = activation
        self.T: int = T
        self.X, self.Y, self.N = self.__generate_dataset()
        self.history = None
        self.validation_target = None
        self.validation_predictions = None
        self.model = None
        self.__fig_ext: str = 'pdf' if fig_ext is None else fig_ext
        self.__fig_path: Path = Path('.') if fig_path is None else fig_path
        self.__legend_params: dict = dict(loc='center', bbox_to_anchor=(0.5, 1.1), edgecolor='none',
                                          fancybox=True, ncol=2, framealpha=1, facecolor='none')
    
    def __generate_dataset(self) -> tuple:
        T: int = self.T
        D: int = 1
        X_: list = []
        Y_: list = []

        for t in range(len(self.series) - T):
            x: np.ndarray = self.series[t:t + T]
            X_.append(x)
            y: np.ndarray = self.series[t + T]
            Y_.append(y)

        X: np.ndarray = np.array(X_).reshape(-1, T, 1)  # (N, T, D)
        Y: np.ndarray = np.array(Y_)
        N: int = len(X)
        return X, Y, N

    def plot_sine_wave(self, title: Optional[str] = None, save: bool = False):
        plt.rc('font', size=12)
        font = {'family': 'serif',
                'color': 'k',
                'weight': 'normal',
                'size': 18,
                }
        plt.figure(figsize=(10, 3))
        plt.plot(self.series, c='k', ls='-')
        plt.ylabel(r'$\sin(\omega t) + \epsilon_t \cdot \omega$' if self.__noise else r'$\sin(\omega t)$',
                   fontdict=font)
        plt.xlabel(r'$t$', fontdict=font)
        if title is not None:
            plt.title(title)
        plt.tight_layout()
        if save:
            imgname: str = 'sine_wave_noisy' if self.__noise else 'sine_wave'
            out = self.__fig_path / f'{imgname}.{self.__fig_ext}'
            plt.savefig(out, dpi=600, transparent=True)
            print(f"Saved plot in {out}")
        plt.show()
        plt.rc('font', size=14)

    # def fit_linreg(self,
    #         model,
    #         verbose: bool = True,
    #         epochs: int = 80):
    #     X, Y, N = self.X, self.Y, self.N
    #     r = model.fit(X[:-N // 2],
    #                   Y[:-N // 2],
    #                   epochs=epochs,
    #                   validation_data=(X[-N // 2:], Y[-N // 2:]),
    #                   verbose=verbose)
    #     self.model = model
    #     self.history = r
    #     return r

    def build_linreg(self):
        T = self.T
        i = Input(shape=(T,))
        x = Dense(1)(i)
        model = Model(i, x)
        return model

    def build_train_linreg(self,
                           verbose: bool = True,
                           epochs: int = 80,
                           loss='mse',
                           lr: float = 0.1):
        self.X = np.squeeze(self.X, axis=2)
        X, Y, N = self.X, self.Y, self.N
        model = self.build_linreg()
        model.compile(loss=loss, optimizer=Adam(lr=lr))
        r = model.fit(X[:-N // 2], Y[:-N // 2],
                      epochs=epochs,
                      validation_data=(X[-N // 2:], Y[-N // 2:]),
                      verbose=verbose)
        self.model = model
        self.history = r
        return r
    
    def build_train_simple_rnn(self,
                               verbose: bool = True,
                               epochs: int = 80,
                               loss='mse',
                               lr: float = 0.1,
                               units: int = 5):
        T, X, Y, N = self.T, self.X, self.Y, self.N
        inp = Input(shape=(T, 1))
        x = SimpleRNN(units=units, activation=self.activation)(inp)
        x = Dense(1)(x)
        model = Model(inp, x)
        model.compile(optimizer=Adam(learning_rate=lr), loss=loss)
        r = model.fit(X[:-N//2], Y[:-N//2], epochs=epochs, validation_data=(X[-N//2:], Y[-N//2:]), verbose=verbose)
        self.model = model
        self.history = r
        return r
    
    def plot_loss(self, verbose: bool = False, save: bool = False, imgname: str = None):
        r = self.history
        if r is None:
            print("No model to plot loss. So, created one on the fly.")
            r = self.build_train_simple_rnn(verbose)
        plt.rc('axes', prop_cycle=loss_cycler)
        plt.plot(r.history['loss'], label='Training')
        plt.plot(r.history['val_loss'], label='Validation')
        plt.legend(**self.__legend_params)
        plt.grid(c='k', ls=':')
        plt.ylabel('Loss')
        plt.xlabel('Iteration')
        plt.tight_layout()
        if save:
            if not isinstance(imgname, str):
                print("Pass an image name of type 'str'.")
            else:
                filename = self.__fig_path / imgname
                plt.savefig(f'{filename}.{self.__fig_ext}', dpi=600, transparent=True)
                print(f"Figure successfully saved in {filename}.{self.__fig_ext}")
        plt.show()
    
    def predict(self, is_wrong: bool = False) -> None:
        if self.model is not None:
            N = self.N
            X = self.X
            Y = self.Y
            validation_target = Y[-N//2:]
            validation_predictions = []

            if is_wrong:
                print("WRONG FORECAST (-> the forecast is subject to 'look-ahead bias')")
                i: int = -N//2

                while len(validation_predictions) < len(validation_target):
                    p = self.model.predict(X[i].reshape(1, -1, 1))[0, 0]  # Scalar    
                    validation_predictions.append(p)
                    i += 1
            else:
                print("RIGHT FORECAST (-> the forecast does not exhibit 'look-ahead bias')")
                last_x: np.ndarray = X[-N//2]  # (1, T)

                while len(validation_predictions) < len(validation_target):
                    p = self.model.predict(last_x.reshape(1, -1, 1))[0, 0]  # Scalar
                    validation_predictions.append(p)

                    last_x = np.roll(last_x, -1)
                    last_x[-1] = p

            self.validation_target = validation_target
            self.validation_predictions = validation_predictions
        else:
            print("You have to build a model first.")
    
    def plot_target_vs_prediction(self, save: bool = False, imgname: str = None):
        if (self.validation_predictions and self.validation_target) is not None:
            plt.rc('axes', prop_cycle=target_pred_cycler)
            plt.plot(self.validation_target, label='forecast target')
            plt.plot(self.validation_predictions, label='forecast prediction')
            plt.legend(**self.__legend_params)
            plt.tight_layout()
            if save:
                if not isinstance(imgname, str):
                    print("Pass an image name of type 'str'.")
                else:
                    plt.savefig(f'{self.__fig_path / imgname}.{self.__fig_ext}', dpi=600, transparent=True)
            plt.show()
        else:
            print("First run 'predict' method.")
            

def plot_combinations():
    args = {'noise': (True, False), 'activation': ('tanh', 'relu', None), 'is_wrong': (True, False)}
    comb = tuple(product(args['noise'], args['is_wrong']))
    # ((True, True), (True, False), (False, True), (False, False))
    f = plt.figure(figsize=(10, 35))
    it = iter(range(12))
    alpha = iter(ascii_uppercase)

    for i, fn in enumerate(('tanh', 'relu', None)):
        for tup in comb:
            n = next(it)
            print(f"Computing model {n + 1}/12")
            m = SineModel(noise=tup[0], activation=fn)
            m.build_train_simple_rnn(verbose=False)
            m.predict(is_wrong=tup[1])

            ax = f.add_subplot(12, 2, n + 1)
            ax.set_title(f"({next(alpha)}) (noise={tup[0]}, fn={fn}, is wrong? {tup[1]})")
            ax.plot(m.validation_target, label='forecast target')
            ax.plot(m.validation_predictions, label='forecast prediction')
            ax.legend()
    
    plt.tight_layout()
    plt.show()


def plot_n_samples(num_samples: int = 5,
                   noise: bool = False,
                   activation=None,
                   epochs: int = 80,
                   loss='mse',
                   learning_rate: float = 0.1,
                   units: int = 5,
                   save: bool = False,
                   extension: str = 'pdf',
                   shift_row_counter: int = 0,
                   fig_path: Optional[Path] = None):
    start: float = time.time()
    n = num_samples
    fig_path: Path = Path('.').resolve() if fig_path is None else fig_path
    sm = SineModel(noise=noise, activation=activation, fig_path=fig_path)
    legend_params = dict(loc='center', bbox_to_anchor=(0.5, 1.1), edgecolor='none',
                         fancybox=True, ncol=2, framealpha=1, facecolor='none')
    counter = count(1)
    alpha_counter = chain(ascii_uppercase)
    vspace = 4

    fig = plt.figure(figsize=(15, vspace * n))
    rcParams['axes.prop_cycle'] = target_pred_cycler
    rcParams['font.size'] = 12
    for i in range(1, n + 1):
        sm.build_train_simple_rnn(verbose=False, epochs=epochs, loss=loss, lr=learning_rate, units=units)

        # Loss per iteration
        ax1 = fig.add_subplot(n, 3, next(counter))
        ax1.plot(sm.history.history['loss'], label='Training', c='b', ls='-')
        ax1.plot(sm.history.history['val_loss'], label='Validation', c='r', ls='-')
        ax1.legend(**legend_params)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.grid(c='k', ls=':')
        ax1.annotate(f"({i + shift_row_counter})", xy=(-0.3, 0.5), annotation_clip=False, textcoords='axes fraction',
                     fontweight='bold')

        # Wrong forecast
        ax2 = fig.add_subplot(n, 3, next(counter))
        sm.predict(is_wrong=True)
        ax2.plot(sm.validation_target, label='forecast target')
        ax2.plot(sm.validation_predictions, label='forecast prediction')
        ax2.legend(**legend_params)

        # Right forecast
        ax3 = fig.add_subplot(n, 3, next(counter), sharey=ax2)
        sm.predict(is_wrong=False)
        ax3.plot(sm.validation_target, label='forecast target')
        ax3.plot(sm.validation_predictions, label='forecast prediction')
        ax3.legend(**legend_params)

        if i == 1 + shift_row_counter:  # if shift_row_counter > 0, it will not annotate
            ax1.annotate(f"({next(alpha_counter)})",
                         xy=(0.5, 1.25),
                         annotation_clip=False,
                         textcoords='axes fraction',
                         fontweight='bold')
            ax2.annotate(f"({next(alpha_counter)})",
                         xy=(0.5, 1.25),
                         annotation_clip=False,
                         textcoords='axes fraction',
                         fontweight='bold')
            ax3.annotate(f"({next(alpha_counter)})",
                         xy=(0.5, 1.25),
                         annotation_clip=False,
                         textcoords='axes fraction',
                         fontweight='bold')

    fig.tight_layout()
    if save:
        name: str = 'sine'
        name += '_noise' if noise else ''
        name += f"_{str(activation).lower()}"
        name += f"_{loss}"
        name += f"_lr{learning_rate:.0e}"
        name += f"_u{units}"
        name += f"_{n}x3.{extension}"
        filename = fig_path / name
        fig.savefig(filename)
        print(f"Saved plot in {filename}")
    fig.show()
    print_elapsed_time(start)


if __name__ == '__main__':
    sm = SineModel(noise=False, T=10)