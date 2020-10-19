# Deep Learning
Deep Learning (DL) case studies.

## Time Series Analysis / Sequential data modeling

1. [Understandig time series](https://github.com/kbantoec/deep_learning/blob/master/tsa/understanding_time_series.ipynb)
   * In this notebook we create a batch generator object in order to forecast with neural networks. Meaning that we can feed into a deep learning model when training it. We also touch on the following topics: Fast Fourier Transformations, serial correlation, median forecasting, and ARIMA models. This notebook are my notes from chapter 4 of the following reference: Klaas, J. (2019). *Machine learning for finance: principles and practice for financial insiders*. Packt Publishing Ltd.
2. [Autoregressive model with TF2.3.1](https://github.com/kbantoec/deep_learning/blob/master/tf2/autoregressive_model.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kbantoec/deep_learning/blob/master/tf2/autoregressive_model.ipynb)
   * In this notebook we generate synthetic data from a sine function to test how well an autoregressive model predicts multiple steps ahead. We show the wrong way and the right way of forecasting. Hence, understanding the implications of accidentally incorporating look-ahead bias in the multi-step forecast. This notebook are my notes from the *"Tensorflow 2.0: Deep Learning and Artificial Intelligence"* course of Udemy.

## TensorFlow 2.0 study cases

1. [Basics of deep learning and neural networks](https://github.com/kbantoec/deep_learning/tree/master/basics_of_deep_learning_and_neural_networks/basics_of_deep_learning_and_neural_networks.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kbantoec/deep_learning/master?filepath=basics_of_deep_learning_and_neural_networks%2Fbasics_of_deep_learning_and_neural_networks.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kbantoec/deep_learning/blob/master/basics_of_deep_learning_and_neural_networks.ipynb)
2. [Linear Regression with TF2.3](https://github.com/kbantoec/deep_learning/blob/master/tf2/linear_regression.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kbantoec/deep_learning/master?filepath=tf2%2Flinear_regression.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kbantoec/deep_learning/blob/master/tf2/linear_regression.ipynb)
3. [Basics of TensorFlow 2.0](https://github.com/kbantoec/deep_learning/blob/master/tf2/tensorflow2.ipynb)
4. [Linear Models](https://github.com/kbantoec/deep_learning/blob/master/tf2/linear_models.ipynb)
   * How to build, solve and make predictions with models in TensorFlow 2. We will focus on a simple linear regression model and will try to predict housing prices. We will load and manipulate data, construct loss functions, perform minimization, make predictions, and reduce resource use with batch training.
   * Tags: `tensorflow.Variable`, `tensorflow.keras.losses.mae`, `tensorflow.keras.losses.mse`, `tensorflow.keras.optimizers.Adam`
5. [Classification problems with Deep Learning](https://github.com/kbantoec/deep_learning/blob/master/tf2/classification.ipynb)
   * In this notebook, we will see how to solve binary, multi-class, and multi-label problems with neural networks.
   * Tags: `tensorflow.keras.callbacks.EarlyStopping`, `tensorflow.keras.callbacks.ModelCheckpoint`, `sklearn.model_selection.train_test_split`, `tensorflow.keras.utils.to_categorical`, `tensorflow.keras.models.Sequential`, `tensorflow.keras.layers.Dense`, `pandas.Categorical`, `seaborn.pairplot`
6. [ResNet50](https://github.com/kbantoec/deep_learning/blob/master/tf2/resnet50.ipynb)
   * Here I provide an example on how to use the pre-trained ResNet50 model.
   * Libraries: 
     * From `tensorflow.keras.applications.resnet50` imported `ResNet50`, `decode_predictions`, `preprocess_input`
     * From `tensorflow.keras.preprocessing` imported `image`
     * From `numpy` used `expand_dims`
7. [Is Loki a border collie?](https://github.com/kbantoec/deep_learning/blob/master/tf2/cnn_loki.ipynb)
   * This is a fun project in which I pass images of my border collie to the ResNet50 convolutional neural network to observe the breed it predicts.
   * Tags: `tensorflow.keras.applications.resnet50.ResNet50`

## RNN

1. [Simple RNN for forecasting a sine function](https://github.com/kbantoec/deep_learning/blob/master/rnn/simple_rnn.ipynb): This notebook demonstrates the importance of doing sanity checks when building models in deep learning.
2. [Non linear LSTM](https://github.com/kbantoec/deep_learning/blob/master/rnn/lstm_nonlinear.ipynb)
   * In this notebook we will demystify LSTM comparing it to an AR (linear) model and a Simple RNN model. Throughout this notebook we are going to see that LSTMs are not magic and that they are not necessarily better at everything compared to the Simple RNN. These are my notes from the *"Tensorflow 2.0: Deep Learning and Artificial Intelligence"* course of Udemy.

