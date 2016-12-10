# TF Learn

TF Learn is a simplified interface for TensorFlow, to get people started on predictive analytics and data mining. The library covers a variety of needs: from linear models to *Deep Learning* applications like text and image understanding.

### Why *TensorFlow*?

* TensorFlow provides a good backbone for building different shapes of machine learning applications.
* It will continue to evolve both in the distributed direction and as general pipelinining machinery.

### Why *TensorFlow Learn*?

- To smooth the transition from the [scikit-learn](http://scikit-learn.org/stable/) world of one-liner machine learning into the more open world of building different shapes of ML models. You can start by using [fit](../../../../g3doc/api_docs/python/contrib.learn.md#Estimator.fit)/[predict](../../../../g3doc/api_docs/python/contrib.learn.md#Estimator.predict) and slide into TensorFlow APIs as you are getting comfortable.
- To provide a set of reference models that will be easy to integrate with existing code.

## Installation

[Install TensorFlow](../../../../g3doc/get_started/os_setup.md), and then simply import `learn` via `from tensorflow.contrib.learn` or use `tf.contrib.learn`.

Optionally you can install [scikit-learn](http://scikit-learn.org/stable/) and [pandas](http://pandas.pydata.org/) for additional functionality.

### Tutorials

- [TF Learn Quickstart](../../../../g3doc/tutorials/tflearn/index.md). Build, train, and evaluate a neural network with just a few lines of code.
- [Linear Model](../../../../g3doc/tutorials/wide/index.md). Learn the basics of building linear models.
- [Logging and Monitoring](../../../../g3doc/tutorials/monitors/index.md). Use the Monitor API to audit training of a neural network.
- [Wide and Deep Learning](../../../../g3doc/tutorials/wide_and_deep/index.md). Jointly train a linear model and a deep neural network.
-  More coming soon.

### Community

- Twitter [#tensorflow](https://twitter.com/search?q=tensorflow&src=typd).
- StackOverflow with [tensorflow tag](http://stackoverflow.com/questions/tagged/tensorflow) for questions and struggles.
- GitHub [issues](https://github.com/tensorflow/tensorflow/issues) for technical discussions and feature requests.

### Usage

Below are a few simple examples of the API. For more examples, please see [examples](https://www.tensorflow.org/code/tensorflow/examples/skflow).

General tips:

-  It's useful to rescale a dataset to 0 mean and unit standard deviation before passing it to an [`Estimator`](../../../../g3doc/api_docs/python/contrib.learn.md#estimators). [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) doesn't always do the right thing when variable are at very different scales.

-  Categorical variables should be managed before passing input to the estimator.

## Linear Classifier

Simple linear classification:

```python
import tensorflow.contrib.learn.python.learn as learn
from sklearn import datasets, metrics

iris = datasets.load_iris()
feature_columns = learn.infer_real_valued_columns_from_input(iris.data)
classifier = learn.LinearClassifier(n_classes=3, feature_columns=feature_columns)
classifier.fit(iris.data, iris.target, steps=200, batch_size=32)
iris_predictions = list(classifier.predict(iris.data, as_iterable=True))
score = metrics.accuracy_score(iris.target, iris_predictions)
print("Accuracy: %f" % score)
```

## Linear Regressor

Simple linear regression:

```python
import tensorflow.contrib.learn.python.learn as learn
from sklearn import datasets, metrics, preprocessing

boston = datasets.load_boston()
x = preprocessing.StandardScaler().fit_transform(boston.data)
feature_columns = learn.infer_real_valued_columns_from_input(x)
regressor = learn.LinearRegressor(feature_columns=feature_columns)
regressor.fit(x, boston.target, steps=200, batch_size=32)
boston_predictions = list(regressor.predict(x, as_iterable=True))
score = metrics.mean_squared_error(boston_predictions, boston.target)
print ("MSE: %f" % score)
```

## Deep Neural Network

Example of 3 layer network with 10, 20 and 10 hidden units respectively:

```python
import tensorflow.contrib.learn.python.learn as learn
from sklearn import datasets, metrics

iris = datasets.load_iris()
feature_columns = learn.infer_real_valued_columns_from_input(iris.data)
classifier = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3, feature_columns=feature_columns)
classifier.fit(iris.data, iris.target, steps=200, batch_size=32)
iris_predictions = list(classifier.predict(iris.data, as_iterable=True))
score = metrics.accuracy_score(iris.target, iris_predictions)
print("Accuracy: %f" % score)
```

## Custom model

Example of how to pass a custom model to the Estimator:

```python
from sklearn import datasets
from sklearn import metrics
import tensorflow as tf
import tensorflow.contrib.layers.python.layers as layers
import tensorflow.contrib.learn.python.learn as learn

iris = datasets.load_iris()

def my_model(features, labels):
  """DNN with three hidden layers."""
  # Convert the labels to a one-hot tensor of shape (length of features, 3) and
  # with a on-value of 1 for each one-hot vector of length 3.
  labels = tf.one_hot(labels, 3, 1, 0)

  # Create three fully connected layers respectively of size 10, 20, and 10.
  features = layers.stack(features, layers.fully_connected, [10, 20, 10])

  # Create two tensors respectively for prediction and loss.
  prediction, loss = (
      tf.contrib.learn.models.logistic_regression(features, labels)
  )

  # Create a tensor for training op.
  train_op = tf.contrib.layers.optimize_loss(
      loss, tf.contrib.framework.get_global_step(), optimizer='Adagrad',
      learning_rate=0.1)

  return {'class': tf.argmax(prediction, 1), 'prob': prediction}, loss, train_op

classifier = learn.Estimator(model_fn=my_model)
classifier.fit(iris.data, iris.target, steps=1000)

y_predicted = [
  p['class'] for p in classifier.predict(iris.data, as_iterable=True)]
score = metrics.accuracy_score(iris.target, y_predicted)
print('Accuracy: {0:f}'.format(score))
```

## Saving / Restoring models

Each estimator supports a `model_dir` argument, which takes a folder path where all model information will be saved:

```python
classifier = learn.DNNClassifier(..., model_dir="/tmp/my_model")
```

If you run multiple `fit` operations on the same `Estimator`, training will resume where the last operation left off, e.g.:

<pre><strong>classifier = learn.DNNClassifier(..., model_dir="/tmp/my_model")
classifier.fit(..., steps=300)</strong>
INFO:tensorflow:Create CheckpointSaverHook
INFO:tensorflow:loss = 2.40115, step = 1
INFO:tensorflow:Saving checkpoints for 1 into /tmp/leftoff/model.ckpt.
INFO:tensorflow:loss = 0.338706, step = 101
INFO:tensorflow:loss = 0.159414, step = 201
INFO:tensorflow:Saving checkpoints for 300 into /tmp/leftoff/model.ckpt.
INFO:tensorflow:Loss for final step: 0.0953846.

<strong>classifier.fit(..., steps=300)</strong>
INFO:tensorflow:Create CheckpointSaverHook
INFO:tensorflow:loss = 0.113173, step = 301
INFO:tensorflow:Saving checkpoints for 301 into /tmp/leftoff/model.ckpt.
INFO:tensorflow:loss = 0.175782, step = 401
INFO:tensorflow:loss = 0.119735, step = 501
INFO:tensorflow:Saving checkpoints for 600 into /tmp/leftoff/model.ckpt.
INFO:tensorflow:Loss for final step: 0.0518137.</pre>

To restore checkpoints to a new `Estimator`, just pass it the same `model_dir` argument, e.g.:

<pre><strong>classifier = learn.DNNClassifier(..., model_dir="/tmp/my_model")
classifier.fit(..., steps=300)</strong>
INFO:tensorflow:Create CheckpointSaverHook
INFO:tensorflow:loss = 1.16335, step = 1
INFO:tensorflow:Saving checkpoints for 1 into /tmp/leftoff/model.ckpt.
INFO:tensorflow:loss = 0.176995, step = 101
INFO:tensorflow:loss = 0.184573, step = 201
INFO:tensorflow:Saving checkpoints for 300 into /tmp/leftoff/model.ckpt.
INFO:tensorflow:Loss for final step: 0.0512496.

<strong>classifier2 = learn.DNNClassifier(..., model_dir="/tmp/my_model")
classifier2.fit(..., steps=300)</strong>
INFO:tensorflow:Create CheckpointSaverHook
INFO:tensorflow:loss = 0.0543797, step = 301
INFO:tensorflow:Saving checkpoints for 301 into /tmp/leftoff/model.ckpt.
INFO:tensorflow:loss = 0.101036, step = 401
INFO:tensorflow:loss = 0.137956, step = 501
INFO:tensorflow:Saving checkpoints for 600 into /tmp/leftoff/model.ckpt.
INFO:tensorflow:Loss for final step: 0.0162506.</pre>

## Summaries

If you supply a `model_dir` argument to your `Estimator`s, TensorFlow will write summaries for ``loss`` and histograms for variables in this directory. (You can also add custom summaries in your custom model function by calling [Summary](../../../../g3doc/api_docs/python/train.md#summary-operations) operations.)

To view the summaries in TensorBoard, run the following command, where `logdir` is the `model_dir` for your `Estimator`:

```shell
tensorboard --logdir=/tmp/tf_examples/my_model_1
```

and then load the reported URL.

**Graph visualization**

![Text classification RNN Graph](https://raw.githubusercontent.com/tensorflow/skflow/master/g3doc/images/text_classification_rnn_graph.png)

**Loss visualization**

![Text classification RNN Loss](https://raw.githubusercontent.com/tensorflow/skflow/master/g3doc/images/text_classification_rnn_loss.png)

## More examples

See the [examples folder](https://www.tensorflow.org/code/tensorflow/examples/skflow) for:

-  An easy way to handle [categorical variables](https://www.tensorflow.org/code/tensorflow/examples/skflow/text_classification.py) (words are just an example of a categorical variable)
-  Text Classification: see examples for [RNN](https://www.tensorflow.org/code/tensorflow/examples/skflow/text_classification_character_rnn.py) and [CNN](https://www.tensorflow.org/code/tensorflow/examples/skflow/text_classification_character_cnn.py) on characters
-  [Language modeling and text sequence to sequence](https://www.tensorflow.org/code/tensorflow/examples/skflow/language_model.py)
-  [Digit recognition using a CNN](https://www.tensorflow.org/code/tensorflow/examples/skflow/digits.py)
-  And much more!
