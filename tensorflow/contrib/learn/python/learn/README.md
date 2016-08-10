# TF Learn (aka Scikit Flow)

[![Join the chat at https://gitter.im/tensorflow/skflow](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/tensorflow/skflow?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

This is a simplified interface for TensorFlow, to get people started on predictive analytics and data mining.

Library covers variety of needs from linear models to *Deep Learning* applications like text and image understanding.

### Why *TensorFlow*?

- TensorFlow provides a good backbone for building different shapes of machine learning applications.
- It will continue to evolve both in the distributed direction and as general pipelinining machinery.

### Why *TensorFlow Learn* (Scikit Flow)?

- To smooth the transition from the Scikit Learn world of one-liner machine learning into the more open world of building different shapes of ML models. You can start by using fit/predict and slide into TensorFlow APIs as you are getting comfortable.
- To provide a set of reference models that would be easy to integrate with existing code.

## Installation

Optionally you can install Scikit Learn and Pandas for additional functionality.

Then you can simply import `learn` via `from tensorflow.contrib.learn` or use `tf.contrib.learn`.


### Tutorial

-  [Introduction to Scikit Flow and Why You Want to Start Learning
   TensorFlow](https://medium.com/@ilblackdragon/tensorflow-tutorial-part-1-c559c63c0cb1)
-  [DNNs, Custom model and Digit Recognition
   examples](https://medium.com/@ilblackdragon/tensorflow-tutorial-part-2-9ffe47049c92)
-  [Categorical Variables: One Hot vs Distributed
   representation](https://medium.com/@ilblackdragon/tensorflow-tutorial-part-3-c5fc0662bc08)
-  [Scikit Flow Key Features Illustrated](http://terrytangyuan.github.io/2016/03/14/scikit-flow-intro/)
-  More coming soon.

### Community

- Twitter [#skflow](https://twitter.com/search?q=skflow&src=typd).
- StackOverflow with [skflow tag](http://stackoverflow.com/questions/tagged/skflow) for questions and struggles.
- Github [issues](https://github.com/tensorflow/tensorflow/issues) for technical discussions and feature requests.
- [Gitter channel](https://gitter.im/tensorflow/skflow) for non-trivial discussions.

### Usage

Below are few simple examples of the API. For more examples, please see [examples](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/skflow).

## General tips

-  It's useful to re-scale dataset before passing to estimator to 0 mean and unit standard deviation. Stochastic Gradient Descent doesn't always do the right thing when variable are very different scale.

-  Categorical variables should be managed before passing input to the estimator.

## Linear Classifier

Simple linear classification:

```python
import tensorflow.contrib.learn.python.learn as learn
from sklearn import datasets, metrics

iris = datasets.load_iris()
classifier = learn.LinearClassifier(n_classes=3)
classifier.fit(iris.data, iris.target, steps=200, batch_size=32)
score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
print("Accuracy: %f" % score)
```

## Linear Regressor

Simple linear regression:

```python
import tensorflow.contrib.learn.python.learn as learn
from sklearn import datasets, metrics, preprocessing

boston = datasets.load_boston()
x = preprocessing.StandardScaler().fit_transform(boston.data)
regressor = learn.LinearRegressor()
regressor.fit(x, boston.target, steps=200, batch_size=32)
score = metrics.mean_squared_error(regressor.predict(x), boston.target)
print ("MSE: %f" % score)
```

## Deep Neural Network

Example of 3 layer network with 10, 20 and 10 hidden units respectively:

```python
import tensorflow.contrib.learn.python.learn as learn
from sklearn import datasets, metrics

iris = datasets.load_iris()
classifier = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3)
classifier.fit(iris.data, iris.target, steps=200, batch_size=32)
score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
print("Accuracy: %f" % score)
```

## Custom model

Example of how to pass a custom model to the Estimator:

```python
import tensorflow.contrib.learn.python.learn as learn
from sklearn import datasets, metrics

iris = datasets.load_iris()

def my_model(x, y):
    """This is DNN with 10, 20, 10 hidden layers, and dropout of 0.5 probability."""
    layers = learn.ops.dnn(x, [10, 20, 10], dropout=0.5)
    return learn.models.logistic_regression(layers, y)

classifier = learn.Estimator(model_fn=my_model, n_classes=3)
classifier.fit(iris.data, iris.target)
score = metrics.accuracy_score(iris.target, classifier.predict(iris.data))
print("Accuracy: %f" % score)
```

## Saving / Restoring models

Each estimator has a ``save`` method which takes folder path where all model information will be saved. For restoring you can just call ``learn.Estimator.restore(path)`` and it will return object of your class.

Some example code:

```python
classifier = learn.LinearRegressor()
classifier.fit(...)
classifier.save('/tmp/tf_examples/my_model_1/')

new_classifier = Estimator.restore('/tmp/tf_examples/my_model_2')
new_classifier.predict(...)
```

## Summaries

To get nice visualizations and summaries you can use ``logdir`` parameter on ``fit``. It will start writing summaries for ``loss`` and histograms for variables in your model. You can also add custom summaries in your custom model function by calling ``tf.summary`` and passing Tensors to report.

```python
classifier = learn.LinearRegressor()
classifier.fit(x, y, logdir='/tmp/tf_examples/my_model_1/')
```

Then run next command in command line:

```shell
tensorboard --logdir=/tmp/tf_examples/my_model_1
```

and follow reported url.

Graph visualization: |Text classification RNN Graph|

Loss visualization: |Text classification RNN Loss|

## More examples

See [examples folder](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/skflow) for:

-  Easy way to handle categorical variables - words are just an example of categorical variable.
-  Text Classification - see examples for RNN, CNN on word and characters.
-  Language modeling and text sequence to sequence.
-  Images (CNNs) - see example for digit recognition.
-  More & deeper - different examples showing DNNs and CNNs

![Text classification RNN Graph](https://raw.githubusercontent.com/tensorflow/skflow/master/g3doc/images/text_classification_rnn_graph.png)

![Text classification RNN Loss](https://raw.githubusercontent.com/tensorflow/skflow/master/g3doc/images/text_classification_rnn_loss.png)
