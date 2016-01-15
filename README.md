[![Travis-CI Build Status](https://travis-ci.org/tensorflow/skflow.svg?branch=master)](https://travis-ci.org/tensorflow/skflow)
[![Codecov Status](https://codecov.io/github/tensorflow/skflow/coverage.svg?precision=2)](https://codecov.io/github/tensorflow/skflow)
[![License](https://img.shields.io/github/license/tensorflow/skflow.svg)](http://www.apache.org/licenses/LICENSE-2.0.html)
[![Join the chat at https://gitter.im/tensorflow/skflow](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/tensorflow/skflow?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

# Scikit Flow

This is a simplified interface for TensorFlow, to get people started on predictive analytics and data mining.

Why TensorFlow?
* TensorFlow provides a good backbone for building different shapes of machine learning applications.
* It will continue to evolve both in the distributed direction and as general pipelinining machinery.

Why Scikit Flow?
* To smooth the transition from the Scikit Learn world of one-liner machine learning into the
more open world of building different shapes of ML models. You can start by using fit/predict and slide into TensorFlow APIs as you are getting comfortable.
* To provide a set of reference models that would be easy to integrate with existing code.

## Installation

Support versions of dependencies:
  * Python: 2.7, 3.4+
  * Scikit learn: 0.16, 0.17, 0.18+
  * Tensorflow: 0.5, 0.6+

First, make sure you have TensorFlow and Scikit Learn installed, then just run:

```Bash
pip install git+git://github.com/tensorflow/skflow.git
```

## Tutorial

* [Introduction to Scikit Flow and why you want to start learning TensorFlow](https://medium.com/@ilblackdragon/tensorflow-tutorial-part-1-c559c63c0cb1)
* [DNNs, custom model and Digit recognition examples](https://medium.com/@ilblackdragon/tensorflow-tutorial-part-2-9ffe47049c92)
* More coming soon.

## Usage

Below are few simple examples of the API. 
For more examples, please see [examples](https://github.com/tensorflow/skflow/tree/master/examples).

### General tips

* It's useful to re-scale dataset before passing to estimator to 0 mean and unit standard deviation. 
Stochastic Gradient Descent doesn't always do the right thing when variable are very different scale.

* Categorical variables should be managed before passing input to the estimator. I'll write a tutorial in coming days on how to handle categorical variables Deep Learning-style.

### Linear Classifier

Simple linear classification:

```Python
import skflow
from sklearn import datasets, metrics

iris = datasets.load_iris()
classifier = skflow.TensorFlowLinearClassifier(n_classes=3)
classifier.fit(iris.data, iris.target)
score = metrics.accuracy_score(classifier.predict(iris.data), iris.target)
print("Accuracy: %f" % score)
```

### Linear Regressor

Simple linear regression:

```Python
import skflow
from sklearn import datasets, metrics, preprocessing

boston = datasets.load_boston()
X = preprocessing.StandardScaler().fit_transform(boston.data)
regressor = skflow.TensorFlowLinearRegressor()
regressor.fit(X, boston.target)
score = metrics.mean_squared_error(regressor.predict(X), boston.target)
print ("MSE: %f" % score)
```

### Deep Neural Network

Example of 3 layer network with 10, 20 and 10 hidden units respectively:

```Python
import skflow
from sklearn import datasets, metrics

iris = datasets.load_iris()
classifier = skflow.TensorFlowDNNClassifier(hidden_units=[10, 20, 10], n_classes=3)
classifier.fit(iris.data, iris.target)
score = metrics.accuracy_score(classifier.predict(iris.data), iris.target)
print("Accuracy: %f" % score)
```

### Custom model

Example of how to pass a custom model to the TensorFlowEstimator:

```Python
import skflow
from sklearn import datasets, metrics

iris = datasets.load_iris()

def my_model(X, y):
    """This is DNN with 10, 20, 10 hidden layers, and dropout of 0.5 probability."""
    layers = skflow.ops.dnn(X, [10, 20, 10], keep_prob=0.5)
    return skflow.models.logistic_regression(layers, y)

classifier = skflow.TensorFlowEstimator(model_fn=my_model, n_classes=3)
classifier.fit(iris.data, iris.target)
score = metrics.accuracy_score(classifier.predict(iris.data), iris.target)
print("Accuracy: %f" % score)
```

### Custom model with multiple GPUs

To use multiple GPUs to build a custom model, everything else is the same as the example above
except that in the definition of custom model you'll need to specify the device:

```Python
import tensorflow as tf

def my_model(X, y):
    """
    This is DNN with 10, 20, 10 hidden layers, and dropout of 0.5 probability.

    Note: If you want to run this example with multiple GPUs, Cuda Toolkit 7.0 and
    CUDNN 6.5 V2 from NVIDIA need to be installed beforehand. 
    """
    with tf.device('/gpu:1'):
    	layers = skflow.ops.dnn(X, [10, 20, 10], keep_prob=0.5)
    with tf.device('/gpu:2'):
    	return skflow.models.logistic_regression(layers, y)
```

### Saving / Restoring models

Each estimator has a `save` method which takes folder path where all model information will be saved. 
For restoring you can just call `skflow.TensorFlowEstimator.restore(path)` and it will return object of your class.

Some example code:

```Python
import skflow

classifier = skflow.TensorFlowLinearRegression()
classifier.fit(...)
classifier.save('/tmp/tf_examples/my_model_1/')

new_classifier = TensorFlowEstimator.restore('/tmp/tf_examples/my_model_2')
new_classifier.predict(...)
```

### Summaries

To get nice visualizations and summaries you can use `logdir` parameter on `fit`.
It will start writing summaries for `loss` and histograms for variables in your model.
You can also add custom summaries in your custom model function by calling `tf.summary` and
passing Tensors to report.

```Python
classifier = skflow.TensorFlowLinearRegression()
classifier.fit(X, y, logdir='/tmp/tf_examples/my_model_1/')
```

Then run next command in commandline:
```bash
tensorboard --logdir=/tmp/tf_examples/my_model_1
```
and follow reported url.

Graph visualization:
![Text classification RNN Graph](https://raw.githubusercontent.com/tensorflow/skflow/master/docs/images/text_classification_rnn_graph.png)

Loss visualization:
![Text classification RNN Loss](https://raw.githubusercontent.com/tensorflow/skflow/master/docs/images/text_classification_rnn_loss.png)

## More examples

See examples folder for:

* Easy way to handle categorical variables - words are just an example of categorical variable.
* Text Classification - see examples for RNN, CNN on word and characters.
* Images (CNNs) - see example for digit recognition. 
* More & deeper - different examples showing DNNs and CNNs

