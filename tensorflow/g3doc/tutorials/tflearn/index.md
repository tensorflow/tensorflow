## TF.Learn Quickstart

TensorFlow’s Learn API (TF.Learn) makes it easy to configure, train, and evaluate a
variety of machine learning models. In this quickstart tutorial, you’ll use TF.Learn
to construct a [Deep Neural Network](https://en.wikipedia.org/wiki/Artificial_neural_network)
classifier model and train it on [Fisher’s Iris data set](https://en.wikipedia.org/wiki/Iris_flower_data_set)
to predict flower species based on sepal/petal geometry. You’ll perform the following four steps:

1. Load CSVs containing Iris training/test data into a TensorFlow `Dataset`
2. Construct a [Deep Neural Network classifier](https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.learn.html#DNNClassifier)
3. Fit the DNN model using the training data
4. Evaluate the accuracy of the model

## Get Started
Remember to [install TensorFlow on your machine](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#download-and-setup) 
before getting started with this tutorial. The full code and datasets for this tutorial
can be found [here](https://www.tensorflow.org/code/tensorflow/examples/tutorials/tflearnqs/),
and the following sections walk through them in detail.

## Load the Iris CSV data to TensorFlow

The [Iris data set](https://en.wikipedia.org/wiki/Iris_flower_data_set) 
contains 150 rows of data, comprising 50 samples from each of three related 
Iris species: *Iris setosa*, *Iris virginica*, and *Iris versicolor*. Each row 
contains the following data for each flower sample: [sepal](https://en.wikipedia.org/wiki/Sepal) 
length, sepal width, [petal](https://en.wikipedia.org/wiki/Petal) length, petal width,
and flower species. Flower species are represented as integers, with 0 denoting *Iris
setosa*, 1 denoting *Iris versicolor*, and 2 denoting *Iris virginica*.

Sepal Length | Sepal Width | Petal Length | Petal Width | Species
:----------- | :---------- | :----------- | :---------- | :------
5.1          | 3.5         | 1.4          | 0.2         | 0
4.9          | 3.0         | 1.4          | 0.2         | 0
4.7          | 3.2         | 1.3          | 0.2         | 0
&hellip;     | &hellip;    | &hellip;     | &hellip;    | &hellip;
7.0          | 3.2         | 4.7          | 1.4         | 1
6.4          | 3.2         | 4.5          | 1.5         | 1
6.9          | 3.1         | 4.9          | 1.5         | 1
&hellip;     | &hellip;    | &hellip;     | &hellip;    | &hellip;
6.5          | 3.0         | 5.2          | 2.0         | 2
6.2          | 3.4         | 5.4          | 2.3         | 2
5.9          | 3.0         | 5.1          | 1.8         | 2

<!-- TODO: The rest of this section presumes that CSVs will live in same directory as tutorial examples; if not, update links and code -->
For this tutorial, the Iris data has been randomized and split into two separate CSVs: 
a training set of 120 samples ([iris_training.csv](https://www.tensorflow.org/code/tensorflow/examples/tutorials/tflearnqs/iris_training.csv)).
and a test set of 30 samples ([iris_test.csv](https://www.tensorflow.org/code/tensorflow/examples/tutorials/tflearnqs/iris_test.csv)).

To get started, first import TensorFlow, TF.Learn, and numpy:

```python
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
```

Next, load the training and test sets into `Dataset`s using the [`load_csv()`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/base.py#L36) 
method in `learn.datasets.base`. `load_csv()` has two required arguments: 
`filename`, which takes the filepath to the CSV file, 
and `target_dtype`, which takes the [`numpy` datatype](http://docs.scipy.org/doc/numpy/user/basics.types.html)
of the dataset's target value. Here, the target (the value you're training the model to predict) is
flower species, which is an integer from 0&ndash;2, so the appropriate `numpy` datatype is `np.int`:

```python
# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TEST = "iris_test.csv"

# Load datasets.
training_set = learn.datasets.base.load_csv(filename=IRIS_TRAINING, target_dtype=np.int)
test_set = learn.datasets.base.load_csv(filename=IRIS_TEST, target_dtype=np.int)
```

Next, assign variables to the feature data and target values: `x_train` for training-set feature data,
`x_test` for test-set feature data, `y_train` for training-set target values, and `y_test` for test-set
target values. Datasets in TensorFlow are [named tuples](https://docs.python.org/2/library/collections.html#collections.namedtuple),
and you can access feature data and target values via the `data` and `target` fields, respectively:

```python
x_train, x_test, y_train, y_test = training_set.data, test_set.data, \
  training_set.target, test_set.target
```

Later on, in "Fit the DNNClassifier to the Iris Training Data," you'll use `x_train` and `y_train` to 
train your model, and in "Evaluate Model Accuracy", you'll use `x_test` and `y_test`. But first,
you'll construct your model in the next section.

## Construct a Deep Neural Network Classifier

TF.Learn offers a variety of predefined models, called [`Estimator`s](https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.learn.html#estimators), 
which you can use "out of the box" to run training and evaluation operations on your data. 
Here, you'll configure a Deep Neural Network Classifier model to fit the iris data. Using TF.Learn, 
you can instantiate your [`DNNClassifier`](https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.learn.html#DNNClassifier)
with just one line of code:

```python
# Build 3 layer DNN with 10, 20, 10 units respectively. 
classifier = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3)
```

The code above creates a `DNNClassifier` model with three [hidden layers](http://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw), 
containing 10, 20, and 10 neurons, respectively (`hidden_units=[10, 20, 10]`), and three target
classes (`n_classes=3`).


## Fit the DNNClassifier to the Iris Training Data

Now that you've configured your DNN `classifier` model, you can fit it to the Iris training data
using the [`fit`](https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.learn.html#BaseEstimator.fit) 
method. Pass as arguments your feature data (`x_train`), target values
(`y_train`), and the number of steps to train (here, 200):

```python
# Fit model
classifier.fit(x=x_train, y=y_train, steps=200)
```

<!-- Style the below (up to the next section) as an aside (note?) -->

<!-- Pretty sure the following is correct, but maybe a SWE could verify? -->
The state of the model is preserved in the `classifier`, which means you can train iteratively if
you like. For example, the above is equivalent to the following:

```python
classifier.fit(x=x_train, y=y_train, steps=100)
classifier.fit(x=x_train, y=y_train, steps=100)
```

<!-- TODO: When tutorial exists for monitoring, link to it here -->
However, if you're looking to track the model while it trains, you'll likely
want to instead use a TensorFlow [`monitor`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/monitors.py)
to perform logging operations.

## Evaluate Model Accuracy

You've fit your `DNNClassifier` model on the Iris training data; now, you can check its accuracy on
the Iris test data using the [`evaluate`](https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.learn.html#BaseEstimator.evaluate)
method. Like `fit`, `evaluate` takes feature data and target values as arguments,
and returns a `dict` with the evaluation results. The following code passes the Iris
test data&mdash;`x_test` and `y_test`&mdash;to `evaluate`, retrieves `accuracy` from the
results, and prints it to output:

```python
accuracy_score = classifier.evaluate(x=x_test, y=y_test)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))
```

Run the full script, and check the accuracy results. You should get:

```
Accuracy: 0.933333
```

Not bad for a relatively small data set!

## Additional Resources

* For further reference materials on TF.Learn, see the official [API docs](https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.learn.html).

<!-- David, will the below be live when this tutorial is released? -->
* To learn more about using TF.Learn to create linear models, see 
[Large-scale Linear Models with TensorFlow](https://www.tensorflow.org/versions/r0.9/tutorials/linear/index.html).

* To experiment with neural network modeling and visualization in the browser, check out [Deep Playground](http://playground.tensorflow.org/).

* For more advanced tutorials on neural networks, see [Convolutional Neural Networks](https://www.tensorflow.org/versions/r0.9/tutorials/deep_cnn/index.html)
and [Recurrent Neural Networks](https://www.tensorflow.org/versions/r0.9/tutorials/recurrent/index.html).
