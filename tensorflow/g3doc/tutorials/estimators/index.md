# Creating Estimators in tf.contrib.learn

The tf.contrib.learn framework makes it easy to construct and train machine
learning models via its high-level
[Estimator](../../api_docs/python/contrib.learn.md#estimators) API. `Estimator`
offers classes you can instantiate to quickly configure common model types such
as regressors and classifiers:

*   [`LinearClassifier`](../../api_docs/python/contrib.learn.md#LinearClassifier):
    Constructs a linear classification model.
*   [`LinearRegressor`](../../api_docs/python/contrib.learn.md#LinearRegressor):
    Constructs a linear regression model.
*   [`DNNClassifier`](../../api_docs/python/contrib.learn.md#DNNClassifier):
    Construct a neural network classification model.
*   [`DNNRegressor`](../../api_docs/python/contrib.learn.md#DNNRegressor):
    Construct a neural network regressions model.

But what if none of `tf.contrib.learn`'s predefined model types meets your
needs? Perhaps you need more granular control over model configuration, such as
the ability to customize the loss function used for optimization, or specify
different activation functions for each neural network layer. Or maybe you're
implementing a ranking or recommendation system, and neither a classifier nor a
regressor is appropriate for generating predictions.

This tutorial covers how to create your own `Estimator` using the building blocks
provided in `tf.contrib.learn`, which will predict the ages of
[abalones](https://en.wikipedia.org/wiki/Abalone) based on their physical
measurements. You'll learn how to do the following:

*   Instantiate an `Estimator`
*   Construct a custom model function
*   Configure a neural network using `tf.contrib.layers`
*   Choose an appropriate loss function from `tf.contrib.losses`
*   Define a training op for your model
*   Generate and return predictions

## Prerequisites

This tutorial assumes you already know tf.contrib.learn API basics, such as
feature columns and `fit()` operations. If you've never used tf.contrib.learn
before, or need a refresher, you should first review the following tutorials:

*   [tf.contrib.learn Quickstart](../tflearn/index.md): Quick introduction to
    training a neural network using tf.contrib.learn.
*   [TensorFlow Linear Model Tutorial](../wide/index.md): Introduction to
    feature columns, and an overview on building a linear classifier in
    tf.contrib.learn.

## An Abalone Age Predictor {#abalone-predictor}

It's possible to estimate the age of an
[abalone](https://en.wikipedia.org/wiki/Abalone) (sea snail) by the number of
rings on its shell. However, because this task requires cutting, staining, and
viewing the shell under a microscope, it's desirable to find other measurements
that can predict age.

The [Abalone Data Set](https://archive.ics.uci.edu/ml/datasets/Abalone) contains
the following [feature
data](https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names)
for abalone:

| Feature        | Description                                               |
| -------------- | --------------------------------------------------------- |
| Length         | Length of abalone (in longest direction; in mm)           |
| Diameter       | Diameter of abalone (measurement perpendicular to length; |
:                : in mm)                                                    :
| Height         | Height of abalone (with its meat inside shell; in mm)     |
| Whole Weight   | Weight of entire abalone (in grams)                       |
| Shucked Weight | Weight of abalone meat only (in grams)                    |
| Viscera Weight | Gut weight of abalone (in grams), after bleeding          |
| Shell Weight   | Weight of dried abalone shell (in grams)                  |

The label to predict is number of rings, as a proxy for abalone age.

![Abalone shell](../../images/abalone_shell.jpg) **[“Abalone
shell”](https://www.flickr.com/photos/thenickster/16641048623/) (by [Nicki Dugan
Pogue](https://www.flickr.com/photos/thenickster/), CC BY-SA 2.0)**

## Setup

This tutorial uses three data sets.
[`abalone_train.csv`](http://download.tensorflow.org/data/abalone_train.csv)
contains labeled training data comprising 3,320 examples.
[`abalone_test.csv`](http://download.tensorflow.org/data/abalone_test.csv)
contains labeled test data for 850 examples.
[`abalone_predict`](http://download.tensorflow.org/data/abalone_predict.csv)
contains 7 examples on which to make predictions.

The following sections walk through writing the `Estimator` code step by step;
the [full, final code is available
here](https://www.tensorflow.org/code/tensorflow/examples/tutorials/estimators/abalone.py).

## Loading Abalone CSV Data into TensorFlow Datasets

To feed the abalone dataset into the model, you'll need to download and load the
CSVs into TensorFlow `Dataset`s. First, add some standard Python and TensorFlow
imports:

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import urllib

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
```

Then define flags to allow users to optionally specify CSV files for training,
test, and prediction datasets via the command line (by default, files will be
downloaded from [tensorflow.org](https://www.tensorflow.org/)), and enable
logging:

```python
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "train_data",
    "",
    "Path to the training data.")
flags.DEFINE_string(
    "test_data",
    "",
    "Path to the test data.")
flags.DEFINE_string(
    "predict_data",
    "",
    "Path to the prediction data.")

tf.logging.set_verbosity(tf.logging.INFO)
```

Then define a function to load the CSVs (either from files specified in
command-line options, or downloaded from
[tensorflow.org](https://www.tensorflow.org/)):

```python
def maybe_download():
  """Maybe downloads training data and returns train and test file names."""
  if FLAGS.train_data:
    train_file_name = FLAGS.train_data
  else:
    train_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.urlretrieve("http://download.tensorflow.org/data/abalone_train.csv", train_file.name)
    train_file_name = train_file.name
    train_file.close()
    print("Training data is downloaded to %s" % train_file_name)

  if FLAGS.test_data:
    test_file_name = FLAGS.test_data
  else:
    test_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.urlretrieve("http://download.tensorflow.org/data/abalone_test.csv", test_file.name)
    test_file_name = test_file.name
    test_file.close()
    print("Test data is downloaded to %s" % test_file_name)

  if FLAGS.predict_data:
    predict_file_name = FLAGS.predict_data
  else:
    predict_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.urlretrieve("http://download.tensorflow.org/data/abalone_predict.csv", predict_file.name)
    predict_file_name = predict_file.name
    predict_file.close()
    print("Prediction data is downloaded to %s" % predict_file_name)

  return train_file_name, test_file_name, predict_file_name
```

Finally, create `main()` and load the abalone CSVs into `Datasets`:

```python
def main(unused_argv):
  # Load datasets
  abalone_train, abalone_test, abalone_predict = maybe_download()

  # Training examples
  training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename=abalone_train,
      target_dtype=np.int,
      features_dtype=np.float64)

  # Test examples
  test_set = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename=abalone_test,
      target_dtype=np.int,
      features_dtype=np.float64)

  # Set of 7 examples for which to predict abalone ages
  prediction_set = tf.contrib.learn.datasets.base.load_csv_without_header(
      filename=abalone_predict,
      target_dtype=np.int,
      features_dtype=np.float64)

if __name__ == "__main__":
  tf.app.run()
```

## Instantiating an Estimator

When defining a model using one of tf.contrib.learn's provided classes, such as
`DNNClassifier`, you supply all the configuration parameters right in the
constructor, e.g.:

```python
my_nn = tf.contrib.learn.DNNClassifier(feature_columns=[age, height, weight],
                                       hidden_units=[10, 10, 10],
                                       activation_fn=tf.nn.relu,
                                       dropout=0.2,
                                       n_classes=3,
                                       optimizer="Adam")
```

You don't need to write any further code to instruct TensorFlow how to train the
model, calculate loss, or return predictions; that logic is already baked into
the `DNNClassifier`.

By contrast, when you're creating your own estimator from scratch, the
constructor accepts just two high-level parameters for model configuration,
`model_fn` and `params`:

```python
nn = tf.contrib.learn.Estimator(
    model_fn=model_fn, params=model_params)
```

*   `model_fn`: A function object that contains all the aforementioned logic to
    support training, evaluation, and prediction. You are responsible for
    implementing that functionality. The next section, [Constructing the
    `model_fn`](#constructing-modelfn) covers creating a model function in
    detail.

*   `params`: An optional dict of hyperparameters (e.g., learning rate, dropout)
    that will be passed into the `model_fn`.

NOTE: Just like `tf.contrib.learn`'s predefined regressors and classifiers, the
`Estimator` initializer also accepts the general configuration
arguments `model_dir` and `config`.

For the abalone age predictor, the model will accept one hyperparameter:
learning rate. Define `LEARNING_RATE` as a constant at the beginning of your
code (highlighted in bold below), right after the logging configuration:

<pre><code class="lang-python">tf.logging.set_verbosity(tf.logging.INFO)

<strong># Learning rate for the model
LEARNING_RATE = 0.001</strong></code></pre>

NOTE: Here, `LEARNING_RATE` is set to `0.001`, but you can tune this value as
needed to achieve the best results during model training.

Then, add the following code to `main()`, which creates the dict `model_params`
containing the learning rate and instantiates the `Estimator`:

```python
# Set model params
model_params = {"learning_rate": LEARNING_RATE}

# Build 2 layer fully connected DNN with 10, 10 units respectively.
nn = tf.contrib.learn.Estimator(
    model_fn=model_fn, params=model_params)
```

## Constructing the `model_fn` {#constructing-modelfn}

The basic skeleton for an `Estimator` API model function looks like this:

```python
def model_fn(features, targets, mode, params):
   # Logic to do the following:
   # 1. Configure the model via TensorFlow operations
   # 2. Define the loss function for training/evaluation
   # 3. Define the training operation/optimizer
   # 4. Generate predictions
   return predictions, loss, train_op
```

The `model_fn` must accept three arguments:

*   `features`: A dict containing the features passed to the model via `fit()`,
    `evaluate()`, or `predict()`.
*   `targets`: A `Tensor` containing the labels passed to the model via `fit()`,
    `evaluate()`, or `predict()`. Will be empty for `predict()` calls, as these
    are the values the model will infer.
*   `mode`: One of the following
    [`ModeKeys`](../../api_docs/python/contrib.learn.md#ModeKeys) string values
    indicating the context in which the model_fn was invoked:
    *   `tf.contrib.learn.ModeKeys.TRAIN` The `model_fn` was invoked in training
        mode—e.g., via a `fit()` call.
    *   `tf.contrib.learn.ModeKeys.EVAL`. The `model_fn` was invoked in
        evaluation mode—e.g., via an `evaluate()` call.
    *   `tf.contrib.learn.ModeKeys.INFER`. The `model_fn` was invoked in
        inference mode—e.g., via a `predict()` call.

`model_fn` may also accept a `params` argument containing a dict of
hyperparameters used for training (as shown in the skeleton above).

The body of the function perfoms the following tasks (described in detail in the
sections that follow):

*   Configuring the model—here, for the abalone predictor, this will be a neural
    network.
*   Defining the loss function used to calculate how closely the model's
    predictions match the target values.
*   Defining the training operation that specifies the `optimizer` algorithm to
    minimize the loss values calculated by the loss function.

Finally, depending on the `mode` in which `model_fn` is run, it must return one
or more of the following three values:

*   `predictions` (required in `INFER` and `EVAL` modes). A dict that maps key
    names of your choice to `Tensor`s containing the predictions from the model,
    e.g.:

    ```python
    predictions = {"results": tensor_of_predictions}
    ```

    In `INFER` mode, the dict that you return from `model_fn` will then be
    returned by `predict()`, so you can construct it in the format in which
    you'd like to consume it.

    In `EVAL` mode, the dict is used by [metric
    functions](../../api_docs/python/contrib.metrics.md#metric-ops) to compute
    metrics. Any
    [`MetricSpec`](https://www.tensorflow.org/code/tensorflow/contrib/learn/python/learn/metric_spec.py)
    objects passed to the `metrics` argument of `evaluate()` must have a
    `prediction_key` that matches the key name of the corresponding predictions
    in `predictions`.

*   `loss` (required in `EVAL` and `TRAIN` mode). A `Tensor` containing a scalar
    loss value: the output of the model's loss function (discussed in more depth
    later in [Defining loss for the model](#defining-loss)) calculated over all
    the input examples. This is used in `TRAIN` mode for error handling and
    logging, and is automatically included as a metric in `EVAL` mode.

*   `train_op` (required only in `TRAIN` mode). An Op that runs one step of
    training.

### Configuring a neural network with `tf.contrib.layers`

Constructing a [neural
network](https://en.wikipedia.org/wiki/Artificial_neural_network) entails
creating and connecting the input layer, the hidden layers, and the output
layer.

The input layer is a series of nodes (one for each feature in the model) that
will accept the feature data that is passed to the `model_fn` in the `features`
argument. If `features` contains an n-dimenional `Tensor` with all your feature
data (which is the case if `x` and `y` `Dataset`s are passed to `fit()`,
`evaluate()`, and `predict()` directly), then it can serve as the input layer.
If `features` contains a dict of [feature
columns](../linear/overview.md#feature-columns-and-transformations) passed to
the model via an input function, you can convert it to an input-layer `Tensor`
with the `input_from_feature_columns()` function in
[tf.contrib.layers](../../api_docs/python/contrib.layers.md#layers-contrib).

```python
input layer = tf.contrib.layers.input_from_feature_columns(columns_to_tensors=features, feature_columns=[age, height, weight])
```

As shown above, `input_from_feature_columns()` takes two required arguments:

*   `columns_to_tensors`. A mapping of the model's `FeatureColumns` to the
    `Tensors` containing the corresponding feature data. This is exactly what is
    passed to the `model_fn` in the `features` argument.
*   `feature_columns`. A list of all the `FeatureColumns` in the model—`age`,
    `height`, and `weight` in the above example.

The input layer of the neural network then must be connected to one or more
hidden layers via an [activation
function](https://en.wikipedia.org/wiki/Activation_function) that performs a
nonlinear transformation on the data from the previous layer. The last hidden
layer is then connected to the output layer, the final layer in the model.
tf.contrib.layers provides the following convenience functions for constructing
fully connected layers:

*   `relu(inputs, num_outputs)`. Create a layer of `num_outputs` nodes fully
    connected to the previous layer `inputs` with a [ReLU activation
    function](https://en.wikipedia.org/wiki/Rectifier_\(neural_networks\))
    ([tf.nn.relu](../../api_docs/python/nn.md#relu)):

    ```python
    hidden_layer = tf.contrib.layers.relu(inputs=input_layer, num_outputs=10)
    ```

*   `relu6(inputs, num_outputs)`. Create a layer of `num_outputs` nodes fully
    connected to the previous layer `hidden_layer` with a ReLU 6 activation
    function ([tf.nn.relu6](../../api_docs/python/nn.md#relu6)):

    ```python
    second_hidden_layer = tf.contrib.layers.relu6(inputs=hidden_layer, num_outputs=20)
    ```

*   `linear(inputs, num_outputs)`. Create a layer of `num_outputs` nodes fully
    connected to the previous layer `second_hidden_layer` with *no* activation
    function, just a linear transformation:

    ```python
    output_layer = tf.contrib.layers.linear(inputs=second_hidden_layer, num_outputs=3)
    ```

All these functions are
[partials](https://docs.python.org/2/library/functools.html#functools.partial)
of the more general
[`fully_connected()`](../../api_docs/python/contrib.layers.md#fully_connected)
function, which can be used to add fully connected layers with other activation
functions, e.g.:

```python
output_layer = tf.contrib.layers.fully_connected(inputs=second_hidden_layer,
                                                 num_outputs=10,
                                                 activation_fn=tf.sigmoid)
```

The above code creates the neural network layer `output_layer`, which is fully
connected to `second_hidden_layer` with a sigmoid activation function
([`tf.sigmoid`](../../api_docs/python/nn.md#sigmoid)). For a list of predefined
activation functions available in TensorFlow, see the [API
docs](../../api_docs/python/nn.md#activation-functions).

Putting it all together, the following code constructs a full neural network for
the abalone predictor, and captures its predictions:

```python
def model_fn(features, targets, mode, params):
  """Model function for Estimator."""

  # Connect the first hidden layer to input layer
  # (features) with relu activation
  first_hidden_layer = tf.contrib.layers.relu(features, 10)

  # Connect the second hidden layer to first hidden layer with relu
  second_hidden_layer = tf.contrib.layers.relu(first_hidden_layer, 10)

  # Connect the output layer to second hidden layer (no activation fn)
  output_layer = tf.contrib.layers.linear(second_hidden_layer, 1)

  # Reshape output layer to 1-dim Tensor to return predictions
  predictions = tf.reshape(output_layer, [-1])
  predictions_dict = {"ages": predictions}
  ...
```

Here, because you'll be passing the abalone `Datasets` directly to `fit()`,
`evaluate()`, and `predict()` via `x` and `y` arguments, the input layer is the
`features` `Tensor` passed to the `model_fn`. The network contains two hidden
layers, each with 10 nodes and a ReLU activation function. The output layer
contains no activation function, and is
[reshaped](../../api_docs/python/array_ops.md#reshape) to a one-dimensional
tensor to capture the model's predictions, which are stored in
`predictions_dict`.

### Defining loss for the model {#defining-loss}

The `model_fn` must return a `Tensor` that contains the loss value, which
quantifies how well the model's predictions reflect the target values during
training and evaluation runs. The
[`tf.contrib.losses`](../../api_docs/python/contrib.losses.md#losses-contrib)
module provides convenience functions for calculating loss using a variety of
metrics, including:

*   `absolute_difference(predictions, targets)`. Calculates loss using the
    [absolute-difference
    formula](https://en.wikipedia.org/wiki/Deviation_\(statistics\)#Unsigned_or_absolute_deviation)
    (also known as L<sub>1</sub> loss).

*   `log_loss(predictions, targets)`. Calculates loss using the [logistic loss
    forumula](https://en.wikipedia.org/wiki/Loss_functions_for_classification#Logistic_loss)
    (typically used in logistic regression).

*   `mean_squared_error(predictions, targets)`. Calculates loss using the [mean
    squared error](https://en.wikipedia.org/wiki/Mean_squared_error) (MSE; also
    known as L<sub>2</sub> loss).

The following example adds a definition for `loss` to the abalone `model_fn`
using `mean_squared_error()` (in bold):

<pre><code class="lang-python">def model_fn(features, targets, mode, params):
  """Model function for Estimator."""

  # Connect the first hidden layer to input layer
  # (features) with relu activation
  first_hidden_layer = tf.contrib.layers.relu(features, 10)

  # Connect the second hidden layer to first hidden layer with relu
  second_hidden_layer = tf.contrib.layers.relu(first_hidden_layer, 10)

  # Connect the output layer to second hidden layer (no activation fn)
  output_layer = tf.contrib.layers.linear(second_hidden_layer, 1)

  # Reshape output layer to 1-dim Tensor to return predictions
  predictions = tf.reshape(output_layer, [-1])
  predictions_dict = {"ages": predictions}

  <strong># Calculate loss using mean squared error
  loss = tf.contrib.losses.mean_squared_error(predictions, targets)</strong>
  ...</code></pre>

See the [API docs](../../api_docs/python/contrib.losses.md#losses-contrib) for
tf.contrib.loss for a full list of loss functions and more details on supported
arguments and usage.

### Defining the training op for the model

The training op defines the optimization algorithm TensorFlow will use when
fitting the model to the training data. Typically when training, the goal is to
minimize loss. The tf.contrib.layers API provides the function `optimize_loss`,
which returns a training op that will do just that. `optimize_loss` has four
required arguments:

*   `loss`. The loss value calculated by the `model_fn` (see [Defining Loss for
    the Model](#defining-loss)).
*   `global_step`. An integer
    [`Variable`](../../api_docs/python/state_ops.md#Variable) representing the
    step counter to increment for each model training run. Can easily be
    created/incremented in TensorFlow via the
    [`get_global_step()`](../../api_docs/python/contrib.framework.md#get_global_step)
    function.
*   `learning_rate`. The [learning
    rate](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Background)
    (also known as _step size_) hyperparameter that the optimization algorithm
    uses when training.
*   `optimizer`. The optimization algorithm to use during training. `optimizer`
    can accept any of the following string values, representing an optimization
    algorithm predefined in `tf.contrib.layers.optimizers`:
    *   `SGD`. Implementation of [gradient
        descent](https://en.wikipedia.org/wiki/Gradient_descent)
        ([tf.train.GradientDescentOptimizer](../../api_docs/python/train.md#GradientDescentOptimizer))
    *   `Adagrad`. Implementation of the [AdaGrad optimization
        algorithm](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
        ([tf.train.AdagradOptimizer](../../api_docs/python/train.md#AdagradOptimizer))
    *   `Adam`. Implementation of the [Adam optimization
        algorithm](http://arxiv.org/pdf/1412.6980.pdf)
        ([tf.train.AdamOptimizer](../../api_docs/python/train.md#AdamOptimizer))
    *   `Ftrl`. Implementation of the
        [FTRL-Proximal](https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf)
        ("Follow The (Proximally) Regularized Leader") algorithm
        ([tf.train.FtrlOptimizer](../../api_docs/python/train.md#FtrlOptimizer))
    *   `Momentum`. Implementation of stochastic gradient descent with
        [momentum](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum)
        ([tf.train.MomentumOptimizer](../../api_docs/python/train.md#MomentumOptimizer))
    *   `RMSProp`. Implementation of the
        [RMSprop](http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop)
        algorithm
        ([tf.train.RMSPropOptimizer](../../api_docs/python/train.md#RMSPropOptimizer))

NOTE: The `optimize_loss` function supports additional optional arguments to
further configure the optimizer, such as for implementing decay. See the [API
docs](../../api_docs/python/contrib.layers.md#optimize_loss) for more info.

The following code defines a training op for the abalone `model_fn`, using the
loss value calculated in [Defining Loss for the Model](#defining-loss), the
learning rate passed to the function in `params`, and the SGD optimizer. For
`global_step`, the convenience function
[`get_global_step()`](../../api_docs/python/contrib.framework.md#get_global_step)
in tf.contrib.framework takes care of generating an integer variable:

```python
train_op = tf.contrib.layers.optimize_loss(
    loss=loss,
    global_step=tf.contrib.framework.get_global_step(),
    learning_rate=params["learning_rate"],
    optimizer="SGD")
```

### The complete abalone `model_fn`

Here's the final, complete `model_fn` for the abalone age predictor. The
following code configures the neural network; defines loss and the training op;
and returns `predictions_dict`, `loss`, and `train_op`:

```python
def model_fn(features, targets, mode, params):
  """Model function for Estimator."""

  # Connect the first hidden layer to input layer
  # (features) with relu activation
  first_hidden_layer = tf.contrib.layers.relu(features, 10)

  # Connect the second hidden layer to first hidden layer with relu
  second_hidden_layer = tf.contrib.layers.relu(first_hidden_layer, 10)

  # Connect the output layer to second hidden layer (no activation fn)
  output_layer = tf.contrib.layers.linear(second_hidden_layer, 1)

  # Reshape output layer to 1-dim Tensor to return predictions
  predictions = tf.reshape(output_layer, [-1])
  predictions_dict = {"ages": predictions}

  # Calculate loss using mean squared error
  loss = tf.contrib.losses.mean_squared_error(predictions, targets)

  train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=params["learning_rate"],
      optimizer="SGD")

  return predictions_dict, loss, train_op
```

## Running the Abalone Model

You've instantiated an `Estimator` for the abalone predictor and defined its
behavior in `model_fn`; all that's left to do is train, evaluate, and make
predictions.

Add the following code to the end of `main()` to fit the neural network to the
training data and evaluate accuracy:

```python
# Fit
nn.fit(x=training_set.data, y=training_set.target, steps=5000)

# Score accuracy
ev = nn.evaluate(x=test_set.data, y=test_set.target, steps=1)
loss_score = ev["loss"]
print("Loss: %s" % loss_score)
```

Then run the code. You should see output like the following:

```none
...
INFO:tensorflow:loss = 4.86658, step = 4701
INFO:tensorflow:loss = 4.86191, step = 4801
INFO:tensorflow:loss = 4.85788, step = 4901
...
INFO:tensorflow:Saving evaluation summary for 5000 step: loss = 5.581
Loss: 5.581
```

The loss score reported is the mean squared error returned from the `model_fn`
when run on the `ABALONE_TEST` data set.

To predict ages for the `ABALONE_PREDICT` data set, add the following to
`main()`:

```python
# Print out predictions
predictions = nn.predict(x=prediction_set.data,
                         as_iterable=True)
for i, p in enumerate(predictions):
  print("Prediction %s: %s" % (i + 1, p["ages"]))
```

Here, the `predict()` function returns results in `predictions` as an iterable.
The `for` loop enumerates and prints out the results. Rerun the code, and you
should see output similar to the following:

```python
...
Prediction 1: 4.92229
Prediction 2: 10.3225
Prediction 3: 7.384
Prediction 4: 10.6264
Prediction 5: 11.0862
Prediction 6: 9.39239
Prediction 7: 11.1289
```

## Additional Resources

Congrats! You've successfully built a tf.contrib.learn `Estimator` from scratch.
For additional reference materials on building `Estimator`s, see the following
sections of the API docs:

*   [Estimators](../../api_docs/python/contrib.learn.md#estimators)
*   [Layers](../../api_docs/python/contrib.layers.md#layers-contrib)
*   [Losses](../../api_docs/python/contrib.losses.md#losses-contrib)
*   [Optimization](../../api_docs/python/contrib.layers.md#optimization)
