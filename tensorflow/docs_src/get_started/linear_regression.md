# Regression Examples

This unit provides the following short examples demonstrating how
to implement regression in Estimators:

<table>
  <tr> <th>Example</th> <th>Data Set</th> <th>Demonstrates How To...</th></tr>

  <tr>
    <td><a href="https://www.tensorflow.org/code/tensorflow/examples/get_started/regression/linear_regression.py">linear_regression.py</a></td>
    <td>[imports85](https://archive.ics.uci.edu/ml/datasets/automobile)</td>
    <td>Use the @{tf.estimator.LinearRegressor} Estimator to train a
        regression model on numeric data.</td>
  </tr>

  <tr>
    <td><a href="https://www.tensorflow.org/code/tensorflow/examples/get_started/regression/linear_regression_categorical.py">linear_regression_categorical.py</a></td>
    <td>[imports85](https://archive.ics.uci.edu/ml/datasets/automobile)</td>
    <td>Use the @{tf.estimator.LinearRegressor} Estimator to train a
        regression model on categorical data.</td>
  </tr>

  <tr>
    <td><a href="https://www.tensorflow.org/code/tensorflow/examples/get_started/regression/dnn_regression.py">dnn_regression.py</a></td>
    <td>[imports85](https://archive.ics.uci.edu/ml/datasets/automobile)</td>
    <td>Use the @{tf.estimator.DNNRegressor} Estimator to train a
        regression model on discrete data with a deep neural network.</td>
  </tr>

</table>

The preceding examples rely on the following data set utility:

<table>
  <tr> <th>Utility</th> <th>Description</th></tr>

  <tr>
    <td><a href="../../examples/get_started/regression/imports85.py">imports85.py</a></td>
    <td>This program provides utility functions that load the
        <tt>imports85</tt> data set into formats that other TensorFlow
        programs (for example, <tt>linear_regression.py</tt> and
        <tt>dnn_regression.py</tt>) can use.</td>
  </tr>


</table>


<!--
## Linear regression concepts

If you are new to machine learning and want to learn about regression,
watch the following video:

(todo:jbgordon) Video introduction goes here.
-->

<!--
[When MLCC becomes available externally, add links to the relevant MLCC units.]
-->


<a name="running"></a>
## Running the examples

You must @{$install$install TensorFlow} prior to running these examples.
Depending on the way you've installed TensorFlow, you might also
need to activate your TensorFlow environment.  Then, do the following:

1. Clone the TensorFlow repository from github.
2. `cd` to the top of the downloaded tree.
3. Check out the branch for you current tensorflow version: `git checkout rX.X`
4. `cd tensorflow/examples/get_started/regression`.

You can now run any of the example TensorFlow programs in the
`tensorflow/examples/get_started/regression` directory as you
would run any Python program:

```bsh
python linear_regressor.py
```

During training, all three programs output the following information:

* The name of the checkpoint directory, which is important for TensorBoard.
* The training loss after every 100 iterations, which helps you
  determine whether the model is converging.

For example, here's some possible output for the `linear_regressor.py`
program:

```bsh
INFO:tensorflow:Saving checkpoints for 1 into /tmp/tmpAObiz9/model.ckpt.
INFO:tensorflow:loss = 161.308, step = 1
INFO:tensorflow:global_step/sec: 1557.24
INFO:tensorflow:loss = 15.7937, step = 101 (0.065 sec)
INFO:tensorflow:global_step/sec: 1529.17
INFO:tensorflow:loss = 12.1988, step = 201 (0.065 sec)
INFO:tensorflow:global_step/sec: 1663.86
...
INFO:tensorflow:loss = 6.99378, step = 901 (0.058 sec)
INFO:tensorflow:Saving checkpoints for 1000 into /tmp/tmpAObiz9/model.ckpt.
INFO:tensorflow:Loss for final step: 5.12413.
```


<a name="basic"></a>
## linear_regressor.py

`linear_regressor.py` trains a model that predicts the price of a car from
two numerical features.

<table>
  <tr>
    <td>Estimator</td>
    <td><tt>LinearRegressor</tt>, which is a pre-made Estimator for linear
        regression.</td>
  </tr>

  <tr>
    <td>Features</td>
    <td>Numerical: <tt>body-style</tt> and <tt>make</tt>.</td>
  </tr>

  <tr>
    <td>Label</td>
    <td>Numerical: <tt>price</tt>
  </tr>

  <tr>
    <td>Algorithm</td>
    <td>Linear regression.</td>
  </tr>
</table>

After training the model, the program concludes by outputting predicted
car prices for two car models.



<a name="categorical"></a>
## linear_regression_categorical.py

This program illustrates ways to represent categorical features. It
also demonstrates how to train a linear model based on a mix of
categorical and numerical features.

<table>
  <tr>
    <td>Estimator</td>
    <td><tt>LinearRegressor</tt>, which is a pre-made Estimator for linear
        regression. </td>
  </tr>

  <tr>
    <td>Features</td>
    <td>Categorical: <tt>curb-weight</tt> and <tt>highway-mpg</tt>.<br/>
        Numerical: <tt>body-style</tt> and <tt>make</tt>.</td>
  </tr>

  <tr>
    <td>Label</td>
    <td>Numerical: <tt>price</tt>.</td>
  </tr>

  <tr>
    <td>Algorithm</td>
    <td>Linear regression.</td>
  </tr>
</table>


<a name="dnn"></a>
## dnn_regression.py

Like `linear_regression_categorical.py`, the `dnn_regression.py` example
trains a model that predicts the price of a car from two features.
Unlike `linear_regression_categorical.py`, the `dnn_regression.py` example uses
a deep neural network to train the model.  Both examples rely on the same
features; `dnn_regression.py` demonstrates how to treat categorical features
in a deep neural network.

<table>
  <tr>
    <td>Estimator</td>
    <td><tt>DNNRegressor</tt>, which is a pre-made Estimator for
        regression that relies on a deep neural network.  The
        `hidden_units` parameter defines the topography of the network.</td>
  </tr>

  <tr>
    <td>Features</td>
    <td>Categorical: <tt>curb-weight</tt> and <tt>highway-mpg</tt>.<br/>
        Numerical: <tt>body-style</tt> and <tt>make</tt>.</td>
  </tr>

  <tr>
    <td>Label</td>
    <td>Numerical: <tt>price</tt>.</td>
  </tr>

  <tr>
    <td>Algorithm</td>
    <td>Regression through a deep neural network.</td>
  </tr>
</table>

After printing loss values, the program outputs the Mean Square Error
on a test set.
