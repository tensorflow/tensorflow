# How to Use TensorFlow Debugger (tfdbg) with tf.contrib.learn

[TOC]

In [a previous tutorial](index.md), we described how to use TensorFlow Debugger (**tfdbg**)
to debug TensorFlow graphs running in
[`tf.Session`](https://tensorflow.org/api_docs/python/client.html#Session)
objects managed by yourself. However, many users find
[`tf.contrib.learn`](https://tensorflow.org/tutorials/tflearn/index.html)
[Estimator](https://tensorflow.org/api_docs/python/contrib.learn.html?cl=head#Estimator)s
to be a convenient higher-level API for creating and using models
in TensorFlow. Part of the convenience is that `Estimator`s manage `Session`s
internally. Fortunately, you can still use `tfdbg` with `Estimator`s by adding
special hooks.

## Debugging tf.contrib.learn Estimators

Currently, **tfdbg** can debug the
[fit()](https://tensorflow.org/api_docs/python/contrib.learn.html#BaseEstimator.fit)
and
[evaluate()](https://tensorflow.org/api_docs/python/contrib.learn.html#BaseEstimator.evaluate)
methods of tf-learn `Estimator`s. To debug `Estimator.fit()`,
create a `LocalCLIDebugHook` and supply it as the `monitors` argument. For example:

```python
# First, let your BUILD target depend on "//tensorflow/python/debug:debug_py"
from tensorflow.python import debug as tf_debug

hooks = [tf_debug.LocalCLIDebugHook()]

# Create a local CLI debug hook and use it as a monitor when calling fit().
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=1000,
               monitors=hooks)
```

To debug `Estimator.evaluate()`, you can follow the example below:

```python
accuracy_score = classifier.evaluate(x=test_set.data,
                                     y=test_set.target,
                                     hooks=hooks)["accuracy"]
```


For a detailed [example](https://www.tensorflow.org/code/tensorflow/python/debug/examples/debug_tflearn_iris.py) based on
[tf-learn's iris tutorial](../../../g3doc/tutorials/tflearn/index.md),
run:

```none
python $(python -c "import tensorflow as tf; import os; print(os.path.dirname(tf.__file__));")/python/debug/examples/debug_tflearn_iris.py --debug
```

## Debugging tf.contrib.learn Experiments

`Experiment` is a construct in `tf.contrib.learn` at a higher level than
`Estimator`.
It provides a single interface for training and evaluating a model. To debug
the `train()` and `evaluate()` calls to an `Experiment` object, you can
use the keyword arguments `train_monitors` and `eval_hooks`, respectively, when
calling its constructor. For example:

```python
# First, let your BUILD target depend on "//tensorflow/python/debug:debug_py"
from tensorflow.python import debug as tf_debug

hooks = [tf_debug.LocalCLIDebugHook()]

ex = experiment.Experiment(classifier,
                           train_input_fn=iris_input_fn,
                           eval_input_fn=iris_input_fn,
                           train_steps=FLAGS.train_steps,
                           eval_delay_secs=0,
                           eval_steps=1,
                           train_monitors=hooks,
                           eval_hooks=hooks)

ex.train()
accuracy_score = ex.evaluate()["accuracy"]
```

To see the `debug_tflearn_iris` example run in the `Experiment` mode, do:

```none
python $(python -c "import tensorflow as tf; import os; print(os.path.dirname(tf.__file__));")/python/debug/examples/debug_tflearn_iris.py \
    --use_experiment --debug
```
