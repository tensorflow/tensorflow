# How to Use TensorFlow Debugger (tfdbg) with tf.contrib.learn

[TOC]

In @{$debugger$a previous tutorial}, we described how to use TensorFlow Debugger (**tfdbg**)
to debug TensorFlow graphs running in
@{tf.Session}
objects managed by yourself. However, many users find
@{$tflearn$`tf.contrib.learn`}
@{tf.contrib.learn.Estimator$Estimator}s
to be a convenient higher-level API for creating and using models
in TensorFlow. Part of the convenience is that `Estimator`s manage `Session`s
internally. Fortunately, you can still use `tfdbg` with `Estimator`s by adding
special hooks.

## Debugging tf.contrib.learn Estimators

Currently, **tfdbg** can debug the
@{tf.contrib.learn.BaseEstimator.fit$`fit()`}
@{tf.contrib.learn.BaseEstimator.evaluate$`evaluate()`}
methods of tf-learn `Estimator`s. To debug `Estimator.fit()`,
create a `LocalCLIDebugHook` and supply it as the `monitors` argument. For example:

```python
# First, let your BUILD target depend on "//tensorflow/python/debug:debug_py"
# (You don't need to worry about the BUILD dependency if you are using a pip
#  install of open-source TensorFlow.)
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
@{$tflearn$tf-learn's iris tutorial},
run:

```none
python -m tensorflow.python.debug.examples.debug_tflearn_iris --debug
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
# (You don't need to worry about the BUILD dependency if you are using a pip
#  install of open-source TensorFlow.)
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
python -m tensorflow.python.debug.examples.debug_tflearn_iris \
    --use_experiment --debug
```

## Debugging Estimators and Experiments without Terminal Access

If your `Estimator` or `Experiment` is running in an environment to which you
do not have command-line access (e.g., a remote server), you can use the
non-interactive `DumpingDebugHook`. For example:

```python
# Let your BUILD target depend on "//tensorflow/python/debug:debug_py
# (You don't need to worry about the BUILD dependency if you are using a pip
#  install of open-source TensorFlow.)
from tensorflow.python import debug as tf_debug

hooks = [tf_debug.DumpingDebugHook("/shared/storage/location/tfdbg_dumps_1")]
```

Then this `hook` can be used in the same way as the `LocalCLIDebugHook` examples
above. As the training and/or evalution of `Estimator` or `Experiment`
happens, directories of the naming pattern
`/shared/storage/location/tfdbg_dumps_1/run_<epoch_timestamp_microsec>_<uuid>`
will appear. Each directory corresponds to a `Session.run()` call that underlies
the `fit()` or `evaluate()` call. You can load these directories and inspect
them in a command-line interface in an offline manner using the
`offline_analyzer` offered by **tfdbg**. For example:

```bash
python -m tensorflow.python.debug.cli.offline_analyzer \
    --dump_dir="/shared/storage/location/tfdbg_dumps_1/run_<epoch_timestamp_microsec>_<uuid>"
```

The `LocalCLIDebugHook` also allows you to configure a `watch_fn` that can be
used to flexibly specify what `Tensor`s to watch on different `Session.run()`
calls, as a function of the `fetches` and `feed_dict` and other states. See
@{tfdbg.DumpingDebugWrapperSession.__init__$this API doc}
for more details.
