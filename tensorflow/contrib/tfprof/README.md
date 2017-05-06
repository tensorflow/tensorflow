# tfprof: A Profiling Tool for TensorFlow Models

# Full Docment in tensorflow/tools/tfprof/README.md

Author: Xin Pan (xpan@google.com, github: panyx0718)

Consultants: Jon Shlens, Pete Warden

###Major Features

1.  Measure model parameters, float operations, tensor shapes.
2.  Measure op execution times, requested memory size and device placement.
3.  Inspect checkpoint tensors' shapes and their values.
4.  Explore model based on name scope or graph structure.
5.  Selectively grouping/filtering/accounting/ordering ops.

tfprof can be used as Python API, Interactive CLI and One-shot Script.

## Python API Tutorials

tfprof is part of TensorFlow core. Simply ```import tensorflow as tf```.

### Examine the shapes and sizes of all trainiable Variables.
```python
# Print trainable variable parameter statistics to stdout.
param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
    tf.get_default_graph(),
    tfprof_options=tf.contrib.tfprof.model_analyzer.
        TRAINABLE_VARS_PARAMS_STAT_OPTIONS)

# param_stats is tensorflow.tfprof.TFProfNode proto. It organize the statistics
# of each graph node in tree scructure. Let's print the root below.
sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
```

### Examine the number of floating point operations
``` python
# Print to stdout an analysis of the number of floating point operations in the
# model broken down by individual operations.
#
# Note: Only Ops with RegisterStatistics('flops') defined have flop stats. It
# also requires complete shape information. It is common that shape is unknown
# statically. To complete the shape, provide run-time shape information with
# tf.RunMetadata to the API (See next example on how to provide RunMetadata).
tf.contrib.tfprof.model_analyzer.print_model_analysis(
    tf.get_default_graph(),
    tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)
```

### Examine the timing and memory usage
You will first need to run the following set up in your model in order to
compute the memory and timing statistics.

```python
# Generate the meta information for the model that contains the memory usage
# and timing information.
run_metadata = tf.RunMetadata()
with tf.Session() as sess:
  _ = sess.run(train_op,
               options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
               run_metadata=run_metadata)
```

Finally, you may run `print_model_analysis` to explore the timing and memory
demands of the model.

``` python
# Print to stdout an analysis of the memory usage and the timing information
# from running the graph broken down by operations.
tf.contrib.tfprof.model_analyzer.print_model_analysis(
    tf.get_default_graph(),
    run_meta=run_metadata,
    tfprof_options=tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY)
```

Users can change ```tfprof_options``` to fully leverage tfprof's power.
