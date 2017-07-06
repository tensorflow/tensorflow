## Python API Tutorials

* [Parameters and Shapes](#parameters-and-shapes)
* [Float Operations](#float-operations)
* [Time and Memory](#time-and-memory)
* [Visualize](#visualize)
* [Multi-step Profiling](#multi-step-profiling)

```import tensorflow as tf```.

### Parameters and Shapes.
```python
# Print trainable variable parameter statistics to stdout.
param_stats = tf.profiler.profile(
    tf.get_default_graph(),
    options=tf.contrib.tfprof.model_analyzer.
        TRAINABLE_VARS_PARAMS_STAT_OPTIONS)

# Use code view to associate statistics with Python codes.
opts = tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS
opts['show_name_regexes'] = ['.*my_code1.py.*', '.*my_code2.py.*']
param_stats = tf.profiler.profile(
    tf.get_default_graph(),
    cmd='code'
    options=opts)

# param_stats can be tensorflow.tfprof.TFGraphNodeProto or
# tensorflow.tfprof.TFMultiGraphNodeProto, depending on the view.
# Let's print the root below.
sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
```

### Float Operations

#### Note: See [Caveats](profile_model_architecture.md#caveats) in "Profile Model Architecture" Tutorial
``` python
# Print to stdout an analysis of the number of floating point operations in the
# model broken down by individual operations.
tf.profiler.profile(
    tf.get_default_graph(),
    options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)
```

### Time and Memory
You will first need to run the following set up in your model in order to
compute the memory and timing statistics.

```python
# Generate the RunMetadata that contains the memory and timing information.
#
# Note: When run on GPU, a kernel is first scheduled (enqueued) and then
#       executed asynchronously. tfprof only tracks the execution time.
#
run_metadata = tf.RunMetadata()
with tf.Session() as sess:
  _ = sess.run(train_op,
               options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
               run_metadata=run_metadata)
```

Finally, you may run `print_model_analysis` to explore the timing and memory
information of the model.

``` python
# See model_analyzer_test.py for more examples.
#
# Print to stdout an analysis of the memory usage and the timing information
# broken down by python codes.
opts = tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY.copy()
opts['show_name_regexes'] = ['.*my_code.py.*']
tf.profiler.profile(
    tf.get_default_graph(),
    run_meta=run_metadata,
    cmd='code',
    options=opts)

# Print to stdout an analysis of the memory usage and the timing information
# broken down by operations.
tf.profiler.profile(
    tf.get_default_graph(),
    run_meta=run_metadata,
    options=tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY)
```

### Visualize

```
To visualize the result of Python API results:
Set opts['output'] = 'timeline:outfile=<filename>' to generate a timeline json file.
Open a Chrome Browser, open URL chrome://tracing, and load the json file.
```

Below are 2 examples of graph view and scope view. See code view example in later examples.

<left>
![CodeTimeline](graph_timeline.png)
![CodeTimeline](scope_timeline.png)
</left>

### Multi-step Profiling

tfprof allows you to profile statistics across multiple steps.

```python
opts = model_analyzer.PRINT_ALL_TIMING_MEMORY.copy()
opts['account_type_regexes'] = ['.*']

with session.Session() as sess:
  r1, r2, r3 = lib.BuildSplitableModel()
  sess.run(variables.global_variables_initializer())

  # Create a profiler.
  profiler = model_analyzer.Profiler(sess.graph)
  # Profile without RunMetadata of any step.
  pb0 = profiler.profile_name_scope(opts)

  run_meta = config_pb2.RunMetadata()
  _ = sess.run(r1,
               options=config_pb2.RunOptions(
                   trace_level=config_pb2.RunOptions.FULL_TRACE),
               run_metadata=run_meta)

  # Add run_meta of step 1.
  profiler.add_step(1, run_meta)
  pb1 = profiler.profile_name_scope(opts)

  run_meta2 = config_pb2.RunMetadata()
  _ = sess.run(r2,
               options=config_pb2.RunOptions(
                   trace_level=config_pb2.RunOptions.FULL_TRACE),
               run_metadata=run_meta2)
  # Add run_meta of step 2.
  profiler.add_step(2, run_meta2)
  pb2 = profiler.profile_name_scope(opts)

  run_meta3 = config_pb2.RunMetadata()
  _ = sess.run(r3,
               options=config_pb2.RunOptions(
                   trace_level=config_pb2.RunOptions.FULL_TRACE),
               run_metadata=run_meta3)
  # Add run_meta of step 3.
  profiler.add_step(3, run_meta3)
  pb3 = profiler.profile_name_scope(opts)
```