## Python API Tutorials

* [Profiling with tf.profiler](#profiling-with-tfprofiler)
* [Using TensorBoard Profiler](#using-tensorboard-profiler)
* [Programmatic Profiling](#programmatic-profiling)
* [Legacy API (TF1)](#legacy-api-tf1)

```python
import tensorflow as tf
```

### Profiling with tf.profiler

TensorFlow 2.x uses eager execution by default. To profile `tf.function`-decorated
functions, use `tf.profiler.experimental`:

```python
import tensorflow as tf

# Define a model or computation
@tf.function
def my_model(x):
    return tf.nn.relu(tf.matmul(x, weights) + bias)

# Start profiling
tf.profiler.experimental.start('logdir')

# Run your computation
for step in range(100):
    result = my_model(input_data)

# Stop profiling
tf.profiler.experimental.stop()
```

### Using TensorBoard Profiler

For interactive profiling, use the TensorBoard profiler callback with Keras:

```python
import tensorflow as tf
from tensorflow.keras import layers

# Create a simple model
model = tf.keras.Sequential([
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Create a TensorBoard callback with profiling
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs',
    profile_batch='10, 20'  # Profile batches 10-20
)

# Train with profiling
model.fit(
    train_data,
    epochs=5,
    callbacks=[tensorboard_callback]
)
```

Then visualize in TensorBoard:
```bash
tensorboard --logdir=./logs
```

### Programmatic Profiling

For fine-grained control, use the profiler context manager:

```python
import tensorflow as tf

# Profile a specific section of code
with tf.profiler.experimental.Profile('logdir'):
    # Your computation here
    for step in range(num_steps):
        train_step(data)
```

To capture a trace for Chrome tracing visualization:

```python
import tensorflow as tf

# Start tracing
tf.profiler.experimental.start('logdir')

# Run computation
result = my_function(input_data)

# Stop and save trace
tf.profiler.experimental.stop()

# The trace can be viewed in TensorBoard or by loading the
# trace.json file in chrome://tracing
```

### Trace Visualization

To visualize profiling results:

1. **TensorBoard (recommended):** Run `tensorboard --logdir=logdir` and navigate
   to the Profile tab.

2. **Chrome Tracing:** Open `chrome://tracing` in Chrome and load the generated
   `trace.json` file.

<left>
![CodeTimeline](graph_timeline.png)
![CodeTimeline](scope_timeline.png)
</left>

---

### Legacy API (TF1)

The following examples use the TensorFlow 1.x Session-based API. For new code,
prefer the TensorFlow 2.x APIs shown above.

#### Parameters and Shapes (TF1)
```python
# Print trainable variable parameter statistics to stdout.
ProfileOptionBuilder = tf.compat.v1.profiler.ProfileOptionBuilder

param_stats = tf.compat.v1.profiler.profile(
    tf.compat.v1.get_default_graph(),
    options=ProfileOptionBuilder.trainable_variables_parameter())

# Use code view to associate statistics with Python codes.
opts = ProfileOptionBuilder(
    ProfileOptionBuilder.trainable_variables_parameter()
    ).with_node_names(show_name_regexes=['.*my_code1.py.*', '.*my_code2.py.*']
    ).build()
param_stats = tf.compat.v1.profiler.profile(
    tf.compat.v1.get_default_graph(),
    cmd='code',
    options=opts)

# param_stats can be tensorflow.tfprof.GraphNodeProto or
# tensorflow.tfprof.MultiGraphNodeProto, depending on the view.
# Let's print the root below.
sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)
```

#### Float Operations (TF1)

See [Caveats](profile_model_architecture.md#caveats) in "Profile Model Architecture" Tutorial.

``` python
# Print to stdout an analysis of the number of floating point operations in the
# model broken down by individual operations.
tf.compat.v1.profiler.profile(
    tf.compat.v1.get_default_graph(),
    options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
```

#### Time and Memory (TF1)

You will first need to run the following set up in your model in order to
compute the memory and timing statistics.

```python
# Generate the RunMetadata that contains the memory and timing information.
#
# Note: When run on accelerator (e.g. GPU), an operation might perform some
#       cpu computation, enqueue the accelerator computation. The accelerator
#       computation is then run asynchronously. The profiler considers 3
#       times: 1) accelerator computation. 2) cpu computation (might wait on
#       accelerator). 3) the sum of 1 and 2.
#
run_metadata = tf.compat.v1.RunMetadata()
with tf.compat.v1.Session() as sess:
  _ = sess.run(train_op,
               options=tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE),
               run_metadata=run_metadata)
```

Finally, you may run `tf.compat.v1.profiler.profile` to explore the timing and memory
information of the model.

``` python
# Print to stdout an analysis of the memory usage and the timing information
# broken down by python codes.
ProfileOptionBuilder = tf.compat.v1.profiler.ProfileOptionBuilder
opts = ProfileOptionBuilder(ProfileOptionBuilder.time_and_memory()
    ).with_node_names(show_name_regexes=['.*my_code.py.*']).build()

tf.compat.v1.profiler.profile(
    tf.compat.v1.get_default_graph(),
    run_meta=run_metadata,
    cmd='code',
    options=opts)

# Print to stdout an analysis of the memory usage and the timing information
# broken down by operation types.
tf.compat.v1.profiler.profile(
    tf.compat.v1.get_default_graph(),
    run_meta=run_metadata,
    cmd='op',
    options=tf.compat.v1.profiler.ProfileOptionBuilder.time_and_memory())
```

#### Multi-step Profiling (TF1)

tfprof allows you to profile statistics across multiple steps.

```python
opts = model_analyzer.PRINT_ALL_TIMING_MEMORY.copy()
opts['account_type_regexes'] = ['.*']

with tf.compat.v1.Session() as sess:
  r1, r2, r3 = lib.BuildSplittableModel()
  sess.run(tf.compat.v1.global_variables_initializer())

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
