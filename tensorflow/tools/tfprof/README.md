# tfprof: A Profiling Tool for TensorFlow Models

Author: Xin Pan (xpan@google.com, github: panyx0718), Jon Shlens, Yao Zhang

Consultants: Jon Shlens, Pete Warden


###Major Features

1.  Measure model parameters, float operations, tensor shapes.
2.  Profile op execution times, requested memory size and device placement.
3.  Inspect checkpoint tensors' shapes and their values.
4.  Selectively group, filter, account and order ops.

####tfprof supports 4 views to organize TensorFlow model profiles

    *  code view: graph nodes are grouped by Python codes that generate them.
    *  op view: graph nodes are grouped by operation type (E.g. MatMul, Conv2D) of the graph nodes.
    *  scope view: graph nodes are organized based on name scope hierarchies.
    *  graph view: graph nodes are organized based on op input/output.

####For each view, there are 3 ways to display outputs:

    *  stdout: Results are written to stdout.
    *  timeline: Visualized in chrome browser as time series.
    *  file: Results are dumped to file.


[Demo](#demo)

[Python API Tutorials](#python-api-tutorials): How to use directly from
Python codes.

[CLI Tutorials](#cli-tutorials): How to run from interactive command line.

[Options](#options):
tfprof supports many options to selectively account/display/order ops and
statistics.


## Demo
### Attribute the TensorFlow graph running time to your Python codes.
```shell
tfprof> code -max_depth 1000 -show_name_regexes .*model_analyzer.*py.* -select micros -account_type_regexes .* -order_by micros
_TFProfRoot (0us/22.44ms)
  model_analyzer_test.py:149:run_filename_as_m...:none (0us/22.44ms)
    model_analyzer_test.py:33:_run_code_in_main:none (0us/22.44ms)
      model_analyzer_test.py:208:<module>:test.main() (0us/22.44ms)
        model_analyzer_test.py:132:testComplexCodeView:x = lib.BuildFull... (0us/22.44ms)
          model_analyzer_testlib.py:63:BuildFullModel:return sgd_op.min... (0us/21.83ms)
          model_analyzer_testlib.py:58:BuildFullModel:cell, array_ops.c... (0us/333us)
          model_analyzer_testlib.py:54:BuildFullModel:seq.append(array_... (0us/254us)
            model_analyzer_testlib.py:42:BuildSmallModel:x = nn_ops.conv2d... (0us/134us)
            model_analyzer_testlib.py:46:BuildSmallModel:initializer=init_... (0us/40us)
            ...
          model_analyzer_testlib.py:61:BuildFullModel:loss = nn_ops.l2_... (0us/28us)
          model_analyzer_testlib.py:60:BuildFullModel:target = array_op... (0us/0us)
        model_analyzer_test.py:134:testComplexCodeView:sess.run(variable... (0us/0us)
```

### Show your model variables and the number of parameters.
```
tfprof> scope -account_type_regexes VariableV2 -max_depth 4 -select params
_TFProfRoot (--/930.58k params)
  global_step (1/1 params)
  init/init_conv/DW (3x3x3x16, 432/864 params)
  pool_logit/DW (64x10, 640/1.28k params)
    pool_logit/DW/Momentum (64x10, 640/640 params)
  pool_logit/biases (10, 10/20 params)
    pool_logit/biases/Momentum (10, 10/10 params)
  unit_last/final_bn/beta (64, 64/128 params)
  unit_last/final_bn/gamma (64, 64/128 params)
  unit_last/final_bn/moving_mean (64, 64/64 params)
  unit_last/final_bn/moving_variance (64, 64/64 params)
```

### Show the most expensive operation types.
```
tfprof> op -select micros,bytes,occurrence -order_by micros
SoftmaxCrossEntropyWithLogits      36.58MB (100.00%, 0.05%),      1.37sec (100.00%, 23.56%),         30
MatMul                        2720.57MB (99.95%, 3.66%),      988.90ms (76.44%, 17.05%),       3450
ConcatV2                       741.37MB (96.29%, 1.00%),       421.44ms (59.38%, 7.27%),       6098
Mul                           3957.24MB (95.29%, 5.33%),       418.90ms (52.12%, 7.22%),       9427
Add                            740.05MB (89.96%, 1.00%),       335.26ms (44.89%, 5.78%),       2180
Sub                             32.46MB (88.97%, 0.04%),       216.44ms (39.11%, 3.73%),       4372
AddN                           733.21MB (88.92%, 0.99%),       208.46ms (35.38%, 3.59%),       5481
Slice                          708.07MB (87.94%, 0.95%),       205.27ms (31.78%, 3.54%),       7277
Fill                           954.27MB (86.98%, 1.28%),       154.50ms (28.24%, 2.66%),       9686
Select                         312.33MB (85.70%, 0.42%),       123.04ms (25.58%, 2.12%),       5746
Sigmoid                        152.57MB (85.28%, 0.21%),        96.66ms (23.46%, 1.67%),       2970
```

### Visualize time and memory.
<left>
[CodeTimeline](g3doc/graph_timeline.png)
</left>

## Python API Tutorials

tfprof is part of TensorFlow core. Simply ```import tensorflow as tf```.

### Examine the shapes and sizes of all trainable Variables.
```python
# Print trainable variable parameter statistics to stdout.
# By default, statistics are associated with each graph node.
param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
    tf.get_default_graph(),
    tfprof_options=tf.contrib.tfprof.model_analyzer.
        TRAINABLE_VARS_PARAMS_STAT_OPTIONS)


# Set tfprof_cmd='code' to associate statistics with Python codes.
opts = tf.contrib.tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS
opts['show_name_regexes'] = ['.*my_code1.py.*', '.*my_code2.py.*']
param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
    tf.get_default_graph(),
    tfprof_cmd='code'
    tfprof_options=opts)

# param_stats is tensorflow.tfprof.TFGraphNodeProto proto.
# Let's print the root below.
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
#
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
#
# Note: When run on GPU, a kernel is first scheduled (enqueued) and then
#       executed asynchronously. tfprof only tracks the execution time.
#       In addition, a substantial of time might be spent between Python and
#       TensorFlow runtime, which is also not tracked by tfprof.
#
run_metadata = tf.RunMetadata()
with tf.Session() as sess:
  _ = sess.run(train_op,
               options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
               run_metadata=run_metadata)
```

Finally, you may run `print_model_analysis` to explore the timing and memory
demands of the model.

``` python
# See model_analyzer_test.py for more examples.
#
# Print to stdout an analysis of the memory usage and the timing information
# broken down by python codes.
opts = tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY.copy()
opts['show_name_regexes'] = ['.*my_code.py.*']
tf.contrib.tfprof.model_analyzer.print_model_analysis(
    tf.get_default_graph(),
    run_meta=run_metadata,
    tfprof_cmd='code',
    tfprof_options=opts)

# Print to stdout an analysis of the memory usage and the timing information
# broken down by operations.
tf.contrib.tfprof.model_analyzer.print_model_analysis(
    tf.get_default_graph(),
    run_meta=run_metadata,
    tfprof_options=tf.contrib.tfprof.model_analyzer.PRINT_ALL_TIMING_MEMORY)
```

### Visualize

```
For example set opts['output'] = 'timeline:outfile=<filename>' to
generate a timeline json file. Open a Chrome Browser, open URL
chrome://tracing, and load the json file. Below are 2 examples of graph
view and scope view. See code view example in later examples.
```

<left>
[CodeTimeline](g3doc/graph_timeline.png)
[CodeTimeline](g3doc/scope_timeline.png)
</left>


## CLI Tutorials

Tutorials below are based on a 32 layers ResNet.

TODO(xpan): Provide graph.pbtxt, model.ckpt, tfprof_log and run_meta download.

### Examples

1) Start `tfprof` command line tool

```shell
# Build the tool.
bazel build --config opt tensorflow/tools/tfprof/...

# Help information, including detail 'option' instructions.
bazel-bin/tensorflow/tools/tfprof/tfprof help
#
# The following commands will start tfprof interactive mode.
#
# Profile model shapes and parameters only.
bazel-bin/tensorflow/tools/tfprof/tfprof \
    --graph_path=graph.pbtxt
#
# Additionally profile ops requested memory and timing.
# See CLI Input Files section on generating run_meta file.
bazel-bin/tensorflow/tools/tfprof/tfprof \
    --graph_path=graph.pbtxt \
    --run_meta_path=run_meta \
#
# Additionally profile checkpoint statistics and values.
# Use '-account_type_regexes _checkpoint_variables' to select
# checkpoint tensors.
bazel-bin/tensorflow/tools/tfprof/tfprof \
    --graph_path=graph.pbtxt \
    --run_meta_path=run_meta \
    --checkpoint_path=model.ckpt
#
# tfprof_log is used to define customized op types, float ops and code traces.
# Use tfprof_logger.write_op_log() to create tfprof_log.
# See 12) in Examples section on generating tfprof_log file.
bazel-bin/tensorflow/tools/tfprof/tfprof \
    --graph_path=graph.pbtxt \
    --run_meta_path=run_meta \
    --op_log_path=tfprof_log \
    --checkpoint_path=model.ckpt
#
# The following command start tfprof in one-shot mode.
#
bazel-bin/tensorflow/tools/tfprof/tfprof scope \
    --graph_path=graph.pbtxt \
    --max_depth=3
```
Note that `graph.pbtxt` is an ASCII text format.

2) Press enter to show the default options

```shell
tfprof>
-max_depth                  4
-min_bytes                  0
-min_micros                 0
-min_params                 0
-min_float_ops              0
-min_occurrence             0
-step                       -1
-order_by                   name
-account_type_regexes       Variable,VariableV2
-start_name_regexes         .*
-trim_name_regexes
-show_name_regexes          .*
-hide_name_regexes          IsVariableInitialized_[0-9]+,save\/.*,^zeros[0-9_]*
-account_displayed_op_only  false
# supported select fields. Availability depends on --[run_meta|checkpoint|op_log]_path.
# [bytes|micros|params|float_ops|occurrence|tensor_value|device|op_types]
-select                     params
# format: output_type:key=value,key=value...
# output_types: stdout (default), timeline, file.
# key=value pairs:
#   1. timeline: outfile=<filename>
#   2. file: outfile=<filename>
#   3. stdout: None.
# E.g. timeline:outfile=/tmp/timeline.json
-output
```

3) I want to see which line of my python codes costs most time!

```shell
# Requires --graph_path --op_log_path
tfprof> code -max_depth 1000 -show_name_regexes .*model_analyzer.*py.* -select micros -account_type_regexes .* -order_by micros
_TFProfRoot (0us/22.44ms)
  model_analyzer_test.py:149:run_filename_as_m...:none (0us/22.44ms)
    model_analyzer_test.py:33:_run_code_in_main:none (0us/22.44ms)
      model_analyzer_test.py:208:<module>:test.main() (0us/22.44ms)
        model_analyzer_test.py:132:testComplexCodeView:x = lib.BuildFull... (0us/22.44ms)
          model_analyzer_testlib.py:63:BuildFullModel:return sgd_op.min... (0us/21.83ms)
          model_analyzer_testlib.py:58:BuildFullModel:cell, array_ops.c... (0us/333us)
          model_analyzer_testlib.py:54:BuildFullModel:seq.append(array_... (0us/254us)
            model_analyzer_testlib.py:42:BuildSmallModel:x = nn_ops.conv2d... (0us/134us)
            model_analyzer_testlib.py:46:BuildSmallModel:initializer=init_... (0us/40us)
            ...
          model_analyzer_testlib.py:61:BuildFullModel:loss = nn_ops.l2_... (0us/28us)
          model_analyzer_testlib.py:60:BuildFullModel:target = array_op... (0us/0us)
        model_analyzer_test.py:134:testComplexCodeView:sess.run(variable... (0us/0us)
```

Set ```-output timeline:outfile=<filename>``` to generate timeline instead of stdout.
<left>
[CodeTimeline](g3doc/code_timeline.png)
</left>


4) I want to see the `BatchNorm`'s gamma value in checkpoint.

```shell
# Requires --graph_path, --checkpoint_path.
tfprof> scope -show_name_regexes unit_1_0.*gamma -select tensor_value -max_depth 5
_TFProfRoot ()
  unit_1_0/shared_activation/init_bn/gamma ()
[1.80 2.10 2.06 1.91 2.26 1.86 1.81 1.37 1.78 1.85 1.96 1.54 2.04 2.34 2.22 1.99 ],
  unit_1_0/sub2/bn2/gamma ()
[1.57 1.83 1.30 1.25 1.59 1.14 1.26 0.82 1.19 1.10 1.48 1.01 0.82 1.23 1.21 1.14 ],
```

5) I want to see my checkpoint tensors shape and number of parameters.

```shell
# Requires --graph_path, --checkpoint_path.
# Increase -max_depth to see all tensors.
tfprof> scope -account_type_regexes _checkpoint_variables -select params -max_depth 4
_TFProfRoot (--/930.58k params)
  global_step (0/0 params)
  init/init_conv/DW (3x3x3x16, 432/864 params)
  pool_logit/DW (64x10, 640/1.28k params)
    pool_logit/DW/Momentum (64x10, 640/640 params)
  pool_logit/biases (10, 10/20 params)
    pool_logit/biases/Momentum (10, 10/10 params)
  unit_last/final_bn/beta (64, 64/128 params)
  unit_last/final_bn/gamma (64, 64/128 params)
  unit_last/final_bn/moving_mean (64, 64/64 params)
  unit_last/final_bn/moving_variance (64, 64/64 params)
```

6) I defined an op named ‘cost’ to calculate the loss. I want to know what ops
it depends on take a long time to run. Hint: Use the ‘graph’ command to explore
graph dependencies.

```shell
# Requires --graph_path, --run_meta_path.
tfprof> graph -start_name_regexes cost.* -max_depth 100 -min_micros 10000 -select micros -account_type_regexes .*
_TFProfRoot (0us/3.61sec)
  init/init_conv/Conv2D (11.75ms/3.10sec)
    random_shuffle_queue_DequeueMany (3.09sec/3.09sec)
  unit_1_0/sub2/conv2/Conv2D (74.14ms/3.19sec)
  unit_1_3/sub2/conv2/Conv2D (60.75ms/3.34sec)
  unit_2_4/sub2/conv2/Conv2D (73.58ms/3.54sec)
  unit_3_3/sub2/conv2/Conv2D (10.26ms/3.60sec)
```

7) I want to know the expensive operations during the back propagation.
Hint: tensorflow prepend ‘gradient’ to your defined name scopes. Use the ‘scope’
command to explore based on name scope hierarchies.

```shell
# Requires --graph_path, --run_meta_path.
tfprof> scope -start_name_regexes gradient.* -max_depth 100 -min_micros 20000 -select micros -account_type_regexes .*
_TFProfRoot (0us/2.29sec)
  gradients/unit_1_0/sub1/conv1/Conv2D_grad/Conv2DBackpropFilter (54.96ms/54.96ms)
  gradients/unit_1_0/sub2/conv2/Conv2D_grad/Conv2DBackpropFilter (83.63ms/83.63ms)
  gradients/unit_1_1/sub1/conv1/Conv2D_grad/Conv2DBackpropFilter (99.25ms/99.25ms)
  gradients/unit_1_2/sub1/conv1/Conv2D_grad/Conv2DBackpropFilter (95.40ms/95.40ms)
  gradients/unit_1_2/sub2/conv2/Conv2D_grad/Conv2DBackpropFilter (99.83ms/99.83ms)
  gradients/unit_1_3/sub1/conv1/Conv2D_grad/Conv2DBackpropFilter (95.39ms/95.39ms)
  ...
```

8) Show the number of float operations in the model.
Note: float operations calculation depends on
1) op.RegisterStatistics. If an op doesn’t
have RegisterStatistics defined, its float operations cannot be counted.
2) fully defined shape is also necessary in order to calculate flops. Sometimes
full shape is not available statically. Use RunMetadata to get run-time shape.
float operations number is provided by tensorflow::tfprof::OpLog logged from
Python API.

```shell
# Requires --graph_path, --op_log_path.
tfprof> scope -min_float_ops 1 -max_depth 10 -select float_ops -account_type_regexes .*
_TFProfRoot (0/17.63b flops)
  gradients/pool_logit/xw_plus_b/MatMul_grad/MatMul (163.84k/163.84k flops)
  gradients/pool_logit/xw_plus_b/MatMul_grad/MatMul_1 (163.84k/163.84k flops)
  init/init_conv/Conv2D (113.25m/113.25m flops)
  pool_logit/xw_plus_b (1.28k/165.12k flops)
    pool_logit/xw_plus_b/MatMul (163.84k/163.84k flops)
  unit_1_0/sub1/conv1/Conv2D (603.98m/603.98m flops)
  unit_1_0/sub2/conv2/Conv2D (603.98m/603.98m flops)
  unit_1_1/sub1/conv1/Conv2D (603.98m/603.98m flops)
  unit_1_1/sub2/conv2/Conv2D (603.98m/603.98m flops)
  ...
```

9) Show the number of parameters of all `tf.trainable_variables()` in the model.

```shell
# Requires --graph_path --op_log_path.
# store option for future commands.
tfprof> set -account_type_regexes _trainable_variables
tfprof> scope -max_depth 4 -select params
_TFProfRoot (--/464.15k params)
  init/init_conv/DW (3x3x3x16, 432/432 params)
  pool_logit/DW (64x10, 640/640 params)
  pool_logit/biases (10, 10/10 params)
  unit_last/final_bn/beta (64, 64/64 params)
  unit_last/final_bn/gamma (64, 64/64 params)
```

Where does “_trainable_variables” come from? It is from the OpLog file
generated by write_op_log() Python API. write_op_log() help users create some
common op types implicitly. Users can define their own op types and log it
through the write_op_log() API.

109) What if I’m lazy and don’t want to define op type? I have given my ops
well-defined names in my model’s code. And want to use names to select a group
of ops. Let’s try it!

```shell
tfprof> set -account_type_regexes .*
tfprof> scope -show_name_regexes unit_2_1.*DW -max_depth 100 -account_displayed_op_only
_TFProfRoot (0/18.43k params)
  unit_2_1/sub1/conv1/DW (3x3x32x32, 9.22k/9.22k params)
  unit_2_1/sub2/conv2/DW (3x3x32x32, 9.22k/9.22k params)
```

The above command allows you to filter ops that match specific names.
`-account_displayed_op_only` asks tfprof to only account ops displayed
in terminal. Otherwise, tfprof accounts all ops matched by
`-account_type_regexes` recursively even if they are hidden due to some
options such as -max_depth.

11) TensorFlow has built-in op types. For example, built-in op type `Variable`
seems to include `Variable's` created by your model. However, be careful when
depending on it because TensorFlow creates extra `Variable` ops implicitly and
the implicitly created ops can have the same prefix as the `Variable's` you
defined.

In the following example, extra `Variables` are created and “/Momentum” is
appended to their names. This might cause you “model capacity” calculation
to get wrong.

```shell
tfprof> scope -account_type_regexes VariableV2 -max_depth 4 -select params
_TFProfRoot (--/930.58k params)
  global_step (1/1 params)
  init/init_conv/DW (3x3x3x16, 432/864 params)
  pool_logit/DW (64x10, 640/1.28k params)
    pool_logit/DW/Momentum (64x10, 640/640 params)
  pool_logit/biases (10, 10/20 params)
    pool_logit/biases/Momentum (10, 10/10 params)
  unit_last/final_bn/beta (64, 64/128 params)
  unit_last/final_bn/gamma (64, 64/128 params)
  unit_last/final_bn/moving_mean (64, 64/64 params)
  unit_last/final_bn/moving_variance (64, 64/64 params)
```


12) A example of defining extra op type for ops using `OpLog`

First, in Python code, create an `OpLog` proto and add op type
information to it:

```python

op_log = tfprof_log_pb2.OpLog()
entry = op_log.log_entries.add()
entry.name = 'pool_logit/DW'
entry.types.append('pool_logit')
entry = op_log.log_entries.add()
entry.name = 'pool_logit/biases'
# Alternatively:
# var = tf.get_variable(xxx)
# entry.name = var.op.name
entry.types.append('pool_logit')
```

Second, call write_op_log to write the OpLog proto.

```python
tf.contrib.tfprof.tfprof_logger.write_op_log(
    sess.graph, /tmp/my_op_log_dir, op_log)

# Get run-time shape information in order to fill shapes and get flops.
tf.contrib.tfprof.tfprof_logger.write_op_log(
    sess.graph, /tmp/my_op_log_dir, op_log, run_meta)
```

Third, when starting the tfprof tool, specify
"--op_log_path /tmp/my_op_log_dir/op_log"

```shell
tfprof> scope -account_type_regexes pool_logit -max_depth 4 -select params
_TFProfRoot (--/650 params)
  pool_logit/DW (64x10, 640/640 params)
  pool_logit/biases (10, 10/10 params)
```

Note that when you call
`tf.contrib.tfprof.tfprof_logger.write_op_log(...)`,
the tool adds all `Variables` inside `tf.trainable_variables()` to
`_trainable_variables`.

12) Run tfprof in one-shot mode and dump result to file.

```shell
# By default output to stdout. Use -output option to change output types.
tfprof scope --graph_path=graph.pbtxt  \
             --max_depth=3 \
             --output="file:outfile=/tmp/dump"
Reading Files...
Parsing GraphDef...
Preparing Views...

cat /tmp/dump
_TFProfRoot (--/930.58k params)
  global_step (0/0 params)
  pool_logit/DW (64x10, 640/1.28k params)
  pool_logit/biases (10, 10/20 params)
```

13) Analyze how balanced Variable are on parameter servers.

In this tutorial, I'm going to use a seq2seq model, which are split
on several gpus at workers and several parameter servers.

In tfprof, 'device' is an op_type. For example, if op1 and op2 are placed on
gpu0. They share an op_type called 'gpu0'.

```shell
bazel-bin/tensorflow/tools/tfprof/tfprof \
  --graph_path ~/tfprof/textsum/graph.pbtxt  \
  --run_meta_path ~/tfprof/textsum/run_meta

# Looks like ps task 1 is holding twice more parameters than task 0.
tfprof> scope -select device,params -account_type_regexes .*ps.*task:0.* -max_depth 1
_TFProfRoot (--/25.81m params)
tfprof> scope -select device,params -account_type_regexes .*ps.*task:1.* -max_depth 1
_TFProfRoot (--/58.84m params)
```

### CLI Input Files

tfprof command line inference (CLI) loads dumped files from a tensorflow model.
Convert them into in-memory data structures. To use it, users need to specify
the locations of the dumped files. The following are the dumped files loaded
by tfprof:

<b>--graph_path:</b> GraphDef text file (required). Used to build in-memory
representation of the model. For example, graph.pbtxt written by tf.Supervisor
is a candidate. If you are not using tf.Supervisor, you can easily get GraphDef
using tf.Graph.as_graph_def() or other API.

<b>--run_meta_path:</b> tensorflow::RunMetadata.
Used to get the memory and time consumption of
each op of the model. Users need to enable it. For example, the following code
snippet writes a RunMetadata file:

```python
run_options = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
run_metadata = config_pb2.RunMetadata()
# Once a while, call it the get the RunMeta.
_ = self._sess.run(..., options=run_options, run_metadata=run_metadata)
with gfile.Open(os.path.join(output_dir, "run_meta"), "w") as f:
  f.write(run_metadata.SerializeToString())
```

<b>--op_log_path:</b>
tensorflow::tfprof::OpLog. A proto used to provide extra op information
for ops. By giving a group of ops a type name, users can easily aggregate the
statistics for those ops without accidentally missing or including extra ops.
tfprof exposes the following Python API to add op information and logging.

```python
tf.contrib.tfprof.tfprof_logger.write_op_log(graph, log_dir, op_log=None)
```

<b>--checkpoint_path:</b>
TensorFlow checkpoint. It defines _checkpoint_variable op type. It also
provides checkpointed tensors' values.


##Options

`-max_depth`: Show ops that are at most this number of hops from starting op in the tree/graph structure.

`-min_bytes`: Show ops that request at least this number of bytes.

`-min_micros`: Show ops that spend at least this number of microseconds to run.

`-min_params`: Show ops that contains at least this number of parameters.

`-min_float_ops`: Show ops that contain at least this number of float operations. Only available if an op has op.RegisterStatistics() defined and OpLog is provided

`-min_occurrence`: Show ops that appear at least this number of times. Only available in "op" view.

`-step`: Show the stats of the this step when multiple steps of RunMetadata were added. By default, show the average of all steps."

`-order_by`: Order the results by [name|depth|bytes|micros|params|float_ops|occurrence]

`-account_type_regexes`: Account and display the ops whose types match one of the type regexes specified. tfprof allow user to define extra op types for ops through tensorflow.tfprof.OpLog proto. regexes are comma-separated.

`-start_name_regexes`: Show ops starting from the ops that matches the regexes, recursively. regexes are comma-separated.

`-trim_name_regexes`: Hide ops starting from the ops that matches the regexes, recursively, regexes are comma-separated.

`-show_name_regexes`: Show ops that match the regexes. regexes are comma-separated.

`-hide_name_regexes`: Hide ops that match the regexes. regexes are comma-separated.

Notes: For each op, `-account_type_regexes` is first evaluated, only ops with
types matching the specified regexes are accounted and selected for displayed.
`-start/trim/show/hide_name_regexes` are used to further filter ops for display.
`-start_name_regexes` is evaluated first to search the starting ops to display.
Descendants of starting ops are then evaluated against `-show/hide_name_regexes`
to make display decision. If an op matches trim_name_regexes, all its
descendants are hidden. Ops statistics are *accounted even if they are hidden*
as long as they match the `-account_xxx` options.

`-account_displayed_op_only`: If True, only account the statistics of ops eventually displayed. If False, account all op statistics matching -account_type_regexes recursively.

`-select`: Comma-separated list of metrics to show: [bytes|micros|params|float_ops|occurrence|tensor_value|device|op_types].

`-output`: Output results as stdout, file or timeline.
The format is ```output_type:key=value,key=value```.
For example: ```timeline:outfile=<filename>```.
timeline: key=outfile, value=<filename>.
stdout: none.
file: key=outfile, value=<filename>.
