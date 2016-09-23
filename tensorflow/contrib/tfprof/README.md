# tfprof: A Profiling Tool for TensorFlow Models

go/tfprof

Author: Xin Pan (xpan@google.com, github: panyx0718)

Consultants: Jon Shlens (shlens@google.com), Pete Warden (petewarden@google.com)

[TOC]

## Introduction

tfprof is a profiling tool for TensorFlow that analyzes model architectures
and measures system performance.

###Major Features

1.  Measure model parameters, float operations, tensor shapes.
2.  Measure op execution times, requested memory size and device placement.
3.  Inspect checkpoint tensors' shapes and their values.
4.  Explore model based on name scope or graph structure.
5.  Selectively grouping/filtering/accounting/ordering ops.

### Interfaces

[CLI Tutorials](#cli-tutorials):
It supports interactive mode for exploration and single-shot mode for
scripts. Outputs can be dumped to files or printed in terminal.

Python API Tutorials: Python API is not released yet.

## CLI Tutorials

Tutorials are based on a 32 layers ResNet.
TODO(xpan): Provide graph.pbtxt, model.ckpt, tfprof_log and run_meta download.

### Examples

1) Start `tfprof` command line tool

```shell
# Build the tool.
bazel build -c opt tensorflow/contrib/tfprof/...

# Help information, including detail 'option' instructions.
bazel-bin/tensorflow/contrib/tfprof/tools/tfprof/tfprof help
#
# The following commands will start tfprof interactive mode.
#
# Profile model shapes and parameters only.
bazel-bin/tensorflow/contrib/tfprof/tools/tfprof/tfprof \
    --graph_path=/graph.pbtxt
#
# Additionally profile checkpoint statistics and values.
# Use '-account_type_regexes _checkpoint_variables' to select
# checkpoint tensors.
bazel-bin/tensorflow/contrib/tfprof/tools/tfprof/tfprof \
    --graph_path=graph.pbtxt \
    --checkpoint_path=model.ckpt
#
# Additionally profile ops requested memory and timing.
# See CLI Input Files section on generating run_meta file.
bazel-bin/tensorflow/contrib/tfprof/tools/tfprof/tfprof \
    --graph_path=graph.pbtxt \
    --run_meta_path=run_meta \
    --checkpoint_path=model.ckpt
#
# tfprof_log is used to define customized op types and float ops.
# Use tfprof_logger.write_op_log() to create tfprof_log.
# See 11) in Examples section on generating tfprof_log file.
bazel-bin/tensorflow/contrib/tfprof/tools/tfprof/tfprof \
    --graph_path=graph.pbtxt \
    --run_meta_path=run_meta \
    --op_log_path=tfprof_log \
    --checkpoint_path=model.ckpt
```
Note that `graph.pbtxt` is an ASCII text format.

2) Press enter to show the default options

```shell
tfprof>
tfprof>
-max_depth                  4
-min_bytes                  0
-min_micros                 0
-min_params                 0
-min_float_ops              0
-device_regexes             .*
-order_by                   name
-account_type_regexes       Variable
-start_name_regexes         .*
-trim_name_regexes
-show_name_regexes          .*
-hide_name_regexes          IsVariableInitialized_[0-9]+,save\/.*,^zeros[0-9_]*
-account_displayed_op_only  false
# supported select fileds. Availability depends on --[run_meta|checkpoint|op_log]_path.
# [bytes|micros|params|float_ops|num_hidden_ops|tensor_value|device|op_types]
-select                     params
-viz                        false
-dump_to_file
```

3) I want to see the `BatchNorm`'s gamma value in checkpoint.

```shell
# Requires --graph_path, --checkpoint_path.
tfprof> scope -show_name_regexes unit_1_0.*gamma -select tensor_value -max_depth 5
_TFProfRoot ()
  unit_1_0/shared_activation/init_bn/gamma ()
[1.80 2.10 2.06 1.91 2.26 1.86 1.81 1.37 1.78 1.85 1.96 1.54 2.04 2.34 2.22 1.99 ],
  unit_1_0/sub2/bn2/gamma ()
[1.57 1.83 1.30 1.25 1.59 1.14 1.26 0.82 1.19 1.10 1.48 1.01 0.82 1.23 1.21 1.14 ],
```

4) I want to see my checkpoint tensors shape and number of parameters.

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

5) I defined an op named ‘cost’ to calculate the loss. I want to know what ops
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

6) I want to know the expensive operations during the back propagation.
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

7) Show the number of float operations in the model.
Note: float operations calculation depends on
1) op.RegisterStatistics. If an op doesn’t
have RegisterStatistics defined, its float operations cannot be counted.
2) fully defined shape is also necessary in order to calculate flops.
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

8) Show the number of parameters of all `tf.trainable_variables()` in the model.

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

9) What if I’m lazy and don’t want to define op type? I have given my ops
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

10) TensorFlow has built-in op types. For example, built-in op type `Variable`
seems to include `Variable's` created by your model. However, be careful when
depending on it because TensorFlow creates extra `Variable` ops implicitly and
the implicitly created ops can have the same prefix as the `Variable's` you
defined.

In the following example, extra `Variables` are created and “/Momentum” is
appended to their names. This might cause you “model capacity” calculation
to get wrong.

```shell
tfprof> scope -account_type_regexes Variable -max_depth 4 -select params
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


11) A example of defining extra op type for ops using `OpLog`

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
tfprof_logger.write_op_log(sess.graph, /tmp/my_op_log_dir, op_log)
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
`tfprof_logger.write_op_log(...)`, the tool adds all `Variables` inside
`tf.trainable_variables()` to `_trainable_variables`.

12) Run tfprof in one-shot mode and dump result to file.

```shell
# Printed to stdout if --dump_to_file is not set.
tfprof scope --graph_path /cns/ij-d/home/xpan/tfprof/graph.pbtxt  \
             --max_depth 3 \
             --dump_to_file "/tmp/dump"
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
bazel-bin/tensorflow/contrib/tfprof/tools/tfprof/tfprof \
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
statistics for those ops without accidently missing or including extra ops.
tfprof exposes the following Python API to add op information and logging.

```python
  def write_op_log(graph, log_dir, op_log=None)
```

<b>--checkpoint_path:</b>
TensorFlow checkpoint. It defines _checkpoint_variable op type. It also
provides checkpointed tensors' values.


## Design


### In-memory representation

<b>Scope:</b> This representation organizes ops based on name scope hierarchy,
similar to filesystem hierarchy. Hence, it is essentially a tree data structure.
For example op1 with name “name1/name2” is a child of op2 with name “name1”.

<b>Graph:</b> The representation organizes ops based on op inputs. Hence it is
a graph structure. The graph is a “directed acyclic graph” (hopefully), with
direction from “output to input”. The direction is design this way so that users
can trace from “result” to its “sources”.

### Command line options

tfprof’s major goals are to measure system performance and quicly analyze
model architectures. Hence, its commands and options should allow users to achieve
these 2 goals easily.

<b>graph:</b> It is expected that users will mostly use graph representation to
debug system performance. Hence, tfprof supports graph command, which pulls the
graph in-memory representation described above.

<b>scope:</b> It is expected that some users might want to explore their model
statistics using the name scope information they defined in the Python codes.
Hence, tfprof supports “scope” command, which pulls the tree in-memory
representation.

<b>set:</b> It is used to store the options so that user doesn’t need to
re-type the same option again and again in the follow up command line. Note that
tfprof has traditional terminal’s history and auto-complete support.

<b>help:</b> print help information.

<b>Options:</b> Run “tfprof help” to get detailed explanations.

```python
"-max_depth",
"-min_bytes",
"-min_micros",
"-min_params",
"-min_float_ops",
"-order_by",
"-account_type_regexes",
"-start_name_regexes",
"-trim_name_regexes",
"-show_name_regexes",
"-hide_name_regexes",
"-account_displayed_op_only",
"-select",
"-viz",  # Only supported for graph command.
"-dump_to_file",
```

A key design is that stats are aggregated from descendants up to ancestors.
`-account_type_regexes` is used to decide which ops stat is accounted. It makes
decision based on op type. Usually set it to `.*` if no extra type information
is added to the ops using OpLog. Intuitively, only accounted ops are displayed.
`-min/max` and `-show/hide/trim/start` options are only used the optionally
displayed or hide ops based on ops’ name and stats. However, they don’t prevent
tfprof from accounting stats of hidden ops. Hence, the stat of a op can be
aggregated by its parent even if it is hidden. `-account_displayed_op_only` is
an option to break this rule. When it is set, only displayed ops are accounted.

Regexes are all comma-separated, for example `-show_name_regexes`
`regex1.*,regex2.*`. It is designed this way because it is convenient and comma
is not expected to show up in op names.

`-order_by` is used to order displayed ops. Displayed ops at the same hierarchy
(notice the indent printed) are sorted according to order_by.

## Future Work

* Load SummaryWriter event logs so that it can show the latest summary value.

* Better sorting and aggregation of outputs. Easier comprehension.

* Currently, shape information is based on `graph.pbtxt`. When the shape
information is incomplete, tfprof ignores it. See if it can use `RunMetadata`
and `Checkpoint` to complete shape information.
