# TensorFlow Debugger (tfdbg) Command-Line-Interface Tutorial: MNIST

**(Experimental)**

TensorFlow debugger (**tfdbg**) is a specialized debugger for TensorFlow. It
provides visibility into the internal structure and states of running
TensorFlow graphs. The insight gained from this visibility should facilitate
debugging of various types of model bugs during training and inference.

This tutorial showcases the features of tfdbg
command-line interface (CLI), by focusing on how to debug a
type of frequently-encountered bug in TensorFlow model development:
bad numerical values (`nan`s and `inf`s) causing training to fail.

To **observe** such an issue, run the following code without the debugger:

```none
bazel build -c opt tensorflow/python/debug:debug_mnist && \
    bazel-bin/tensorflow/python/debug/debug_mnist
```

This code trains a simple NN for MNIST digit image recognition. Notice that the
accuracy increases slightly after the first training step, but then gets stuck
at a low (near-chance) level:

```none
Accuracy at step 0: 0.1113
Accuracy at step 1: 0.3183
Accuracy at step 2: 0.098
Accuracy at step 3: 0.098
Accuracy at step 4: 0.098
...
```

Scratching your head, you suspect that certain nodes in the training graph
generated bad numeric values such as `inf`s and `nan`s. The computation-graph
paradigm of TensorFlow makes it hard to debug such model internal states
with general-purpose debuggers such as Python's pdb.
**tfdbg** specializes in diagnosing these types of issues and pinpointing the
exact node where the problem first surfaced.

## Adding tfdbg to TensorFlow Sessions

To add support for **tfdbg** in our example, we just need to add the following
three lines of code, which wrap the Session object with a debugger wrapper when
the `--debug` flag is provided:

```python
if FLAGS.debug:
  sess = tf_debug.LocalCLIDebugWrapperSession(sess)
  sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
```

This wrapper has the same interface as Session, so enabling debugging requires
no other changes to the code. But the wrapper provides additional features
including:

* Bringing up a terminal-based user interface (UI) before and after each
`run()` call, to let you control the execution and inspect the graph's internal
state.
* Allowing you to register special "filters" for tensor values, to facilitate
the diagnosis of issues.

In this example, we are registering a tensor filter called `"has_inf_or_nan"`,
which simply determines if there are any `nan` or `inf` values in any
intermediate tensor of the graph. (This filter is a common enough use case that
we ship it with the `debug_data` module.)

```python
def has_inf_or_nan(datum, tensor):
  return np.any(np.isnan(tensor)) or np.any(np.isinf(tensor))
```

TIP: You can also write your own custom filters. See
[`tensorflow/python/debug/debug_data.py`](https://www.tensorflow.org/code/tensorflow/python/debug/debug_data.py)
for additional filter examples.

## Debugging Model Training with tfdbg

Let's try training the model again with debugging enabled. Execute the command
from above, this time with the `--debug` flag added:

```none
bazel build -c opt tensorflow/python/debug:debug_mnist && \
    bazel-bin/tensorflow/python/debug/debug_mnist --debug
```

The debug wrapper session will prompt you when it is about to execute the first
`run()` call, with information regarding the fetched tensor and feed
dictionaries displayed on the screen.

```none
--- run-start: run #1: fetch: accuracy/accuracy/Mean:0; 2 feeds
======================================
About to enter Session run() call #1:

Fetch(es):
  accuracy/accuracy/Mean:0

Feed dict(s):
  input/x-input:0
  input/y-input:0
======================================

Select one of the following commands to proceed ---->
  run:
      Execute the run() call with the debug tensor-watching
  run -n:
      Execute the run() call without the debug tensor-watching
  run -f <filter_name>:
      Keep executing run() calls until a dumped tensor passes
      a given, registered filter emerge. Registered filter(s):
        * has_inf_or_nan
--- Scroll: 0.00% ----------------------------------------------
tfdbg>
```

This is what we refer to as the *run-start UI*. If the screen size is
too small to display the content of the message in its entirety, you can use the
**PageUp** / **PageDown** / **Home** / **End** keys to navigate the screen
output.

As the screen output indicates, the first `run()` call calculates the accuracy
using a test data set—i.e., a forward pass on the graph. You can enter the
command `run` (or its shorthand `r`) to launch the `run()` call.

This will bring up another screen
right after the `run()` call has ended, which will display all dumped
intermedate tensors from the run. (These tensors can also be obtained by
running the command `lt` after you executed `run`.) This is called the
**run-end UI**:

```none
--- run-end: run #1: fetch: accuracy/accuracy/Mean:0; 2 feeds --
21 dumped tensor(s):

[0.000 ms] accuracy/correct_prediction/ArgMax/dimension:0
[0.000 ms] softmax/biases/Variable:0
[0.013 ms] hidden/biases/Variable:0
[0.112 ms] softmax/weights/Variable:0
[1.953 ms] hidden/weights/Variable:0
[4.566 ms] accuracy/accuracy/Const:0
[5.188 ms] accuracy/correct_prediction/ArgMax_1:0
[6.901 ms] hidden/biases/Variable/read:0
[9.140 ms] softmax/biases/Variable/read:0
[11.145 ms] softmax/weights/Variable/read:0
[19.563 ms] hidden/weights/Variable/read:0
[171.189 ms] hidden/Wx_plus_b/MatMul:0
[209.433 ms] hidden/Wx_plus_b/add:0
[245.143 ms] hidden/Relu:0
--- Scroll: 0.00% ----------------------------------------------
tfdbg>
```

Try the following commands at the `tfdbg>` prompt (referencing the code at
`third_party/tensorflow/python/debug/examples/debug_mnist.py`):

| Command Example    | Explanation           |
|:----------------------------- |:----------------------------------- |
| `pt hidden/Relu:0` | Print the value of the tensor `hidden/Relu:0`. |
| `pt hidden/Relu:0[0:50,:]` | Print a subarray of the tensor `hidden/Relu:0`, using [numpy](http://www.numpy.org/)-style array slicing. |
| `pt hidden/Relu:0[0:50,:] -a` | For a large tensor like the one here, print its value in its entirety—i.e., without using any ellipsis. May take a long time for large tensors. |
| `pt hidden/Relu:0[0:10,:] -a -r [1,inf]` | Use the `-r` flag to highlight elements falling into the specified numerical range. Multiple ranges can be used in conjunction, e.g., `-r [[-inf,-1],[1,inf]]`.|
| `@[10,0]` or `@10,0` | Navigate to indices [10, 0] in the tensor being displayed. |
| `/inf` | Search the screen output with the regex `inf` and highlight any matches. |
| `/` | Scroll to the next line with matches to the searched regex (if any). |
| `ni -a hidden/Relu` | Display information about the node `hidden/Relu`, including node attributes. |
| `li -r hidden/Relu:0` | List the inputs to the node `hidden/Relu`, recursively—i.e., the input tree. |
| `lo -r hidden/Relu:0` | List the recipients of the output of the node `hidden/Relu`, recursively—i.e., the output recipient tree. |
| `lt -n softmax.*` | List all dumped tensors whose names match the regular-expression pattern `softmax.*`. |
| `lt -t MatMul` | List all dumped tensors whose node type is `MatMul`. |
| `run_info` or `ri` | Display information about the current run, including fetches and feeds. |
| `help` | Print general help information listing all available **tfdbg** commands and their flags. |
| `help lt` | Print the help information for the `lt` command. |

In this first `run()` call, there happen to be no problematic numerical values.
You can move on to the next run by using the command `run` or its shorthand `r`.

> TIP: If you enter `run` or `r` repeatedly, you will be able to move through the
> `run()` calls in a sequential manner.
>
> You can also use the `-t` flag to move ahead a number of `run()` calls at a time, for example:
>
> ```
> tfdbg> run -t 10
> ```

Instead of entering `run` repeatedly and manually searching for `nan`s and
`inf`s in the run-end UI after every `run()` call, you can use the following
command to let the debugger repeatedly execute `run()` calls without stopping at
the run-start or run-end prompt, until the first `nan` or `inf` value shows up
in the graph:

```none
tfdbg> run -f has_inf_or_nan
```

> NOTE: This works because we have previously registered a filter for `nan`s and `inf`s called
> `has_inf_or_nan` (as explained previously). If you have registered any other filters, you can
> let **tfdbg** run till any tensors pass that filter as well, e.g.,
>
> ```
> # In python code:
> sess.add_tensor_filter('my_filter', my_filter_callable)
>
> # Run at tfdbg run-start prompt:
> tfdbg> run -f my_filter
> ```

After you enter `run -f has_inf_or_nan`, you will see the following
screen with a red-colored title line indicating **tfdbg** stopped immediately
after a `run()` call generated intermediate tensors that passed the specified
filter `has_inf_or_nan`:


```none
--- run-end: run #4: fetch: train/Adam; 2 feeds ----------------
30 dumped tensor(s) passing filter "has_inf_or_nan":

[13.255 ms] cross_entropy/Log:0
[13.499 ms] cross_entropy/mul:0
[14.426 ms] train/gradients/cross_entropy/mul_grad/mul:0
[14.681 ms] train/gradients/cross_entropy/mul_grad/Sum:0
[14.885 ms] train/gradients/cross_entropy/Log_grad/Inv:0
[15.239 ms] train/gradients/cross_entropy/Log_grad/mul:0
[15.378 ms] train/gradients/softmax/Softmax_grad/mul:0
--- Scroll: 0.00% ----------------------------------------------
tfdbg>
```

As the screen display indicates, the `has_inf_or_nan` filter is first passed
during the fourth `run()` call: an [Adam optimizer](https://arxiv.org/abs/1412.6980)
forward-backward training pass on the graph. In this run, 30 (out of the total
87) intermediate tensors contain `nan` or `inf` values. These tensors are listed
in chronological order, with their timestamps displayed on the left. At the top
of the list, you can see the first tensor in which the bad numerical values
first surfaced: `cross_entropy/Log:0`.

To view the value of the tensor, run

```none
tfdbg> pt cross_entropy/Log:0
```

Scroll down a little and you will notice some scattered `inf` values. If the
instances of `inf` and `nan` are difficult to spot by eye, you can use the
following command to perform a regex search and highlight the output:

```none
tfdbg> /inf
```

Or, alternatively:

``` none
tfdbg> /(inf|nan)
```

To go back to the list of "offending" tensors, use the up-arrow key to
navigate to the following command, and hit Enter:

```none
tfdbg> lt -f has_inf_or_nan
```

To further debug, display more information about `cross_entropy/Log`:

```none
tfdbg> ni cross_entropy/Log
--- run-end: run #4: fetch: train/Adam; 2 feeds ---
Node cross_entropy/Log

  Op: Log
  Device: /job:localhost/replica:0/task:0/cpu:0

  1 input(s) + 0 control input(s):
    1 input(s):
      [Softmax] softmax/Softmax

  3 recipient(s) + 0 control recipient(s):
    3 recipient(s):
      [Mul] cross_entropy/mul
      [Shape] train/gradients/cross_entropy/mul_grad/Shape_1
      [Mul] train/gradients/cross_entropy/mul_grad/mul
```

You can see that this node has the op type `Log`
and that its input is the node `softmax/Softmax`. Run the following command to
take a closer look at the input tensor:

```none
tfdbg> pt softmax/Softmax:0
```

Examine the values in the input tensor, and search to see if there are any zeros:

``` none
tfdbg> /0\.000
```

Indeed, there are zeros. Now it is clear that the origin of the bad numerical
values is the node `cross_entropy/Log` taking logs of zeros. You can go back to
the source code in [`debug_mnist.py`](https://www.tensorflow.org/code/tensorflow/python/debug/examples/debug_mnist.py)
and infer that the culprit line is:

```python
diff = y_ * tf.log(y)
```

Apply a value clipping on the input to [`tf.log`](../../../g3doc/api_docs/python/math_ops.md#log)
to resolve this problem:

```python
diff = y_ * tf.log(tf.clip_by_value(y, 1e-8, 1.0))
```

Now, try training again with `--debug`:

```none
bazel build -c opt tensorflow/python/debug:debug_mnist && \
    bazel-bin/tensorflow/python/debug/debug_mnist --debug
```

Enter `run -f has_inf_or_nan` at the `tfdbg>` prompt and confirm that no tensors
are flagged as containing `nan` or `inf` values, and accuracy no longer gets
stuck. Success!

## Other Features of the tfdbg Diagnostics CLI:

*   Navigation through command history using the Up and Down arrow keys.
    Prefix-based navigation is also supported.
*   Tab completion of commands and some command arguments.
*   Write screen output to file by using bash-style redirection. For example:

  ```none
  tfdbg> pt cross_entropy/Log:0[:, 0:10] > /tmp/xent_value_slices.txt
  ```

## Frequently Asked Questions

**Q**: _Do the timestamps on the left side of the `lt` output reflect actual
       performance in a non-debugging session?_

**A**: No. The debugger inserts additional special-purpose debug nodes to the
       graph to record the values of intermediate tensors. These nodes certainly
       slow down the graph execution. If you are interested in profiling your
       model, check out
       [tfprof](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/tfprof)
       and other profiling tools for TensorFlow.

**Q**: _How do I link tfdbg against my `Session` in Bazel?_

**A**: In your BUILD rule, declare the dependency: `"//tensorflow:tensorflow_py"`.
       In your Python file, add:

```python
from tensorflow.python import debug as tf_debug

# Then wrap your TensorFlow Session with the local-CLI wrapper.
sess = tf_debug.LocalCLIDebugWrapperSession(sess)
```

**Q**: _Can I use `tfdbg` if I am using tf-learn Estimators, instead of
managing my own `Session` objects?_

**A**: Currently, `tfdbg` can only debug the `fit()` method of tf-learn
Estimators. Support for debugging `evaluate()` will come soon. To debug
`Estimator.fit()`, create a monitor and supply it as an argument. For example:

```python
from tensorflow.python import debug as tf_debug

# Create a local CLI debug hook and use it as a monitor when calling fit().
classifier.fit(x=training_set.data,
               y=training_set.target,
               steps=1000,
               monitors=[tf_debug.LocalCLIDebugHook()])
```

For a detailed [example](https://www.tensorflow.org/code/tensorflow/python/debug/examples/debug_tflearn_iris.py) based on
[tf-learn's iris tutorial](../../../g3doc/tutorials/tflearn/index.md),
run:

```none
bazel build -c opt tensorflow/python/debug:debug_tflearn_iris && \
    bazel-bin/tensorflow/python/debug/debug_tflearn_iris --debug
```

**Q**: _Does tfdbg help debugging runtime errors such as shape mismatches?_

**A**: Yes. tfdbg intercepts errors generated by ops during runtime and presents
       the errors with some debug instructions to the user in the CLI.
       See examples:

```none
# Debugging shape mismatch during matrix multiplication.
bazel build -c opt tensorflow/python/debug:debug_errors && \
    bazel-bin/tensorflow/python/debug/debug_errors \
        -error shape_mismatch --debug

# Debugging uninitialized variable.
bazel build -c opt tensorflow/python/debug:debug_errors && \
    bazel-bin/tensorflow/python/debug/debug_errors \
    -error uninitialized_variable --debug
```
