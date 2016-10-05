# TensorFlow Debugger (tfdbg) Command Line Interface Tutorial: MNIST

**(Under development, subject to change)**

This tutorial showcases how to debug a type of frequently-encountered problem
in TensorFlow model development: bad numerical values (nans and infs) causing
model training to fail, using the TensorFlow Debugger (**tfdbg**).

To observe the issue, run the code without the debugger:

```
bazel build -c opt tensorflow/python/debug:debug_mnist && \
    bazel-bin/tensorflow/python/debug/debug_mnist
```

This code trains a simple NN for MNIST digit image recognition. Notice that the
accuracy increases slightly after the first training step, but then gets stuck
at a near-chance level.

```
Accuracy at step 0: 0.1113
Accuracy at step 1: 0.3183
Accuracy at step 2: 0.098
Accuracy at step 3: 0.098
Accuracy at step 4: 0.098
Accuracy at step 5: 0.098
Accuracy at step 6: 0.098
Accuracy at step 7: 0.098
Accuracy at step 8: 0.098
Accuracy at step 9: 0.098
```

Scratching your head, you suspect that certain nodes in the training graph
experienced bad numeric values such as infs and nans. The computation graph
paradigm of TensorFlow makes it hard to debug such model internal states
with general-purpose debuggers such as Python's pdb. A specialized debugger,
**tfdbg**, makes it easier to diagnose this type of issues and pinpoint the
exact node at which the issue has first surfaced.

Let's now try answering this question with tfdbg. Execute the above command
again, but with the "--debug" flag added:

```
bazel build -c opt tensorflow/python/debug:debug_mnist && \
    bazel-bin/tensorflow/python/debug/debug_mnist --debug
```

Behind the scene, the `--debug` flag wraps the Session object with a debugger
wrapper:

```python
if FLAGS.debug:
  sess = local_cli.LocalCLIDebugWrapperSession(sess)
  sess.add_tensor_filter("has_inf_or_nan", debug_data.has_inf_or_nan)
```

This wrapper has the same interface as Session, so debugging requires no other
changes in the code. But the wrapper provides additional features including:

1. Bringing up a terminal-based user interface (UI) before and after each
`run()` call, to let you control the execution and inspect the graph's internal
states.
2. Allowing you to register special "filters" for tensor values, to facilitate
the diagnosis of issues.

In this example, we are registering a tensor filter called `"has_nan_of_inf"`,
which simply determines if there are any nan or inf values in any intermediate
tensor of the graph. This filter is a common enough use case that we ship it
with the debug_data module.

```python
def has_inf_or_nan(datum, tensor):
  return np.any(np.isnan(tensor)) or np.any(np.isinf(tensor))
```

However, you can write your own custom filters. See
`tensorflow/python/debug/debug_data.py` for more details.

The debug wrapper session will prompt you when it is about to execute the first
`run()` call, with information regarding the fetched tensor and feed
dictionaries displayed on the screen.

```
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

This is what we refer to as the **run-start UI**. If the screen size is
too small to display the content of the message in its entirety, you can use the
`PageUp / PageDown / Home / End` keys to navigate the screen output.

As the screen output indicates, the first `run()` call calculates the accuracy
using a test data set, i.e., a forward pass on the graph. You can enter the
command `run` to launch the `run()` call. This will bring up another screen
right after the `run()` call has ended. The screen displays all dumped
intermedate tensors from the run. This is called the **run-end UI**.

```
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

Try the following commands at the `tfdbg>` prompt:

| Command example    | Explanation           |
| ------------- |:--------------------- |
| `pt hidden/Relu:0` | Print the value of the tensor `hidden/Relu:0`. |
| `ni -a hidden/Relu` | Displays information about the node `hidden/Relu`, including node attributes. |
| `li -r hidden/Relu:0` | List the inputs to the node `hidden/Relu`, recursively, i.e., the input tree. |
| `lo -r hidden/Relu:0` | List the recipients of the output of the node `hidden/Relu`, recursively, i.e., the output recipient tree. |
| `lt -n softmax.*` | List all dumped tensors whose names match the regular-expression pattern `softmax.*`. |
| `lt -t MatMul` | List all dumped tensors whose node type is `MatMul`. |
| `help` | Print general help information listing all available tfdbg commands and their flags. |
| `help lt` | Print the help information for the `lt` command. |

In this 1st `run()` call, there happen to be no problematic numerical values.
You can exit this run-end UI by entering command `exit`. Then you will be at
the 2nd run-start UI.

```
--- run-start: run #2: fetch: train/Adam; 2 feeds --------------
======================================
About to enter Session run() call #2:

Fetch(es):
  train/Adam

Feed dict(s):
  input/x-input:0
  input/y-input:0
======================================
...
```

Instead of entering `run` repeatedly and manually searching for nans and infs
in the run-end UI after every `run()` call, you can use the following command to
let the debugger execute the `run()` calls without stopping at the run-start or
run-end prompt, until the first nan or inf value shows up in the graph:

```
tfdbg> run -f has_inf_or_nan
```

This is because we have previously registered a filter for nans and infs called
`has_inf_or_nan` (see above). If you have registered any other filters, you can
let tfdbg run till any tensors pass that filter as well, e.g.,

```
# In python code:
sess.add_tensor_filter('my_filter', my_filter_callable)

# Run at tfdbg run-start prompt:
tfdbg> run -f my_filter
```

After you have entered `run -f has_inf_or_nan`, you will see the following
screen with a red-colored title line indicating tfdbg has stopped immediately
after a `run()` call has generated intermediate tensors that pass the specified
filter `has_inf_or_nan`:


```
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

As the screen dislpay indicates, the `has_inf_or_nan` filter is first passed
during the 4th `run()` call, i.e., an Adam optimizer forward-backward training
pass on the graph. In this run, there are 30 (out of the total 87) intermediate
tensors containing nan or inf values. These tensors are listed in chronological
order, with the timestamps displayed on the left. At the top of the list, you
can see the first tensor in which the bad numerical values first surfaced,
namely `cross_entropy/Log:0`.

To view the value of the tensor, do

```
tfdbg> pt cross_entropy/Log:0
```

<!---
TODO(cais): Once the "/inf" style regex search+highlight+scroll is checked in,
modify the sentence below to reflect that.
--->
Scroll down a little and you will notice some scattered inf values. Using the
following command, you can see that this node has the op type "Log" and that
its input is the node "softmax/Softmax".

```
tfdbg> ni cross_entropy/Log
```

Looking at the value of the input tensor, i.e., the only output of
"softmax/Softmax": "softmax/Softmax:0", you can notice that there are zeros:

```
tfdbg> pt softmax/Softmax:0
```

Now, it is clear that the origin of the bad numerical values is the node
`cross_entropy/Log` taking logs of zeros. You can go back to the source code in
`debug_mnist.py` and infer that the culprit line is:

```python
diff = y_ * tf.log(y)
```

Applying a value clipping on the input to `tf.log` should resolve this problem.
For example, you can do:

```python
diff = y_ * tf.log(tf.clip_by_value(y, 1e-8, 1.0))
```

**Other features of the tfdbg diagnostics CLI:**

<!---
TODO(cais): Add the following UI features once they are checked in: tab
completion; regex search and highlighting.
--->
*   Navigation through command history using the Up and Down arrow keys.
    Prefix-based navigation is also supported.



Frequently-asked questions:
===========================

*   **Q**: Do the timestamps on the left side of the `lt` output reflect actual
       performance in a non-debugging session?<br />
**A**: No. The debugger inserts additional special-purpose debug nodes to the
       graph to record the values of intermediate tensors. These nodes certainly
       slow down the graph execution. If you are interested in profiling your
       model, check out
       [tfprof](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/tfprof)
       and other profiling tools for TensorFlow.
