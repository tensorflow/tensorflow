# TensorFlow Debugger (TFDBG)

[TOC]

TensorFlow Debugger (TFDBG) is a specialized debugger for TensorFlow's computation
runtime. TFDBG in TensorFlow 2.x provides access to:

- Tensor values during [eager](https://www.tensorflow.org/guide/eager) and
  [graph](https://www.tensorflow.org/api_docs/python/tf/Graph) execution.
- Structure of computation graphs
- Source code and stack traces associated with these execution and
  graph-execution events.

## How to use TFDBG?

TFDBG in TensorFlow 2.x consists of a Python API that enables dumping debug data
to the file system (namely `tf.debugging.experimental.enable_dump_debug_info()`)
and a TensorBoard-based GUI that provides an interactive visualization of the
debug data (i.e., *TensorBoard Debugger V2 Plugin*).

`enable_dump_debug_info()` offers a number of levels of tensor-value
instrumentation varying in the amount of information dumped and the incurred
performance overhead.

See the API documentation of
[`tf.debugging.experimental.enable_dump_debug_info()`](https://www.tensorflow.org/api_docs/python/tf/debugging/experimental/enable_dump_debug_info)

For a detailed walkthrough of the GUI TensorBoard Debugger V2 Plugin, see
[Debugging Numerical Issues in TensorFlow Programs Using TensorBoard Debugger
V2](https://www.tensorflow.org/tensorboard/debugger_v2).

## Known issues and limitations

1.  Using `tf.debugging.experimental.enable_dumpp_debug_info()` leads to
    performance penalty on your TensorFlow program. The amount of slowdown
    varied depending on whether you are using TensorFlow on CPU, GPUs, or TPUs.
    The performance penalty is the highest on TPUs, followed by GPUs, and lowest
    on CPU.
2.  `tf.debugging.experimental.enable_dump_debug_info()` is currently
    incompatible with
    [model saving/loading and checkpointing](https://www.tensorflow.org/tutorials/keras/save_and_load)

## Legacy API for TensorFlow 1.x

TensorFlow 1.x's execution paradigm is different from that of TensorFlow v2; it
is based on the deprecated
[`tf.Session`](https://www.tensorflow.org/api_docs/python/tf/compat/v1/Session)
If you are using TensorFlow 1.x, you can use the deprecated
`tf_debug.LocalCLIDebugWrapperSession` wrapper for `tf.Session`
to inspect tensor values and other types of debug information in a
terminal-based command-line interface. For details, see
[this blog post](https://developers.googleblog.com/2017/02/debug-tensorflow-models-with-tfdbg.html).
