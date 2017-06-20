# TensorFlow Debugger
[TOC]

Public Python API of TensorFlow Debugger (tfdbg).

## Functions for adding debug watches

These functions help you modify `RunOptions` to specify which `Tensor`s are to
be watched when the TensorFlow graph is executed at runtime.

*   @{tfdbg.add_debug_tensor_watch}
*   @{tfdbg.watch_graph}
*   @{tfdbg.watch_graph_with_blacklists}


## Classes for debug-dump data and directories

These classes allow you to load and inspect tensor values dumped from
TensorFlow graphs during runtime.

*   @{tfdbg.DebugTensorDatum}
*   @{tfdbg.DebugDumpDir}


## Functions for loading debug-dump data

*   @{tfdbg.load_tensor_from_event_file}


## Tensor-value predicates

Built-in tensor-filter predicates to support conditional breakpoint between
runs. See `DebugDumpDir.find()` for more details.

*   @{tfdbg.has_inf_or_nan}


## Session wrapper class and `SessionRunHook` implementations

These classes allow you to

* wrap aroundTensorFlow `Session` objects to debug plain TensorFlow models
  (see `DumpingDebugWrapperSession` and `LocalCLIDebugWrapperSession`), or
* generate `SessionRunHook` objects to debug `tf.contrib.learn` models (see
  `DumpingDebugHook` and `LocalCLIDebugHook`).

*   @{tfdbg.DumpingDebugHook}
*   @{tfdbg.DumpingDebugWrapperSession}
*   @{tfdbg.LocalCLIDebugHook}
*   @{tfdbg.LocalCLIDebugWrapperSession}
