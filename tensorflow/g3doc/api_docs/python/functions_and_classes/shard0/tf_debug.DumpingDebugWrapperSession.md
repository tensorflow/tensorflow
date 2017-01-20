Debug Session wrapper that dumps debug data to filesystem.
- - -

#### `tf_debug.DumpingDebugWrapperSession.__enter__()` {#DumpingDebugWrapperSession.__enter__}




- - -

#### `tf_debug.DumpingDebugWrapperSession.__exit__(exec_type, exec_value, exec_tb)` {#DumpingDebugWrapperSession.__exit__}




- - -

#### `tf_debug.DumpingDebugWrapperSession.__init__(sess, session_root, watch_fn=None, log_usage=True)` {#DumpingDebugWrapperSession.__init__}

Constructor of DumpingDebugWrapperSession.

##### Args:


*  <b>`sess`</b>: The TensorFlow `Session` object being wrapped.
*  <b>`session_root`</b>: (`str`) Path to the session root directory. Must be a
    directory that does not exist or an empty directory. If the directory
    does not exist, it will be created by the debugger core during debug
    [`Session.run()`](../../../g3doc/api_docs/python/client.md#session.run)
    calls.
    As the `run()` calls occur, subdirectories will be added to
    `session_root`. The subdirectories' names has the following pattern:
      run_<epoch_time_stamp>_<uuid>
    E.g., run_1480734393835964_ad4c953a85444900ae79fc1b652fb324
*  <b>`watch_fn`</b>: (`Callable`) A Callable that can be used to define per-run
    debug ops and watched tensors. See the doc of
    `NonInteractiveDebugWrapperSession.__init__()` for details.
*  <b>`log_usage`</b>: (`bool`) whether the usage of this class is to be logged.

##### Raises:


*  <b>`ValueError`</b>: If `session_root` is an existing and non-empty directory or
   if `session_root` is a file.


- - -

#### `tf_debug.DumpingDebugWrapperSession.close()` {#DumpingDebugWrapperSession.close}




- - -

#### `tf_debug.DumpingDebugWrapperSession.graph` {#DumpingDebugWrapperSession.graph}




- - -

#### `tf_debug.DumpingDebugWrapperSession.invoke_node_stepper(node_stepper, restore_variable_values_on_exit=True)` {#DumpingDebugWrapperSession.invoke_node_stepper}

See doc of BaseDebugWrapperSession.invoke_node_stepper.


- - -

#### `tf_debug.DumpingDebugWrapperSession.on_run_end(request)` {#DumpingDebugWrapperSession.on_run_end}

See doc of BaseDebugWrapperSession.on_run_end.


- - -

#### `tf_debug.DumpingDebugWrapperSession.on_run_start(request)` {#DumpingDebugWrapperSession.on_run_start}

See doc of BaseDebugWrapperSession.on_run_start.


- - -

#### `tf_debug.DumpingDebugWrapperSession.on_session_init(request)` {#DumpingDebugWrapperSession.on_session_init}

See doc of BaseDebugWrapperSession.on_run_start.


- - -

#### `tf_debug.DumpingDebugWrapperSession.partial_run(handle, fetches, feed_dict=None)` {#DumpingDebugWrapperSession.partial_run}




- - -

#### `tf_debug.DumpingDebugWrapperSession.partial_run_setup(fetches, feeds=None)` {#DumpingDebugWrapperSession.partial_run_setup}

Sets up the feeds and fetches for partial runs in the session.


- - -

#### `tf_debug.DumpingDebugWrapperSession.run(fetches, feed_dict=None, options=None, run_metadata=None)` {#DumpingDebugWrapperSession.run}

Wrapper around Session.run() that inserts tensor watch options.

##### Args:


*  <b>`fetches`</b>: Same as the `fetches` arg to regular `Session.run()`.
*  <b>`feed_dict`</b>: Same as the `feed_dict` arg to regular `Session.run()`.
*  <b>`options`</b>: Same as the `options` arg to regular `Session.run()`.
*  <b>`run_metadata`</b>: Same as the `run_metadata` arg to regular `Session.run()`.

##### Returns:

  Simply forwards the output of the wrapped `Session.run()` call.

##### Raises:


*  <b>`ValueError`</b>: On invalid `OnRunStartAction` value.


- - -

#### `tf_debug.DumpingDebugWrapperSession.sess_str` {#DumpingDebugWrapperSession.sess_str}




- - -

#### `tf_debug.DumpingDebugWrapperSession.session` {#DumpingDebugWrapperSession.session}




