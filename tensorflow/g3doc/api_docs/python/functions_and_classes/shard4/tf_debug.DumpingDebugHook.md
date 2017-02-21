A debugger hook that dumps debug data to filesystem.

Can be used as a monitor/hook for `tf.train.MonitoredSession`s and
`tf.contrib.learn`'s `Estimator`s and `Experiment`s.
- - -

#### `tf_debug.DumpingDebugHook.__enter__()` {#DumpingDebugHook.__enter__}




- - -

#### `tf_debug.DumpingDebugHook.__exit__(exec_type, exec_value, exec_tb)` {#DumpingDebugHook.__exit__}




- - -

#### `tf_debug.DumpingDebugHook.__init__(session_root, watch_fn=None, log_usage=True)` {#DumpingDebugHook.__init__}

Create a local debugger command-line interface (CLI) hook.

##### Args:


*  <b>`session_root`</b>: See doc of
    `dumping_wrapper.DumpingDebugWrapperSession.__init__`.
*  <b>`watch_fn`</b>: See doc of
    `dumping_wrapper.DumpingDebugWrapperSession.__init__`.
*  <b>`log_usage`</b>: (bool) Whether usage is to be logged.


- - -

#### `tf_debug.DumpingDebugHook.after_create_session(session, coord)` {#DumpingDebugHook.after_create_session}

Called when new TensorFlow session is created.

This is called to signal the hooks that a new session has been created. This
has two essential differences with the situation in which `begin` is called:

* When this is called, the graph is finalized and ops can no longer be added
    to the graph.
* This method will also be called as a result of recovering a wrapped
    session, not only at the beginning of the overall session.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that has been created.
*  <b>`coord`</b>: A Coordinator object which keeps track of all threads.


- - -

#### `tf_debug.DumpingDebugHook.after_run(run_context, run_values)` {#DumpingDebugHook.after_run}




- - -

#### `tf_debug.DumpingDebugHook.before_run(run_context)` {#DumpingDebugHook.before_run}




- - -

#### `tf_debug.DumpingDebugHook.begin()` {#DumpingDebugHook.begin}




- - -

#### `tf_debug.DumpingDebugHook.close()` {#DumpingDebugHook.close}




- - -

#### `tf_debug.DumpingDebugHook.end(session)` {#DumpingDebugHook.end}

Called at the end of session.

The `session` argument can be used in case the hook wants to run final ops,
such as saving a last checkpoint.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that will be soon closed.


- - -

#### `tf_debug.DumpingDebugHook.graph` {#DumpingDebugHook.graph}




- - -

#### `tf_debug.DumpingDebugHook.invoke_node_stepper(node_stepper, restore_variable_values_on_exit=True)` {#DumpingDebugHook.invoke_node_stepper}

See doc of BaseDebugWrapperSession.invoke_node_stepper.


- - -

#### `tf_debug.DumpingDebugHook.on_run_end(request)` {#DumpingDebugHook.on_run_end}

See doc of BaseDebugWrapperSession.on_run_end.


- - -

#### `tf_debug.DumpingDebugHook.on_run_start(request)` {#DumpingDebugHook.on_run_start}

See doc of BaseDebugWrapperSession.on_run_start.


- - -

#### `tf_debug.DumpingDebugHook.on_session_init(request)` {#DumpingDebugHook.on_session_init}

See doc of BaseDebugWrapperSession.on_run_start.


- - -

#### `tf_debug.DumpingDebugHook.partial_run(handle, fetches, feed_dict=None)` {#DumpingDebugHook.partial_run}




- - -

#### `tf_debug.DumpingDebugHook.partial_run_setup(fetches, feeds=None)` {#DumpingDebugHook.partial_run_setup}

Sets up the feeds and fetches for partial runs in the session.


- - -

#### `tf_debug.DumpingDebugHook.run(fetches, feed_dict=None, options=None, run_metadata=None)` {#DumpingDebugHook.run}

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

#### `tf_debug.DumpingDebugHook.sess_str` {#DumpingDebugHook.sess_str}




- - -

#### `tf_debug.DumpingDebugHook.session` {#DumpingDebugHook.session}




