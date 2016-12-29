Command-line-interface debugger hook.

Can be used as a monitor/hook for tf.train.MonitoredSession.
- - -

#### `tf_debug.LocalCLIDebugHook.__enter__()` {#LocalCLIDebugHook.__enter__}




- - -

#### `tf_debug.LocalCLIDebugHook.__exit__(exec_type, exec_value, exec_tb)` {#LocalCLIDebugHook.__exit__}




- - -

#### `tf_debug.LocalCLIDebugHook.__init__()` {#LocalCLIDebugHook.__init__}

Create a local debugger command-line interface (CLI) hook.


- - -

#### `tf_debug.LocalCLIDebugHook.add_tensor_filter(filter_name, tensor_filter)` {#LocalCLIDebugHook.add_tensor_filter}

Add a tensor filter.

##### Args:


*  <b>`filter_name`</b>: (`str`) name of the filter.
*  <b>`tensor_filter`</b>: (`callable`) the filter callable. See the doc string of
    `DebugDumpDir.find()` for more details about its signature.


- - -

#### `tf_debug.LocalCLIDebugHook.after_create_session(session)` {#LocalCLIDebugHook.after_create_session}

Called when new TensorFlow session is created.

This is called to signal the hooks that a new session has been created. This
has two essential differences with the situation in which `begin` is called:

* When this is called, the graph is finalized and ops can no longer be added
    to the graph.
* This method will also be called as a result of recovering a wrapped
    session, not only at the beginning of the overall session.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that has been created.


- - -

#### `tf_debug.LocalCLIDebugHook.after_run(run_context, run_values)` {#LocalCLIDebugHook.after_run}




- - -

#### `tf_debug.LocalCLIDebugHook.before_run(run_context)` {#LocalCLIDebugHook.before_run}




- - -

#### `tf_debug.LocalCLIDebugHook.begin()` {#LocalCLIDebugHook.begin}




- - -

#### `tf_debug.LocalCLIDebugHook.close()` {#LocalCLIDebugHook.close}




- - -

#### `tf_debug.LocalCLIDebugHook.end(session)` {#LocalCLIDebugHook.end}

Called at the end of session.

The `session` argument can be used in case the hook wants to run final ops,
such as saving a last checkpoint.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that will be soon closed.


- - -

#### `tf_debug.LocalCLIDebugHook.graph` {#LocalCLIDebugHook.graph}




- - -

#### `tf_debug.LocalCLIDebugHook.invoke_node_stepper(node_stepper, restore_variable_values_on_exit=True)` {#LocalCLIDebugHook.invoke_node_stepper}

Overrides method in base class to implement interactive node stepper.

##### Args:


*  <b>`node_stepper`</b>: (`stepper.NodeStepper`) The underlying NodeStepper API
    object.
*  <b>`restore_variable_values_on_exit`</b>: (`bool`) Whether any variables whose
    values have been altered during this node-stepper invocation should be
    restored to their old values when this invocation ends.

##### Returns:

  The same return values as the `Session.run()` call on the same fetches as
    the NodeStepper.


- - -

#### `tf_debug.LocalCLIDebugHook.on_run_end(request)` {#LocalCLIDebugHook.on_run_end}

Overrides on-run-end callback.

##### Actions taken:

  1) Load the debug dump.
  2) Bring up the Analyzer CLI.

##### Args:


*  <b>`request`</b>: An instance of OnSessionInitRequest.

##### Returns:

  An instance of OnSessionInitResponse.


- - -

#### `tf_debug.LocalCLIDebugHook.on_run_start(request)` {#LocalCLIDebugHook.on_run_start}

Overrides on-run-start callback.

##### Invoke the CLI to let user choose what action to take:

  `run` / `invoke_stepper`.

##### Args:


*  <b>`request`</b>: An instance of `OnSessionInitRequest`.

##### Returns:

  An instance of `OnSessionInitResponse`.

##### Raises:


*  <b>`RuntimeError`</b>: If user chooses to prematurely exit the debugger.


- - -

#### `tf_debug.LocalCLIDebugHook.on_session_init(request)` {#LocalCLIDebugHook.on_session_init}

Overrides on-session-init callback.

##### Args:


*  <b>`request`</b>: An instance of `OnSessionInitRequest`.

##### Returns:

  An instance of OnSessionInitResponse.


- - -

#### `tf_debug.LocalCLIDebugHook.partial_run(handle, fetches, feed_dict=None)` {#LocalCLIDebugHook.partial_run}




- - -

#### `tf_debug.LocalCLIDebugHook.partial_run_setup(fetches, feeds=None)` {#LocalCLIDebugHook.partial_run_setup}

Sets up the feeds and fetches for partial runs in the session.


- - -

#### `tf_debug.LocalCLIDebugHook.run(fetches, feed_dict=None, options=None, run_metadata=None)` {#LocalCLIDebugHook.run}

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

#### `tf_debug.LocalCLIDebugHook.sess_str` {#LocalCLIDebugHook.sess_str}




- - -

#### `tf_debug.LocalCLIDebugHook.session` {#LocalCLIDebugHook.session}




