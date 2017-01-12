Concrete subclass of BaseDebugWrapperSession implementing a local CLI.

This class has all the methods that a `session.Session` object has, in order
to support debugging with minimal code changes. Invoking its `run()` method
will launch the command-line interface (CLI) of tfdbg.
- - -

#### `tf_debug.LocalCLIDebugWrapperSession.__enter__()` {#LocalCLIDebugWrapperSession.__enter__}




- - -

#### `tf_debug.LocalCLIDebugWrapperSession.__exit__(exec_type, exec_value, exec_tb)` {#LocalCLIDebugWrapperSession.__exit__}




- - -

#### `tf_debug.LocalCLIDebugWrapperSession.__init__(sess, dump_root=None, log_usage=True, ui_type='curses')` {#LocalCLIDebugWrapperSession.__init__}

Constructor of LocalCLIDebugWrapperSession.

##### Args:


*  <b>`sess`</b>: The TensorFlow `Session` object being wrapped.
*  <b>`dump_root`</b>: (`str`) optional path to the dump root directory. Must be a
    directory that does not exist or an empty directory. If the directory
    does not exist, it will be created by the debugger core during debug
    `run()` calls and removed afterwards.
*  <b>`log_usage`</b>: (`bool`) whether the usage of this class is to be logged.
*  <b>`ui_type`</b>: (`str`) requested UI type. Currently supported:
    (curses | readline)

##### Raises:


*  <b>`ValueError`</b>: If dump_root is an existing and non-empty directory or if
    dump_root is a file.


- - -

#### `tf_debug.LocalCLIDebugWrapperSession.add_tensor_filter(filter_name, tensor_filter)` {#LocalCLIDebugWrapperSession.add_tensor_filter}

Add a tensor filter.

##### Args:


*  <b>`filter_name`</b>: (`str`) name of the filter.
*  <b>`tensor_filter`</b>: (`callable`) the filter callable. See the doc string of
    `DebugDumpDir.find()` for more details about its signature.


- - -

#### `tf_debug.LocalCLIDebugWrapperSession.close()` {#LocalCLIDebugWrapperSession.close}




- - -

#### `tf_debug.LocalCLIDebugWrapperSession.graph` {#LocalCLIDebugWrapperSession.graph}




- - -

#### `tf_debug.LocalCLIDebugWrapperSession.invoke_node_stepper(node_stepper, restore_variable_values_on_exit=True)` {#LocalCLIDebugWrapperSession.invoke_node_stepper}

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

#### `tf_debug.LocalCLIDebugWrapperSession.on_run_end(request)` {#LocalCLIDebugWrapperSession.on_run_end}

Overrides on-run-end callback.

##### Actions taken:

  1) Load the debug dump.
  2) Bring up the Analyzer CLI.

##### Args:


*  <b>`request`</b>: An instance of OnSessionInitRequest.

##### Returns:

  An instance of OnSessionInitResponse.


- - -

#### `tf_debug.LocalCLIDebugWrapperSession.on_run_start(request)` {#LocalCLIDebugWrapperSession.on_run_start}

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

#### `tf_debug.LocalCLIDebugWrapperSession.on_session_init(request)` {#LocalCLIDebugWrapperSession.on_session_init}

Overrides on-session-init callback.

##### Args:


*  <b>`request`</b>: An instance of `OnSessionInitRequest`.

##### Returns:

  An instance of `OnSessionInitResponse`.


- - -

#### `tf_debug.LocalCLIDebugWrapperSession.partial_run(handle, fetches, feed_dict=None)` {#LocalCLIDebugWrapperSession.partial_run}




- - -

#### `tf_debug.LocalCLIDebugWrapperSession.partial_run_setup(fetches, feeds=None)` {#LocalCLIDebugWrapperSession.partial_run_setup}

Sets up the feeds and fetches for partial runs in the session.


- - -

#### `tf_debug.LocalCLIDebugWrapperSession.run(fetches, feed_dict=None, options=None, run_metadata=None)` {#LocalCLIDebugWrapperSession.run}

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

#### `tf_debug.LocalCLIDebugWrapperSession.sess_str` {#LocalCLIDebugWrapperSession.sess_str}




- - -

#### `tf_debug.LocalCLIDebugWrapperSession.session` {#LocalCLIDebugWrapperSession.session}




