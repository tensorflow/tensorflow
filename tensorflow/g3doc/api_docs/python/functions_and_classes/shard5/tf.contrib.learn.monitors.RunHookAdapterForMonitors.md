Wraps monitors into a SessionRunHook.
- - -

#### `tf.contrib.learn.monitors.RunHookAdapterForMonitors.__init__(monitors)` {#RunHookAdapterForMonitors.__init__}




- - -

#### `tf.contrib.learn.monitors.RunHookAdapterForMonitors.after_create_session(session, coord)` {#RunHookAdapterForMonitors.after_create_session}

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

#### `tf.contrib.learn.monitors.RunHookAdapterForMonitors.after_run(run_context, run_values)` {#RunHookAdapterForMonitors.after_run}




- - -

#### `tf.contrib.learn.monitors.RunHookAdapterForMonitors.before_run(run_context)` {#RunHookAdapterForMonitors.before_run}




- - -

#### `tf.contrib.learn.monitors.RunHookAdapterForMonitors.begin()` {#RunHookAdapterForMonitors.begin}




- - -

#### `tf.contrib.learn.monitors.RunHookAdapterForMonitors.end(session)` {#RunHookAdapterForMonitors.end}




