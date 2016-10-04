Provides information about the `session.run()` call being made.

Provides information about original request to `Session.Run()` function.
SessionRunHook objects can stop the loop by calling `request_stop()` of
`run_context`. In the future we may use this object to add more information
about run without changing the Hook API.
- - -

#### `tf.train.SessionRunContext.__init__(original_args, session)` {#SessionRunContext.__init__}

Initializes SessionRunContext.


- - -

#### `tf.train.SessionRunContext.original_args` {#SessionRunContext.original_args}

A `SessionRunArgs` object holding the original arguments of `run()`.

If user called `MonitoredSession.run(fetches=a, feed_dict=b)`, then this
field is equal to SessionRunArgs(a, b).

##### Returns:

 A `SessionRunArgs` object


- - -

#### `tf.train.SessionRunContext.request_stop()` {#SessionRunContext.request_stop}

Sets stop requested field.

Hooks can use this function to request stop of iterations.
`MonitoredSession` checks whether this is called or not.


- - -

#### `tf.train.SessionRunContext.session` {#SessionRunContext.session}

A TensorFlow session object which will execute the `run`.


- - -

#### `tf.train.SessionRunContext.stop_requested` {#SessionRunContext.stop_requested}

Returns whether a stop is requested or not.

If true, `MonitoredSession` stops iterations.

##### Returns:

  A `bool`


