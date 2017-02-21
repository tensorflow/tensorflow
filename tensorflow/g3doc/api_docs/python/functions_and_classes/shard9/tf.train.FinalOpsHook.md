A run hook which evaluates `Tensors` at the end of a session.
- - -

#### `tf.train.FinalOpsHook.__init__(final_ops, final_ops_feed_dict=None)` {#FinalOpsHook.__init__}

Constructs the FinalOpHook with ops to run at the end of the session.

##### Args:


*  <b>`final_ops`</b>: A single `Tensor`, a list of `Tensors` or a dictionary of
    names to `Tensors`.
*  <b>`final_ops_feed_dict`</b>: A feed dictionary to use when running
    `final_ops_dict`.


- - -

#### `tf.train.FinalOpsHook.after_create_session(session, coord)` {#FinalOpsHook.after_create_session}

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

#### `tf.train.FinalOpsHook.after_run(run_context, run_values)` {#FinalOpsHook.after_run}

Called after each call to run().

The `run_values` argument contains results of requested ops/tensors by
`before_run()`.

The `run_context` argument is the same one send to `before_run` call.
`run_context.request_stop()` can be called to stop the iteration.

##### Args:


*  <b>`run_context`</b>: A `SessionRunContext` object.
*  <b>`run_values`</b>: A SessionRunValues object.


- - -

#### `tf.train.FinalOpsHook.before_run(run_context)` {#FinalOpsHook.before_run}

Called before each call to run().

You can return from this call a `SessionRunArgs` object indicating ops or
tensors to add to the upcoming `run()` call.  These ops/tensors will be run
together with the ops/tensors originally passed to the original run() call.
The run args you return can also contain feeds to be added to the run()
call.

The `run_context` argument is a `SessionRunContext` that provides
information about the upcoming `run()` call: the originally requested
op/tensors, the TensorFlow Session.

At this point graph is finalized and you can not add ops.

##### Args:


*  <b>`run_context`</b>: A `SessionRunContext` object.

##### Returns:

  None or a `SessionRunArgs` object.


- - -

#### `tf.train.FinalOpsHook.begin()` {#FinalOpsHook.begin}

Called once before using the session.

When called, the default graph is the one that will be launched in the
session.  The hook can modify the graph by adding new operations to it.
After the `begin()` call the graph will be finalized and the other callbacks
can not modify the graph anymore. Second call of `begin()` on the same
graph, should not change the graph.


- - -

#### `tf.train.FinalOpsHook.end(session)` {#FinalOpsHook.end}




- - -

#### `tf.train.FinalOpsHook.final_ops_values` {#FinalOpsHook.final_ops_values}




