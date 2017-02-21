Runs `feed_fn` and sets the `feed_dict` accordingly.
- - -

#### `tf.train.FeedFnHook.__init__(feed_fn)` {#FeedFnHook.__init__}

Constructs the FeedFnHook with given `feed_fn`.

##### Args:


*  <b>`feed_fn`</b>: function, no arguments and returns `dict` to feed.


- - -

#### `tf.train.FeedFnHook.after_create_session(session, coord)` {#FeedFnHook.after_create_session}

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

#### `tf.train.FeedFnHook.after_run(run_context, run_values)` {#FeedFnHook.after_run}

Called after each call to run().

The `run_values` argument contains results of requested ops/tensors by
`before_run()`.

The `run_context` argument is the same one send to `before_run` call.
`run_context.request_stop()` can be called to stop the iteration.

##### Args:


*  <b>`run_context`</b>: A `SessionRunContext` object.
*  <b>`run_values`</b>: A SessionRunValues object.


- - -

#### `tf.train.FeedFnHook.before_run(run_context)` {#FeedFnHook.before_run}




- - -

#### `tf.train.FeedFnHook.begin()` {#FeedFnHook.begin}

Called once before using the session.

When called, the default graph is the one that will be launched in the
session.  The hook can modify the graph by adding new operations to it.
After the `begin()` call the graph will be finalized and the other callbacks
can not modify the graph anymore. Second call of `begin()` on the same
graph, should not change the graph.


- - -

#### `tf.train.FeedFnHook.end(session)` {#FeedFnHook.end}

Called at the end of session.

The `session` argument can be used in case the hook wants to run final ops,
such as saving a last checkpoint.

##### Args:


*  <b>`session`</b>: A TensorFlow Session that will be soon closed.


