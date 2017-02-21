Creates a tf.Session for a worker.
- - -

#### `tf.train.WorkerSessionCreator.__init__(scaffold=None, master='', config=None)` {#WorkerSessionCreator.__init__}

Initializes a worker session creator.

##### Args:


*  <b>`scaffold`</b>: A `Scaffold` used for gathering or building supportive ops. If
    not specified a default one is created. It's used to finalize the graph.
*  <b>`master`</b>: `String` representation of the TensorFlow master to use.
*  <b>`config`</b>: `ConfigProto` proto used to configure the session.


- - -

#### `tf.train.WorkerSessionCreator.create_session()` {#WorkerSessionCreator.create_session}




