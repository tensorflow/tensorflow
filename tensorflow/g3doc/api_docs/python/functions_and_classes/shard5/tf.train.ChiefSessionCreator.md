Creates a tf.Session  for a chief.
- - -

#### `tf.train.ChiefSessionCreator.__init__(scaffold=None, master='', config=None, checkpoint_dir=None, checkpoint_filename_with_path=None)` {#ChiefSessionCreator.__init__}

Initializes a chief session creator.

##### Args:


*  <b>`scaffold`</b>: A `Scaffold` used for gathering or building supportive ops. If
    not specified a default one is created. It's used to finalize the graph.
*  <b>`master`</b>: `String` representation of the TensorFlow master to use.
*  <b>`config`</b>: `ConfigProto` proto used to configure the session.
*  <b>`checkpoint_dir`</b>: A string.  Optional path to a directory where to restore
    variables.
*  <b>`checkpoint_filename_with_path`</b>: Full file name path to the checkpoint file.


- - -

#### `tf.train.ChiefSessionCreator.create_session()` {#ChiefSessionCreator.create_session}




