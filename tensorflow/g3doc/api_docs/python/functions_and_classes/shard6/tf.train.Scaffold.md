Structure to create or gather pieces commonly needed to train a model.

When you build a model for training you usually need ops to initialize
variables, a `Saver` to checkpoint them, an op to collect summaries for
the visualizer, and so on.

Various libraries built on top of the core TensorFlow library take care of
creating some or all of these pieces and storing them in well known
collections in the graph.  The `Scaffold` class helps pick these pieces from
the graph collections, creating and adding them to the collections if needed.

If you call the scaffold constructor without any arguments, it will pick
pieces from the collections, creating default ones if needed when
`scaffold.finalize()` is called.  You can pass arguments to the constructor to
provide your own pieces.  Pieces that you pass to the constructor are not
added to the graph collections.

The following pieces are directly accessible as attributes of the `Scaffold`
object:

* `saver`: A `tf.Saver` object taking care of saving the variables.  Picked
  from and stored into the `SAVERS` collection in the graph by default.
* `init_op`: An op to run to initialize the variables.  Picked from and
  stored into the `INIT_OP` collection in the graph by default.
* `ready_op`: An op to verify that the variables are initialized.  Picked
  from and stored into the `READY_OP` collection in the graph by default.
* `ready_for_local_init_op`: An op to verify that global state has been
  initialized and it is alright to run `local_init_op`.  Picked from and
  stored into the `READY_FOR_LOCAL_INIT_OP` collection in the graph by
  default. This is needed when the initialization of local variables depends
  on the values of global variables.
* `local_init_op`: An op to initialize the local variables.  Picked
  from and stored into the `LOCAL_INIT_OP` collection in the graph by default.
* `summary_op`: An op to run and merge the summaries in the graph.  Picked
  from and stored into the `SUMMARY_OP` collection in the graph by default.
* `global_step`: A tensor containing the global step counter.  Picked
  from and stored into the `GLOBAL_STEP` collection in the graph by default.

You can also pass the following additional pieces to the constructor:

* `init_feed_dict`: A sessionn feed dictionary that should be used when
   running the init op.
* `init_fn`: A callable to run run after the init op to perform additional
  initializations.  The callable will be called as
  `init_fn(scaffold, session)`.
- - -

#### `tf.train.Scaffold.__init__(init_op=None, init_feed_dict=None, init_fn=None, ready_op=None, ready_for_local_init_op=None, local_init_op=None, summary_op=None, saver=None)` {#Scaffold.__init__}

Create a scaffold.

##### Args:


*  <b>`init_op`</b>: Optional op for initializing variables.
*  <b>`init_feed_dict`</b>: Optional session feed dictionary to use when running the
    init_op.
*  <b>`init_fn`</b>: Optional function to use to initialize the model after running
    the init_op.  Will be called as `init_fn(scaffold, session)`.
*  <b>`ready_op`</b>: Optional op to verify that the variables are initialized.  Must
    return an empty 1D string tensor when the variables are initialized, or
    a non-empty 1D string tensor listing the names of the non-initialized
    variables.
*  <b>`ready_for_local_init_op`</b>: Optional op to verify that the global variables
    are initialized and `local_init_op` can be run. Must return an empty
    1D string tensor when the global variables are initialized, or a
    non-empty 1D string tensor listing the names of the non-initialized
    global variables.
*  <b>`local_init_op`</b>: Optional op to initialize local variables.
*  <b>`summary_op`</b>: Optional op to gather all summaries.  Must return a scalar
    string tensor containing a serialized `Summary` proto.
*  <b>`saver`</b>: Optional `tf.Saver` object to use to save and restore variables.


- - -

#### `tf.train.Scaffold.finalize()` {#Scaffold.finalize}

Creates operations if needed and finalizes the graph.


- - -

#### `tf.train.Scaffold.get_or_default(arg_name, collection_key, default_constructor)` {#Scaffold.get_or_default}

Get from cache or create a default operation.


- - -

#### `tf.train.Scaffold.init_feed_dict` {#Scaffold.init_feed_dict}




- - -

#### `tf.train.Scaffold.init_fn` {#Scaffold.init_fn}




- - -

#### `tf.train.Scaffold.init_op` {#Scaffold.init_op}




- - -

#### `tf.train.Scaffold.local_init_op` {#Scaffold.local_init_op}




- - -

#### `tf.train.Scaffold.ready_for_local_init_op` {#Scaffold.ready_for_local_init_op}




- - -

#### `tf.train.Scaffold.ready_op` {#Scaffold.ready_op}




- - -

#### `tf.train.Scaffold.saver` {#Scaffold.saver}




- - -

#### `tf.train.Scaffold.summary_op` {#Scaffold.summary_op}




