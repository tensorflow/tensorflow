Represents arguments to be added to a `Session.run()` call.

Args:
  fetches: Exactly like the 'fetches' argument to Session.Run().
    Can be a single tensor or op, a list of 'fetches' or a dictionary
    of fetches.  For example:
      fetches = global_step_tensor
      fetches = [train_op, summary_op, global_step_tensor]
      fetches = {'step': global_step_tensor, 'summ': summary_op}
    Note that this can recurse as expected:
      fetches = {'step': global_step_tensor,
                 'ops': [train_op, check_nan_op]}
  feed_dict: Exactly like the `feed_dict` argument to `Session.Run()`
  options: Exactly like the `options` argument to `Session.run()`, i.e., a
    config_pb2.RunOptions proto.
- - -

#### `tf.train.SessionRunArgs.__getnewargs__()` {#SessionRunArgs.__getnewargs__}

Return self as a plain tuple.  Used by copy and pickle.


- - -

#### `tf.train.SessionRunArgs.__getstate__()` {#SessionRunArgs.__getstate__}

Exclude the OrderedDict from pickling


- - -

#### `tf.train.SessionRunArgs.__new__(cls, fetches, feed_dict=None, options=None)` {#SessionRunArgs.__new__}




- - -

#### `tf.train.SessionRunArgs.__repr__()` {#SessionRunArgs.__repr__}

Return a nicely formatted representation string


- - -

#### `tf.train.SessionRunArgs.feed_dict` {#SessionRunArgs.feed_dict}

Alias for field number 1


- - -

#### `tf.train.SessionRunArgs.fetches` {#SessionRunArgs.fetches}

Alias for field number 0


- - -

#### `tf.train.SessionRunArgs.options` {#SessionRunArgs.options}

Alias for field number 2


