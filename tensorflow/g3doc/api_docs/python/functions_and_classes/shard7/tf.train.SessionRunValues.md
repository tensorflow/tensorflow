Contains the results of `Session.run()`.

In the future we may use this object to add more information about result of
run without changing the Hook API.

Args:
  results: The return values from `Session.run()` corresponding to the fetches
    attribute returned in the RunArgs. Note that this has the same shape as
    the RunArgs fetches.  For example:
      fetches = global_step_tensor
      => results = nparray(int)
      fetches = [train_op, summary_op, global_step_tensor]
      => results = [None, nparray(string), nparray(int)]
      fetches = {'step': global_step_tensor, 'summ': summary_op}
      => results = {'step': nparray(int), 'summ': nparray(string)}
- - -

#### `tf.train.SessionRunValues.__getnewargs__()` {#SessionRunValues.__getnewargs__}

Return self as a plain tuple.  Used by copy and pickle.


- - -

#### `tf.train.SessionRunValues.__getstate__()` {#SessionRunValues.__getstate__}

Exclude the OrderedDict from pickling


- - -

#### `tf.train.SessionRunValues.__new__(_cls, results)` {#SessionRunValues.__new__}

Create new instance of SessionRunValues(results,)


- - -

#### `tf.train.SessionRunValues.__repr__()` {#SessionRunValues.__repr__}

Return a nicely formatted representation string


- - -

#### `tf.train.SessionRunValues.results` {#SessionRunValues.results}

Alias for field number 0


