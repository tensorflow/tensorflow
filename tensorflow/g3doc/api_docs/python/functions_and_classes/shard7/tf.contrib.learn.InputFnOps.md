A return type for an input_fn.

This return type is currently only supported for serving input_fn.
Training and eval input_fn should return a `(features, labels)` tuple.

The expected return values are:
  features: A dict of string to `Tensor` or `SparseTensor`, specifying the
    features to be passed to the model.
  labels: A `Tensor`, `SparseTensor`, or a dict of string to `Tensor` or
    `SparseTensor`, specifying labels for training or eval. For serving, set
    `labels` to `None`.
  default_inputs: a dict of string to `Tensor` or `SparseTensor`, specifying
    the input placeholders (if any) that this input_fn expects to be fed.
    Typically, this is used by a serving input_fn, which expects to be fed
    serialized `tf.Example` protos.
- - -

#### `tf.contrib.learn.InputFnOps.__getnewargs__()` {#InputFnOps.__getnewargs__}

Return self as a plain tuple.  Used by copy and pickle.


- - -

#### `tf.contrib.learn.InputFnOps.__getstate__()` {#InputFnOps.__getstate__}

Exclude the OrderedDict from pickling


- - -

#### `tf.contrib.learn.InputFnOps.__new__(_cls, features, labels, default_inputs)` {#InputFnOps.__new__}

Create new instance of InputFnOps(features, labels, default_inputs)


- - -

#### `tf.contrib.learn.InputFnOps.__repr__()` {#InputFnOps.__repr__}

Return a nicely formatted representation string


- - -

#### `tf.contrib.learn.InputFnOps.default_inputs` {#InputFnOps.default_inputs}

Alias for field number 2


- - -

#### `tf.contrib.learn.InputFnOps.features` {#InputFnOps.features}

Alias for field number 0


- - -

#### `tf.contrib.learn.InputFnOps.labels` {#InputFnOps.labels}

Alias for field number 1


