Configuration for parsing a fixed-length input feature.

To treat sparse input as dense, provide a `default_value`; otherwise,
the parse functions will fail on any examples missing this feature.

Fields:
  shape: Shape of input data.
  dtype: Data type of input.
  default_value: Value to be used if an example is missing this feature. It
      must be compatible with `dtype`.
- - -

#### `tf.FixedLenFeature.__getnewargs__()` {#FixedLenFeature.__getnewargs__}

Return self as a plain tuple.  Used by copy and pickle.


- - -

#### `tf.FixedLenFeature.__getstate__()` {#FixedLenFeature.__getstate__}

Exclude the OrderedDict from pickling


- - -

#### `tf.FixedLenFeature.__new__(_cls, shape, dtype, default_value=None)` {#FixedLenFeature.__new__}

Create new instance of FixedLenFeature(shape, dtype, default_value)


- - -

#### `tf.FixedLenFeature.__repr__()` {#FixedLenFeature.__repr__}

Return a nicely formatted representation string


- - -

#### `tf.FixedLenFeature.default_value` {#FixedLenFeature.default_value}

Alias for field number 2


- - -

#### `tf.FixedLenFeature.dtype` {#FixedLenFeature.dtype}

Alias for field number 1


- - -

#### `tf.FixedLenFeature.shape` {#FixedLenFeature.shape}

Alias for field number 0


