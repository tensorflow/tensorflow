Configuration for a dense input feature in a sequence item.

To treat a sparse input as dense, provide `allow_missing=True`; otherwise,
the parse functions will fail on any examples missing this feature.

Fields:
  shape: Shape of input data.
  dtype: Data type of input.
  allow_missing: Whether to allow this feature to be missing from a feature
    list item.
- - -

#### `tf.FixedLenSequenceFeature.__getnewargs__()` {#FixedLenSequenceFeature.__getnewargs__}

Return self as a plain tuple.  Used by copy and pickle.


- - -

#### `tf.FixedLenSequenceFeature.__getstate__()` {#FixedLenSequenceFeature.__getstate__}

Exclude the OrderedDict from pickling


- - -

#### `tf.FixedLenSequenceFeature.__new__(_cls, shape, dtype, allow_missing=False)` {#FixedLenSequenceFeature.__new__}

Create new instance of FixedLenSequenceFeature(shape, dtype, allow_missing)


- - -

#### `tf.FixedLenSequenceFeature.__repr__()` {#FixedLenSequenceFeature.__repr__}

Return a nicely formatted representation string


- - -

#### `tf.FixedLenSequenceFeature.allow_missing` {#FixedLenSequenceFeature.allow_missing}

Alias for field number 2


- - -

#### `tf.FixedLenSequenceFeature.dtype` {#FixedLenSequenceFeature.dtype}

Alias for field number 1


- - -

#### `tf.FixedLenSequenceFeature.shape` {#FixedLenSequenceFeature.shape}

Alias for field number 0


