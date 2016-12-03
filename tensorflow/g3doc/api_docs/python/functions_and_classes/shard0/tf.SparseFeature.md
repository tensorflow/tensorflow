Configuration for parsing a sparse input feature.

Fields:
  index_key: Name of index feature.  The underlying feature's type must
    be `int64` and its length must always match that of the `value_key`
    feature.
  value_key: Name of value feature.  The underlying feature's type must
    be `dtype` and its length must always match that of the `index_key`
    feature.
  dtype: Data type of the `value_key` feature.
  size: Each value in the `index_key` feature must be in `[0, size)`.
  already_sorted: A boolean to specify whether the values in `index_key` are
    already sorted. If so skip sorting, False by default (optional).
- - -

#### `tf.SparseFeature.__getnewargs__()` {#SparseFeature.__getnewargs__}

Return self as a plain tuple.  Used by copy and pickle.


- - -

#### `tf.SparseFeature.__getstate__()` {#SparseFeature.__getstate__}

Exclude the OrderedDict from pickling


- - -

#### `tf.SparseFeature.__new__(_cls, index_key, value_key, dtype, size, already_sorted=False)` {#SparseFeature.__new__}

Create new instance of SparseFeature(index_key, value_key, dtype, size, already_sorted)


- - -

#### `tf.SparseFeature.__repr__()` {#SparseFeature.__repr__}

Return a nicely formatted representation string


- - -

#### `tf.SparseFeature.already_sorted` {#SparseFeature.already_sorted}

Alias for field number 4


- - -

#### `tf.SparseFeature.dtype` {#SparseFeature.dtype}

Alias for field number 2


- - -

#### `tf.SparseFeature.index_key` {#SparseFeature.index_key}

Alias for field number 0


- - -

#### `tf.SparseFeature.size` {#SparseFeature.size}

Alias for field number 3


- - -

#### `tf.SparseFeature.value_key` {#SparseFeature.value_key}

Alias for field number 1


