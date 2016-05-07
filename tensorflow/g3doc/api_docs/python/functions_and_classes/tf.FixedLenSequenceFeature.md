Configuration for a dense input feature in a sequence item.

To treat a sparse input as dense, provide `allow_missing=True`; otherwise,
the parse functions will fail on any examples missing this feature.

Fields:
  shape: Shape of input data.
  dtype: Data type of input.
  allow_missing: Whether to allow this feature to be missing from a feature
    list item.
- - -

#### `tf.FixedLenSequenceFeature.allow_missing` {#FixedLenSequenceFeature.allow_missing}

Alias for field number 2


- - -

#### `tf.FixedLenSequenceFeature.dtype` {#FixedLenSequenceFeature.dtype}

Alias for field number 1


- - -

#### `tf.FixedLenSequenceFeature.shape` {#FixedLenSequenceFeature.shape}

Alias for field number 0


