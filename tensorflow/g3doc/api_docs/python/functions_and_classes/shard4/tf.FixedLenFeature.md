Configuration for parsing a fixed-length input feature.

To treat sparse input as dense, provide a `default_value`; otherwise,
the parse functions will fail on any examples missing this feature.

Fields:
  shape: Shape of input data.
  dtype: Data type of input.
  default_value: Value to be used if an example is missing this feature. It
      must be compatible with `dtype`.
- - -

#### `tf.FixedLenFeature.default_value` {#FixedLenFeature.default_value}

Alias for field number 2


- - -

#### `tf.FixedLenFeature.dtype` {#FixedLenFeature.dtype}

Alias for field number 1


- - -

#### `tf.FixedLenFeature.shape` {#FixedLenFeature.shape}

Alias for field number 0


