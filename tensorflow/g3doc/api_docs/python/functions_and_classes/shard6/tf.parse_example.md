### `tf.parse_example(serialized, features, name=None, example_names=None)` {#parse_example}

Parses `Example` protos into a `dict` of tensors.

Parses a number of serialized [`Example`](https://www.tensorflow.org/code/tensorflow/core/example/example.proto)
protos given in `serialized`.

`example_names` may contain descriptive names for the corresponding serialized
protos. These may be useful for debugging purposes, but they have no effect on
the output. If not `None`, `example_names` must be the same length as
`serialized`.

This op parses serialized examples into a dictionary mapping keys to `Tensor`
and `SparseTensor` objects. `features` is a dict from keys to `VarLenFeature`,
`SparseFeature`, and `FixedLenFeature` objects. Each `VarLenFeature`
and `SparseFeature` is mapped to a `SparseTensor`, and each
`FixedLenFeature` is mapped to a `Tensor`.

Each `VarLenFeature` maps to a `SparseTensor` of the specified type
representing a ragged matrix. Its indices are `[batch, index]` where `batch`
is the batch entry the value is from in `serialized`, and `index` is the
value's index in the list of values associated with that feature and example.

Each `SparseFeature` maps to a `SparseTensor` of the specified type
representing a sparse matrix of shape
`(serialized.size(), SparseFeature.size)`. Its indices are `[batch, index]`
where `batch` is the batch entry the value is from in `serialized`, and
`index` is the value's index is given by the values in the
`SparseFeature.index_key` feature column.

Each `FixedLenFeature` `df` maps to a `Tensor` of the specified type (or
`tf.float32` if not specified) and shape `(serialized.size(),) + df.shape`.

`FixedLenFeature` entries with a `default_value` are optional. With no default
value, we will fail if that `Feature` is missing from any example in
`serialized`.

Examples:

For example, if one expects a `tf.float32` sparse feature `ft` and three
serialized `Example`s are provided:

```
serialized = [
  features
    { feature { key: "ft" value { float_list { value: [1.0, 2.0] } } } },
  features
    { feature []},
  features
    { feature { key: "ft" value { float_list { value: [3.0] } } }
]
```

then the output will look like:

```
{"ft": SparseTensor(indices=[[0, 0], [0, 1], [2, 0]],
                    values=[1.0, 2.0, 3.0],
                    dense_shape=(3, 2)) }
```

Given two `Example` input protos in `serialized`:

```
[
  features {
    feature { key: "kw" value { bytes_list { value: [ "knit", "big" ] } } }
    feature { key: "gps" value { float_list { value: [] } } }
  },
  features {
    feature { key: "kw" value { bytes_list { value: [ "emmy" ] } } }
    feature { key: "dank" value { int64_list { value: [ 42 ] } } }
    feature { key: "gps" value { } }
  }
]
```

And arguments

```
example_names: ["input0", "input1"],
features: {
    "kw": VarLenFeature(tf.string),
    "dank": VarLenFeature(tf.int64),
    "gps": VarLenFeature(tf.float32),
}
```

Then the output is a dictionary:

```python
{
  "kw": SparseTensor(
      indices=[[0, 0], [0, 1], [1, 0]],
      values=["knit", "big", "emmy"]
      dense_shape=[2, 2]),
  "dank": SparseTensor(
      indices=[[1, 0]],
      values=[42],
      dense_shape=[2, 1]),
  "gps": SparseTensor(
      indices=[],
      values=[],
      dense_shape=[2, 0]),
}
```

For dense results in two serialized `Example`s:

```
[
  features {
    feature { key: "age" value { int64_list { value: [ 0 ] } } }
    feature { key: "gender" value { bytes_list { value: [ "f" ] } } }
   },
   features {
    feature { key: "age" value { int64_list { value: [] } } }
    feature { key: "gender" value { bytes_list { value: [ "f" ] } } }
  }
]
```

We can use arguments:

```
example_names: ["input0", "input1"],
features: {
    "age": FixedLenFeature([], dtype=tf.int64, default_value=-1),
    "gender": FixedLenFeature([], dtype=tf.string),
}
```

And the expected output is:

```python
{
  "age": [[0], [-1]],
  "gender": [["f"], ["f"]],
}
```

Given two `Example` input protos in `serialized`:

```
[
  features {
    feature { key: "val" value { float_list { value: [ 0.5, -1.0 ] } } }
    feature { key: "ix" value { int64_list { value: [ 3, 20 ] } } }
  },
  features {
    feature { key: "val" value { float_list { value: [ 0.0 ] } } }
    feature { key: "ix" value { int64_list { value: [ 42 ] } } }
  }
]
```

And arguments

```
example_names: ["input0", "input1"],
features: {
    "sparse": SparseFeature("ix", "val", tf.float32, 100),
}
```

Then the output is a dictionary:

```python
{
  "sparse": SparseTensor(
      indices=[[0, 3], [0, 20], [1, 42]],
      values=[0.5, -1.0, 0.0]
      dense_shape=[2, 100]),
}
```

##### Args:


*  <b>`serialized`</b>: A vector (1-D Tensor) of strings, a batch of binary
    serialized `Example` protos.
*  <b>`features`</b>: A `dict` mapping feature keys to `FixedLenFeature`,
    `VarLenFeature`, and `SparseFeature` values.
*  <b>`name`</b>: A name for this operation (optional).
*  <b>`example_names`</b>: A vector (1-D Tensor) of strings (optional), the names of
    the serialized protos in the batch.

##### Returns:

  A `dict` mapping feature keys to `Tensor` and `SparseTensor` values.

##### Raises:


*  <b>`ValueError`</b>: if any feature is invalid.

