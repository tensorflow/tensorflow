# Specifying return data type for `py_func` calls

The `py_func` op requires specifying a
[data type](https://www.tensorflow.org/guide/tensors#data_types).

When wrapping a function with `py_func`, for instance using
`@autograph.do_not_convert(run_as=autograph.RunMode.PY_FUNC)`, you have two
options to specify the returned data type:

 * explicitly, with a specified `tf.DType` value
 * by matching the data type of an input argument, which is then assumed to be
     a `Tensor`

Examples:

Specify an explicit data type:

```
  def foo(a):
    return a + 1

  autograph.util.wrap_py_func(f, return_dtypes=[tf.float32])
```

Match the data type of the first argument:

```
  def foo(a):
    return a + 1

  autograph.util.wrap_py_func(
      f, return_dtypes=[autograph.utils.py_func.MatchDType(0)])
```
