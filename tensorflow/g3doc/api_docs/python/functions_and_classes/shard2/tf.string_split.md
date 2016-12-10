### `tf.string_split(source, delimiter=' ')` {#string_split}

Split elements of `source` based on `delimiter` into a `SparseTensor`.

Let N be the size of source (typically N will be the batch size). Split each
element of `source` based on `delimiter` and return a `SparseTensor`
containing the splitted tokens. Empty tokens are ignored.

If `delimiter` is an empty string, each element of the `source` is split
into individual strings, each containing one byte. (This includes splitting
multibyte sequences of UTF-8.) If delimiter contains multiple bytes, it is
treated as a set of delimiters with each considered a potential split point.

For example:
N = 2, source[0] is 'hello world' and source[1] is 'a b c', then the output
will be

st.indices = [0, 0;
              0, 1;
              1, 0;
              1, 1;
              1, 2]
st.shape = [2, 3]
st.values = ['hello', 'world', 'a', 'b', 'c']

##### Args:


*  <b>`source`</b>: `1-D` string `Tensor`, the strings to split.
*  <b>`delimiter`</b>: `0-D` string `Tensor`, the delimiter character, the string should
    be length 0 or 1.

##### Raises:


*  <b>`ValueError`</b>: If delimiter is not a string.

##### Returns:

  A `SparseTensor` of rank `2`, the strings split according to the delimiter.
  The first column of the indices corresponds to the row in `source` and the
  second column corresponds to the index of the split component in this row.

