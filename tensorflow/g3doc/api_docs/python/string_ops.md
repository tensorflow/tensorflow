<!-- This file is machine generated: DO NOT EDIT! -->

# Strings

Note: Functions taking `Tensor` arguments can also take anything accepted by
[`tf.convert_to_tensor`](framework.md#convert_to_tensor).

[TOC]

## Hashing

String hashing ops take a string input tensor and map each element to an
integer.

- - -

### `tf.string_to_hash_bucket_fast(input, num_buckets, name=None)` {#string_to_hash_bucket_fast}

Converts each string in the input Tensor to its hash mod by a number of buckets.

The hash function is deterministic on the content of the string within the
process and will never change. However, it is not suitable for cryptography.
This function may be used when CPU time is scarce and inputs are trusted or
unimportant. There is a risk of adversaries constructing inputs that all hash
to the same bucket. To prevent this problem, use a strong hash function with
`tf.string_to_hash_bucket_strong`.

##### Args:


*  <b>`input`</b>: A `Tensor` of type `string`. The strings to assign a hash bucket.
*  <b>`num_buckets`</b>: An `int` that is `>= 1`. The number of buckets.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `int64`.
  A Tensor of the same shape as the input `string_tensor`.


- - -

### `tf.string_to_hash_bucket_strong(input, num_buckets, key, name=None)` {#string_to_hash_bucket_strong}

Converts each string in the input Tensor to its hash mod by a number of buckets.

The hash function is deterministic on the content of the string within the
process. The hash function is a keyed hash function, where attribute `key`
defines the key of the hash function. `key` is an array of 2 elements.

A strong hash is important when inputs may be malicious, e.g. URLs with
additional components. Adversaries could try to make their inputs hash to the
same bucket for a denial-of-service attack or to skew the results. A strong
hash prevents this by making it dificult, if not infeasible, to compute inputs
that hash to the same bucket. This comes at a cost of roughly 4x higher compute
time than `tf.string_to_hash_bucket_fast`.

##### Args:


*  <b>`input`</b>: A `Tensor` of type `string`. The strings to assign a hash bucket.
*  <b>`num_buckets`</b>: An `int` that is `>= 1`. The number of buckets.
*  <b>`key`</b>: A list of `ints`.
    The key for the keyed hash function passed as a list of two uint64
    elements.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `int64`.
  A Tensor of the same shape as the input `string_tensor`.


- - -

### `tf.string_to_hash_bucket(string_tensor, num_buckets, name=None)` {#string_to_hash_bucket}

Converts each string in the input Tensor to its hash mod by a number of buckets.

The hash function is deterministic on the content of the string within the
process.

Note that the hash function may change from time to time.
This functionality will be deprecated and it's recommended to use
`tf.string_to_hash_bucket_fast()` or `tf.string_to_hash_bucket_strong()`.

##### Args:


*  <b>`string_tensor`</b>: A `Tensor` of type `string`.
*  <b>`num_buckets`</b>: An `int` that is `>= 1`. The number of buckets.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `int64`.
  A Tensor of the same shape as the input `string_tensor`.



## Joining

String joining ops concatenate elements of input string tensors to produce a new
string tensor.

- - -

### `tf.reduce_join(inputs, axis=None, keep_dims=False, separator='', name=None, reduction_indices=None)` {#reduce_join}

Joins a string Tensor across the given dimensions.

Computes the string join across dimensions in the given string Tensor of shape
`[d_0, d_1, ..., d_n-1]`.  Returns a new Tensor created by joining the input
strings with the given separator (default: empty string).  Negative indices are
counted backwards from the end, with `-1` being equivalent to `n - 1`.

For example:

```
# tensor `a` is [["a", "b"], ["c", "d"]]
tf.reduce_join(a, 0) ==> ["ac", "bd"]
tf.reduce_join(a, 1) ==> ["ab", "cd"]
tf.reduce_join(a, -2) = tf.reduce_join(a, 0) ==> ["ac", "bd"]
tf.reduce_join(a, -1) = tf.reduce_join(a, 1) ==> ["ab", "cd"]
tf.reduce_join(a, 0, keep_dims=True) ==> [["ac", "bd"]]
tf.reduce_join(a, 1, keep_dims=True) ==> [["ab"], ["cd"]]
tf.reduce_join(a, 0, separator=".") ==> ["a.c", "b.d"]
tf.reduce_join(a, [0, 1]) ==> ["acbd"]
tf.reduce_join(a, [1, 0]) ==> ["abcd"]
tf.reduce_join(a, []) ==> ["abcd"]
```

##### Args:


*  <b>`inputs`</b>: A `Tensor` of type `string`.
    The input to be joined.  All reduced indices must have non-zero size.
*  <b>`axis`</b>: A `Tensor` of type `int32`.
    The dimensions to reduce over.  Dimensions are reduced in the
    order specified.  Omitting `axis` is equivalent to passing
    `[n-1, n-2, ..., 0]`.  Negative indices from `-n` to `-1` are supported.
*  <b>`keep_dims`</b>: An optional `bool`. Defaults to `False`.
    If `True`, retain reduced dimensions with length `1`.
*  <b>`separator`</b>: An optional `string`. Defaults to `""`.
    The separator to use when joining.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `string`.
  Has shape equal to that of the input with reduced dimensions removed or
  set to `1` depending on `keep_dims`.


- - -

### `tf.string_join(inputs, separator=None, name=None)` {#string_join}

Joins the strings in the given list of string tensors into one tensor;

with the given separator (default is an empty separator).

##### Args:


*  <b>`inputs`</b>: A list of at least 1 `Tensor` objects of type `string`.
    A list of string tensors.  The tensors must all have the same shape,
    or be scalars.  Scalars may be mixed in; these will be broadcast to the shape
    of non-scalar inputs.
*  <b>`separator`</b>: An optional `string`. Defaults to `""`.
    string, an optional join separator.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `string`.



## Splitting

- - -

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


- - -

### `tf.substr(input, pos, len, name=None)` {#substr}

Return substrings from `Tensor` of strings.

For each string in the input `Tensor`, creates a substring starting at index
`pos` with a total length of `len`.

If `len` defines a substring that would extend beyond the length of the input
string, then as many characters as possible are used.

If `pos` is negative or specifies a character index larger than any of the input
strings, then an `InvalidArgumentError` is thrown.

`pos` and `len` must have the same shape, otherwise a `ValueError` is thrown on
Op creation.

*NOTE*: `Substr` supports broadcasting up to two dimensions. More about
broadcasting
[here](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

---

Examples

Using scalar `pos` and `len`:

```
input = [b'Hello', b'World']
position = 1
length = 3

output = [b'ell', b'orl']
```

Using `pos` and `len` with same shape as `input`:

```
input = [[b'ten', b'eleven', b'twelve'],
         [b'thirteen', b'fourteen', b'fifteen'],
         [b'sixteen', b'seventeen', b'eighteen']]
position = [[1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]]
length =   [[2, 3, 4],
            [4, 3, 2],
            [5, 5, 5]]

output = [[b'en', b'eve', b'lve'],
          [b'hirt', b'urt', b'te'],
          [b'ixtee', b'vente', b'hteen']]
```

Broadcasting `pos` and `len` onto `input`:

```
input = [[b'ten', b'eleven', b'twelve'],
         [b'thirteen', b'fourteen', b'fifteen'],
         [b'sixteen', b'seventeen', b'eighteen'],
         [b'nineteen', b'twenty', b'twentyone']]
position = [1, 2, 3]
length =   [1, 2, 3]

output = [[b'e', b'ev', b'lve'],
          [b'h', b'ur', b'tee'],
          [b'i', b've', b'hte'],
          [b'i', b'en', b'nty']]
```

Broadcasting `input` onto `pos` and `len`:

```
input = b'thirteen'
position = [1, 5, 7]
length =   [3, 2, 1]

output = [b'hir', b'ee', b'n"]
```

##### Args:


*  <b>`input`</b>: A `Tensor` of type `string`. Tensor of strings
*  <b>`pos`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`.
    Scalar defining the position of first character in each substring
*  <b>`len`</b>: A `Tensor`. Must have the same type as `pos`.
    Scalar defining the number of characters to include in each substring
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `string`. Tensor of substrings



## Conversion

- - -

### `tf.as_string(input, precision=None, scientific=None, shortest=None, width=None, fill=None, name=None)` {#as_string}

Converts each entry in the given tensor to strings.  Supports many numeric

types and boolean.

##### Args:


*  <b>`input`</b>: A `Tensor`. Must be one of the following types: `int32`, `int64`, `complex64`, `float32`, `float64`, `bool`, `int8`.
*  <b>`precision`</b>: An optional `int`. Defaults to `-1`.
    The post-decimal precision to use for floating point numbers.
    Only used if precision > -1.
*  <b>`scientific`</b>: An optional `bool`. Defaults to `False`.
    Use scientific notation for floating point numbers.
*  <b>`shortest`</b>: An optional `bool`. Defaults to `False`.
    Use shortest representation (either scientific or standard) for
    floating point numbers.
*  <b>`width`</b>: An optional `int`. Defaults to `-1`.
    Pad pre-decimal numbers to this width.
    Applies to both floating point and integer numbers.
    Only used if width > -1.
*  <b>`fill`</b>: An optional `string`. Defaults to `""`.
    The value to pad if width > -1.  If empty, pads with spaces.
    Another typical value is '0'.  String cannot be longer than 1 character.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `string`.


- - -

### `tf.encode_base64(input, pad=None, name=None)` {#encode_base64}

Encode strings into web-safe base64 format.

Refer to the following article for more information on base64 format:
en.wikipedia.org/wiki/Base64. Base64 strings may have padding with '=' at the
end so that the encoded has length multiple of 4. See Padding section of the
link above.

Web-safe means that the encoder uses - and _ instead of + and /.

##### Args:


*  <b>`input`</b>: A `Tensor` of type `string`. Strings to be encoded.
*  <b>`pad`</b>: An optional `bool`. Defaults to `False`.
    Bool whether padding is applied at the ends.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `string`. Input strings encoded in base64.


- - -

### `tf.decode_base64(input, name=None)` {#decode_base64}

Decode web-safe base64-encoded strings.

Input may or may not have padding at the end. See EncodeBase64 for padding.
Web-safe means that input must use - and _ instead of + and /.

##### Args:


*  <b>`input`</b>: A `Tensor` of type `string`. Base64 strings to decode.
*  <b>`name`</b>: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `string`. Decoded strings.


