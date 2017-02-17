/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

REGISTER_OP("StringToHashBucketFast")
    .Input("input: string")
    .Output("output: int64")
    .Attr("num_buckets: int >= 1")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Converts each string in the input Tensor to its hash mod by a number of buckets.

The hash function is deterministic on the content of the string within the
process and will never change. However, it is not suitable for cryptography.
This function may be used when CPU time is scarce and inputs are trusted or
unimportant. There is a risk of adversaries constructing inputs that all hash
to the same bucket. To prevent this problem, use a strong hash function with
`tf.string_to_hash_bucket_strong`.

input: The strings to assign a hash bucket.
num_buckets: The number of buckets.
output: A Tensor of the same shape as the input `string_tensor`.
)doc");

REGISTER_OP("StringToHashBucketStrong")
    .Input("input: string")
    .Output("output: int64")
    .Attr("num_buckets: int >= 1")
    .Attr("key: list(int)")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
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

input: The strings to assign a hash bucket.
num_buckets: The number of buckets.
key: The key for the keyed hash function passed as a list of two uint64
  elements.
output: A Tensor of the same shape as the input `string_tensor`.
)doc");

REGISTER_OP("StringToHashBucket")
    .Input("string_tensor: string")
    .Output("output: int64")
    .Attr("num_buckets: int >= 1")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Converts each string in the input Tensor to its hash mod by a number of buckets.

The hash function is deterministic on the content of the string within the
process.

Note that the hash function may change from time to time.
This functionality will be deprecated and it's recommended to use
`tf.string_to_hash_bucket_fast()` or `tf.string_to_hash_bucket_strong()`.

num_buckets: The number of buckets.
output: A Tensor of the same shape as the input `string_tensor`.
)doc");

REGISTER_OP("ReduceJoin")
    .Input("inputs: string")
    .Input("reduction_indices: int32")
    .Attr("keep_dims: bool = false")
    .Attr("separator: string = ''")
    .Output("output: string")
    .SetShapeFn(shape_inference::ReductionShape)
    .Doc(R"doc(
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

inputs: The input to be joined.  All reduced indices must have non-zero size.
reduction_indices: The dimensions to reduce over.  Dimensions are reduced in the
  order specified.  Omitting `reduction_indices` is equivalent to passing
  `[n-1, n-2, ..., 0]`.  Negative indices from `-n` to `-1` are supported.
keep_dims: If `True`, retain reduced dimensions with length `1`.
separator: The separator to use when joining.

output: Has shape equal to that of the input with reduced dimensions removed or
  set to `1` depending on `keep_dims`.
)doc");

REGISTER_OP("AsString")
    .Input("input: T")
    .Output("output: string")
    .Attr("T: {int32, int64, complex64, float, double, bool, int8}")
    .Attr("precision: int = -1")
    .Attr("scientific: bool = false")
    .Attr("shortest: bool = false")
    .Attr("width: int = -1")
    .Attr("fill: string = ''")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Converts each entry in the given tensor to strings.  Supports many numeric
types and boolean.

precision: The post-decimal precision to use for floating point numbers.
  Only used if precision > -1.
scientific: Use scientific notation for floating point numbers.
shortest: Use shortest representation (either scientific or standard) for
  floating point numbers.
width: Pad pre-decimal numbers to this width.
  Applies to both floating point and integer numbers.
  Only used if width > -1.
fill: The value to pad if width > -1.  If empty, pads with spaces.
  Another typical value is '0'.  String cannot be longer than 1 character.
)doc");

REGISTER_OP("StringJoin")
    .Input("inputs: N * string")
    .Attr("N: int")
    .Attr("separator: string = ''")
    .Output("output: string")
    .SetShapeFn([](InferenceContext* c) {
      // If all inputs are scalars, then return a scalar.
      bool all_scalar = true;
      for (int i = 0; i < c->num_inputs(); ++i) {
        if (c->Rank(c->input(i)) != 0) all_scalar = false;
      }
      if (all_scalar) {
        c->set_output(0, c->Scalar());
        return Status::OK();
      }

      // At least one input is unknown or a scalar.
      // Merge the non-scalars to find the output shape.
      // Don't merge inputs with unknown rank, as they can actually be scalars
      // or the output shape.
      ShapeHandle out = c->UnknownShape();
      for (int i = 0; i < c->num_inputs(); ++i) {
        if (c->RankKnown(c->input(i)) && c->Rank(c->input(i)) != 0) {
          TF_RETURN_IF_ERROR(c->Merge(out, c->input(i), &out));
        }
      }
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
Joins the strings in the given list of string tensors into one tensor;
with the given separator (default is an empty separator).

inputs: A list of string tensors.  The tensors must all have the same shape,
  or be scalars.  Scalars may be mixed in; these will be broadcast to the shape
  of non-scalar inputs.
separator: string, an optional join separator.
)doc");

REGISTER_OP("StringSplit")
    .Input("input: string")
    .Input("delimiter: string")
    .Output("indices: int64")
    .Output("values: string")
    .Output("shape: int64")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));

      c->set_output(0, c->Matrix(InferenceContext::kUnknownDim, 2));
      c->set_output(1, c->Vector(InferenceContext::kUnknownDim));
      c->set_output(2, c->Vector(2));
      return Status::OK();
    })
    .Doc(R"doc(
Split elements of `input` based on `delimiter` into a `SparseTensor`.

Let N be the size of source (typically N will be the batch size). Split each
element of `input` based on `delimiter` and return a `SparseTensor`
containing the splitted tokens. Empty tokens are ignored.

`delimiter` can be empty, or a string of split characters. If `delimiter` is an
 empty string, each element of `input` is split into individual single-byte
 character strings, including splitting of UTF-8 multibyte sequences. Otherwise
 every character of `delimiter` is a potential split point.

For example:
  N = 2, input[0] is 'hello world' and input[1] is 'a b c', then the output
  will be

  indices = [0, 0;
             0, 1;
             1, 0;
             1, 1;
             1, 2]
  shape = [2, 3]
  values = ['hello', 'world', 'a', 'b', 'c']

input: 1-D. Strings to split.
delimiter: 0-D. Delimiter characters (bytes), or empty string.
indices: A dense matrix of int64 representing the indices of the sparse tensor.
values: A vector of strings corresponding to the splited values.
shape: a length-2 vector of int64 representing the shape of the sparse
  tensor, where the first value is N and the second value is the maximum number
  of tokens in a single input entry.
)doc");

REGISTER_OP("EncodeBase64")
    .Input("input: string")
    .Output("output: string")
    .Attr("pad: bool = false")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Encode strings into web-safe base64 format.

Refer to the following article for more information on base64 format:
en.wikipedia.org/wiki/Base64. Base64 strings may have padding with '=' at the
end so that the encoded has length multiple of 4. See Padding section of the
link above.

Web-safe means that the encoder uses - and _ instead of + and /.

input: Strings to be encoded.
output: Input strings encoded in base64.
pad: Bool whether padding is applied at the ends.
)doc");

REGISTER_OP("DecodeBase64")
    .Input("input: string")
    .Output("output: string")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Decode web-safe base64-encoded strings.

Input may or may not have padding at the end. See EncodeBase64 for padding.
Web-safe means that input must use - and _ instead of + and /.

input: Base64 strings to decode.
output: Decoded strings.
)doc");

REGISTER_OP("Substr")
    .Input("input: string")
    .Input("pos: T")
    .Input("len: T")
    .Output("output: string")
    .Attr("T: {int32, int64}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle pos_shape = c->input(1);
      ShapeHandle len_shape = c->input(2);
      ShapeHandle unused;
      // Check that pos/len have same rank
      TF_RETURN_IF_ERROR(c->WithRank(pos_shape, c->Rank(len_shape), &unused));
      // Check that dimensions are equal
      for (int32 i = 0; i < c->Rank(pos_shape); ++i) {
        DimensionHandle pos_dim = c->Dim(pos_shape, i);
        DimensionHandle len_dim = c->Dim(len_shape, i);
        if (c->Value(pos_dim) != c->Value(len_dim)) {
          return errors::InvalidArgument("pos and len shapes must match: ",
                                         c->DebugString(pos_shape), " vs. ",
                                         c->DebugString(len_shape));
        }
      }
      // c->input(0) is the ShapeHandle to input strings
      // BroadcastBinaryOpShapeFn infers shape from c->input(0) and c->input(1).
      return shape_inference::BroadcastBinaryOpShapeFn(c);
    })
    .Doc(R"doc(
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

input: Tensor of strings 
pos: Scalar defining the position of first character in each substring
len: Scalar defining the number of characters to include in each substring
output: Tensor of substrings
)doc");

}  // namespace tensorflow
