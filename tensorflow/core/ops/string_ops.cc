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

#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("StringToHashBucketFast")
    .Input("input: string")
    .Output("output: int64")
    .Attr("num_buckets: int >= 1")
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
time than tf.string_to_hash_bucket_fast.

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
    .Deprecated(10,
                "Use `tf.string_to_hash_bucket_fast()` or "
                "`tf.string_to_hash_bucket_strong()`")
    .Doc(R"doc(
Converts each string in the input Tensor to its hash mod by a number of buckets.

The hash function is deterministic on the content of the string within the
process.

Note that the hash function may change from time to time.

num_buckets: The number of buckets.
output: A Tensor of the same shape as the input `string_tensor`.
)doc");

REGISTER_OP("ReduceJoin")
    .Input("inputs: string")
    .Input("reduction_indices: int32")
    .Attr("keep_dims: bool = false")
    .Attr("separator: string = ''")
    .Output("output: string")
    .Doc(R"doc(
Joins a string Tensor across the given dimensions.

Computes the string join across dimensions in the given string Tensor of shape
`[d_0, d_1, ..., d_n-1]`.  Returns a new Tensor created by joining the input
strings with the given separator (default: empty string).  Negative indices are
counted backwards from the end, with `-1` being equivalent to `n - 1`.  Passing
an empty `reduction_indices` joins all strings in linear index order and outputs
a scalar string.


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
  order specified.  If `reduction_indices` has higher rank than `1`, it is
  flattened.  Omitting `reduction_indices` is equivalent to passing
  `[n-1, n-2, ..., 0]`.  Negative indices from `-n` to `-1` are supported.
keep_dims: If `True`, retain reduced dimensions with length `1`.
separator: The separator to use when joining.

output: Has shape equal to that of the input with reduced dimensions removed or
  set to `1` depending on `keep_dims`.
)doc");

}  // namespace tensorflow
