#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("StringToHashBucket")
    .Input("string_tensor: string")
    .Output("output: int64")
    .Attr("num_buckets: int >= 1")
    .Doc(R"doc(
Converts each string in the input Tensor to its hash mod by a number of buckets.

The hash function is deterministic on the content of the string within the
process.

Note that the hash function may change from time to time.

num_buckets: The number of buckets.
output: A Tensor of the same shape as the input string_tensor.
)doc");

}  // namespace tensorflow
