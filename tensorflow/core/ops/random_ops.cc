#include "tensorflow/core/framework/op.h"

namespace tensorflow {

REGISTER_OP("RandomUniform")
    .Input("shape: T")
    .SetIsStateful()
    .Output("output: dtype")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("dtype: {float,double}")
    .Attr("T: {int32, int64}")
    .Doc(R"doc(
Outputs random values from a uniform distribution.

The generated values follow a uniform distribution in the range `[0, 1)`. The
lower bound 0 is included in the range, while the upper bound 1 is excluded.

shape: The shape of the output tensor.
dtype: The type of the output.
seed: If either `seed` or `seed2` are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: A second seed to avoid seed collision.

output: A tensor of the specified shape filled with uniform random values.
)doc");

REGISTER_OP("RandomStandardNormal")
    .Input("shape: T")
    .SetIsStateful()
    .Output("output: dtype")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("dtype: {float,double}")
    .Attr("T: {int32, int64}")
    .Doc(R"doc(
Outputs random values from a normal distribution.

The generated values will have mean 0 and standard deviation 1.

shape: The shape of the output tensor.
dtype: The type of the output.
seed: If either `seed` or `seed2` are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: A second seed to avoid seed collision.

output: A tensor of the specified shape filled with random normal values.
)doc");

REGISTER_OP("TruncatedNormal")
    .Input("shape: T")
    .SetIsStateful()
    .Output("output: dtype")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("dtype: {float,double}")
    .Attr("T: {int32, int64}")
    .Doc(R"doc(
Outputs random values from a truncated normal distribution.

The generated values follow a normal distribution with mean 0 and standard
deviation 1, except that values whose magnitude is more than 2 standard
deviations from the mean are dropped and re-picked.

shape: The shape of the output tensor.
dtype: The type of the output.
seed: If either `seed` or `seed2` are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: A second seed to avoid seed collision.

output: A tensor of the specified shape filled with random truncated normal
  values.
)doc");

REGISTER_OP("RandomShuffle")
    .Input("value: T")
    .SetIsStateful()
    .Output("output: T")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("T: type")
    .Doc(R"doc(
Randomly shuffles a tensor along its first dimension.

  The tensor is shuffled along dimension 0, such that each `value[j]` is mapped
  to one and only one `output[i]`. For example, a mapping that might occur for a
  3x2 tensor is:

```prettyprint
[[1, 2],       [[5, 6],
 [3, 4],  ==>   [1, 2],
 [5, 6]]        [3, 4]]
```

value: The tensor to be shuffled.
seed: If either `seed` or `seed2` are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: A second seed to avoid seed collision.

output: A tensor of same shape and type as `value`, shuffled along its first
  dimension.
)doc");

}  // namespace tensorflow
