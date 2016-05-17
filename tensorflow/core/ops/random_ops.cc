/* Copyright 2015 Google Inc. All Rights Reserved.

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

REGISTER_OP("RandomUniform")
    .Input("shape: T")
    .SetIsStateful()
    .Output("output: dtype")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("dtype: {half,float,double}")
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

REGISTER_OP("RandomUniformInt")
    .Input("shape: T")
    .Input("minval: Tout")
    .Input("maxval: Tout")
    .SetIsStateful()
    .Output("output: Tout")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("Tout: {int32, int64}")
    .Attr("T: {int32, int64}")
    .Doc(R"doc(
Outputs random integers from a uniform distribution.

The generated values are uniform integers in the range `[minval, maxval)`.
The lower bound `minval` is included in the range, while the upper bound
`maxval` is excluded.

The random integers are slightly biased unless `maxval - minval` is an exact
power of two.  The bias is small for values of `maxval - minval` significantly
smaller than the range of the output (either `2^32` or `2^64`).

shape: The shape of the output tensor.
minval: 0-D.  Inclusive lower bound on the generated integers.
maxval: 0-D.  Exclusive upper bound on the generated integers.
seed: If either `seed` or `seed2` are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: A second seed to avoid seed collision.

output: A tensor of the specified shape filled with uniform random integers.
)doc");

REGISTER_OP("RandomStandardNormal")
    .Input("shape: T")
    .SetIsStateful()
    .Output("output: dtype")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("dtype: {half,float,double}")
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
    .Attr("dtype: {half,float,double}")
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

REGISTER_OP("Multinomial")
    .SetIsStateful()
    .Input("logits: T")
    .Input("num_samples: int32")
    .Output("output: int64")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("T: realnumbertype")
    .Doc(R"doc(
Draws samples from a multinomial distribution.

logits: 2-D Tensor with shape `[batch_size, num_classes]`.  Each slice `[i, :]`
  represents the unnormalized log probabilities for all classes.
num_samples: 0-D.  Number of independent samples to draw for each row slice.
seed: If either seed or seed2 is set to be non-zero, the internal random number
  generator is seeded by the given seed.  Otherwise, a random seed is used.
seed2: A second seed to avoid seed collision.
output: 2-D Tensor with shape `[batch_size, num_samples]`.  Each slice `[i, :]`
  contains the drawn class labels with range `[0, num_classes)`.
)doc");

}  // namespace tensorflow
