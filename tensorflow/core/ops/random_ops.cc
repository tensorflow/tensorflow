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

REGISTER_OP("RandomUniform")
    .Input("shape: T")
    .SetIsStateful()
    .Output("output: dtype")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("dtype: {half,float,double}")
    .Attr("T: {int32, int64}")
    .SetShapeFn(shape_inference::RandomShape)
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
    .SetShapeFn(shape_inference::RandomShape)
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
    .SetShapeFn(shape_inference::RandomShape)
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

REGISTER_OP("ParameterizedTruncatedNormal")
    .Input("shape: T")
    .Input("means: dtype")
    .Input("stdevs: dtype")
    .Input("minvals: dtype")
    .Input("maxvals: dtype")
    .SetIsStateful()
    .Output("output: dtype")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("dtype: {half,float,double}")
    .Attr("T: {int32, int64}")
    .SetShapeFn(shape_inference::RandomShape)
    .Doc(R"doc(
Outputs random values from a normal distribution. The parameters may each be a
scalar which applies to the entire output, or a vector of length shape[0] which
stores the parameters for each batch.

shape: The shape of the output tensor. Batches are indexed by the 0th dimension.
means: The mean parameter of each batch.
stdevs: The standard deviation parameter of each batch. Must be greater than 0.
minvals: The minimum cutoff. May be -infinity.
maxvals: The maximum cutoff. May be +infinity, and must be more than the minval
  for each batch.
dtype: The type of the output.
seed: If either `seed` or `seed2` are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: A second seed to avoid seed collision.

output: A matrix of shape num_batches x samples_per_batch, filled with random
  truncated normal values using the parameters for each row.
)doc");

REGISTER_OP("TruncatedNormal")
    .Input("shape: T")
    .SetIsStateful()
    .Output("output: dtype")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("dtype: {half,float,double}")
    .Attr("T: {int32, int64}")
    .SetShapeFn(shape_inference::RandomShape)
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
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
Randomly shuffles a tensor along its first dimension.

  The tensor is shuffled along dimension 0, such that each `value[j]` is mapped
  to one and only one `output[i]`. For example, a mapping that might occur for a
  3x2 tensor is:

```
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
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle logits_shape;
      ShapeHandle unused;
      DimensionHandle num_samples;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &logits_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->MakeDimForScalarInput(1, &num_samples));
      c->set_output(0, c->Matrix(c->Dim(logits_shape, 0), num_samples));
      return Status::OK();
    })
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

REGISTER_OP("RandomGamma")
    .SetIsStateful()
    .Input("shape: S")
    .Input("alpha: T")
    .Output("output: T")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("S: {int32, int64}")
    .Attr("T: {half, float, double}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
      TF_RETURN_IF_ERROR(c->Concatenate(out, c->input(1), &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
Outputs random values from the Gamma distribution(s) described by alpha.

This op uses the algorithm by Marsaglia et al. to acquire samples via
transformation-rejection from pairs of uniform and normal random variables.
See http://dl.acm.org/citation.cfm?id=358414

shape: 1-D integer tensor. Shape of independent samples to draw from each
  distribution described by the shape parameters given in alpha.
alpha: A tensor in which each scalar is a "shape" parameter describing the
  associated gamma distribution.
seed: If either `seed` or `seed2` are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: A second seed to avoid seed collision.

output: A tensor with shape `shape + shape(alpha)`. Each slice
  `[:, ..., :, i0, i1, ...iN]` contains the samples drawn for
  `alpha[i0, i1, ...iN]`. The dtype of the output matches the dtype of alpha.
)doc");

REGISTER_OP("RandomPoisson")
    .SetIsStateful()
    .Input("shape: S")
    .Input("rate: dtype")
    .Output("output: dtype")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("S: {int32, int64}")
    .Attr("dtype: {half, float, double}")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
      TF_RETURN_IF_ERROR(c->Concatenate(out, c->input(1), &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Deprecated(25, "Replaced by RandomPoissonV2")
    .Doc(R"doc(
Use RandomPoissonV2 instead.
)doc");

REGISTER_OP("RandomPoissonV2")
    .SetIsStateful()
    .Input("shape: S")
    .Input("rate: R")
    .Output("output: dtype")
    .Attr("seed: int = 0")
    .Attr("seed2: int = 0")
    .Attr("S: {int32, int64}")
    .Attr("R: {half, float, double, int32, int64} = DT_DOUBLE")
    .Attr("dtype: {half, float, double, int32, int64} = DT_INT64")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle out;
      TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &out));
      TF_RETURN_IF_ERROR(c->Concatenate(out, c->input(1), &out));
      c->set_output(0, out);
      return Status::OK();
    })
    .Doc(R"doc(
Outputs random values from the Poisson distribution(s) described by rate.

This op uses two algorithms, depending on rate. If rate >= 10, then
the algorithm by Hormann is used to acquire samples via
transformation-rejection.
See http://www.sciencedirect.com/science/article/pii/0167668793909974.

Otherwise, Knuth's algorithm is used to acquire samples via multiplying uniform
random variables.
See Donald E. Knuth (1969). Seminumerical Algorithms. The Art of Computer
Programming, Volume 2. Addison Wesley

shape: 1-D integer tensor. Shape of independent samples to draw from each
  distribution described by the shape parameters given in rate.
rate: A tensor in which each scalar is a "rate" parameter describing the
  associated poisson distribution.
seed: If either `seed` or `seed2` are set to be non-zero, the random number
  generator is seeded by the given seed.  Otherwise, it is seeded by a
  random seed.
seed2: A second seed to avoid seed collision.

output: A tensor with shape `shape + shape(rate)`. Each slice
  `[:, ..., :, i0, i1, ...iN]` contains the samples drawn for
  `rate[i0, i1, ...iN]`.
)doc");

}  // namespace tensorflow
