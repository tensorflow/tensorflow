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

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::ShapeHandle;

static Status StatelessShape(shape_inference::InferenceContext* context) {
  // Check seed shape
  ShapeHandle seed;
  TF_RETURN_IF_ERROR(context->WithRank(context->input(1), 1, &seed));
  DimensionHandle unused;
  TF_RETURN_IF_ERROR(context->WithValue(context->Dim(seed, 0), 2, &unused));

  // Set output shape
  shape_inference::ShapeHandle out;
  TF_RETURN_IF_ERROR(context->MakeShapeFromShapeTensor(0, &out));
  context->set_output(0, out);
  return Status::OK();
}

#define REGISTER_STATELESS_OP(name)                  \
  REGISTER_OP(name)                                  \
      .Input("shape: T")                             \
      .Input("seed: int64")                          \
      .Output("output: dtype")                       \
      .Attr("dtype: {half,float,double} = DT_FLOAT") \
      .Attr("T: {int32, int64} = DT_INT32")          \
      .SetShapeFn(StatelessShape)

// This op is exposed through contrib/stateless only.  The interface may change.
REGISTER_STATELESS_OP("StatelessRandomUniform").Doc(R"doc(
Outputs deterministic pseudorandom random values from a uniform distribution.

The generated values follow a uniform distribution in the range `[0, 1)`. The
lower bound 0 is included in the range, while the upper bound 1 is excluded.

The outputs are a deterministic function of `shape` and `seed`.

shape: The shape of the output tensor.
dtype: The type of the output.
seed: 2 seeds (shape [2]).
output: Random values with specified shape.
)doc");

// This op is exposed through contrib/stateless only.  The interface may change.
REGISTER_STATELESS_OP("StatelessRandomNormal").Doc(R"doc(
Outputs deterministic pseudorandom values from a normal distribution.

The generated values will have mean 0 and standard deviation 1.

The outputs are a deterministic function of `shape` and `seed`.

shape: The shape of the output tensor.
dtype: The type of the output.
seed: 2 seeds (shape [2]).
output: Random values with specified shape.
)doc");

// This op is exposed through contrib/stateless only.  The interface may change.
REGISTER_STATELESS_OP("StatelessTruncatedNormal").Doc(R"doc(
Outputs deterministic pseudorandom values from a truncated normal distribution.

The generated values follow a normal distribution with mean 0 and standard
deviation 1, except that values whose magnitude is more than 2 standard
deviations from the mean are dropped and re-picked.

The outputs are a deterministic function of `shape` and `seed`.

shape: The shape of the output tensor.
dtype: The type of the output.
seed: 2 seeds (shape [2]).
output: Random values with specified shape.
)doc");

#undef REGISTER_STATELESS_OP

}  // namespace tensorflow
