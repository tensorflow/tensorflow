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
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/mirror_pad_mode.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

// For now, this file only includes MKL quantized ops. In the
// future, we will move all other MKL ops from nn_ops.cc to this file.

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;



REGISTER_OP("_KdnnSigmoid")
    .Input("logits: T")
    .Output("sigmoid: T")
    .Attr("T: {float} = DT_FLOAT")
    .SetShapeFn(shape_inference::UnchangedShape)
    .Doc(R"doc(
KDNN version of sigmoid operator. Uses KDNN APIs to perform sigmoid
operation.

*NOTE*: Do not invoke this operator directly in Python. Graph rewrite pass is
expected to invoke these operators.
)doc");

}  // namespace tensorflow
