/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

using shape_inference::InferenceContext;

REGISTER_OP("DecodeLibsvm")
    .Input("input: string")
    .Output("label: int64")
    .Output("feature: dtype")
    .Attr("dtype: {float, double, int32, int64} = DT_FLOAT")
    .Attr("num_features: int >= 1")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Convert LibSVM input to tensors. The output consists of
a label and a feature tensor. The shape of the label tensor
is the same as input and the shape of the feature tensor is
`[input_shape, num_features]`.

input: Each string is a record/row in the LibSVM.
label: A tensor of the same shape as input.
feature: A tensor of the shape `[input_shape, num_features]`.
num_features: The number of features.
)doc");

}  // namespace tensorflow
