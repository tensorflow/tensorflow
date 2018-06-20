/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

REGISTER_OP("XlaReduceWindow")
    .Input("input: T")
    .Input("init_value: T")
    .Attr("T: numbertype")
    .Attr("computation: func")
    .Attr("window_dimensions: list(int)")
    .Attr("window_strides: list(int)")
    .Attr("padding_low: list(int)")
    .Attr("padding_high: list(int)")
    .Output("output: T")
    .SetShapeFn(shape_inference::UnknownShape)
    .Doc(R"doc(
Wraps the XLA ReduceWindow operator, documented at
 https://www.tensorflow.org/performance/xla/operation_semantics#reducewindow .

input: the input tensor
init_value: a scalar representing the initial value for the reduction
computation: a reducer function to apply
window_dimensions: the shape of the window
window_strides: the inter-window strides
padding_low: the padding to apply at the start of each input dimensions
padding_high: the padding to apply at the end of each input dimension.
)doc");

}  // namespace tensorflow
