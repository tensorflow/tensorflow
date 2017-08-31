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

#include <string>

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/activation_mode.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

namespace {
// Return the string containing the list of valid activation modes, that can be
// used as an Attr() in REGISTER_OP.
string GetAllActivationModeAttrString() { return "activation_mode: {'Relu'}"; }

}  // namespace

// --------------------------------------------------------------------------
REGISTER_OP("FusedConv2DBiasActivation")
    .Input("input: T")
    .Input("filter: T")
    .Input("bias: T")
    .Output("output: T")
    .Attr("T: {float}")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr(GetAllActivationModeAttrString())
    .SetShapeFn(shape_inference::FusedConvBiasActivationShape)
    .Doc(R"doc(
    Computes a fused 2-D convolution, adds bias, and applies an activation function
    on the output given 4-D `input`, 4-D `filter`, 1-D `bias` tensors and an activation mode.

    input: A 4-D tensor. The dimension order is interpreted according to the value
        of `data_format`, see below for details.
    filter: A 4-D tensor of shape
        `[filter_height, filter_width, in_channels, out_channels]`
    bias: 1-D with size of the `out_channels` dimension in filter.
    output: A 4-D tensor. The dimension order is determined by the value of
        `data_format`, see below for details.
    T: The data type for the elements of input, filter, bias, and output Tensors.
    strides: 1-D tensor of length 4.  The stride of the sliding window for each
        dimension of `input`. The dimension order is determined by the value of
        `data_format`, see below for details.
    padding: The type of padding algorithm to use.
    data_format: Specify the data format of the input and output data. With the
        default format "NHWC", the data is stored in the order of:
        [batch, height, width, channels].
        Alternatively, the format could be "NCHW", the data storage order of:
        [batch, channels, height, width].
    activation_mode: Specify the activation function to apply to the output tensor
        of bias add. Currently only supports "Relu".
)doc");

}  // namespace tensorflow
