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

// TODO(pauldonnelly): Add support for double inputs and scales to this Op,
// (currently Attr does not support double).

REGISTER_OP("FusedConv2DBiasActivation")
    .Input("conv_input: T")
    .Input("filter: T")
    .Input("bias: Tbias")
    .Input("side_input: T")
    .Output("output: T")
    .Attr("T: {float, half, qint8}")
    .Attr("Tbias: {float, half}")
    .Attr("conv_input_scale: float = 1.0")
    .Attr("side_input_scale: float = 0.0")
    .Attr("strides: list(int)")
    .Attr(GetPaddingAttrString())
    .Attr("data_format: {'NHWC', 'NCHW', 'NCHW_VECT_C'} = 'NHWC'")
    .Attr("filter_format: {'HWIO', 'OIHW', 'OIHW_VECT_I'} = 'HWIO'")
    .Attr("activation_mode: {'Relu'} = 'Relu'")
    .SetShapeFn(shape_inference::FusedConvBiasActivationShape)
    .Doc(R"doc(
    Computes a fused kernel which implements: 2-D convolution, adds side input,
    with separate scaling on convolution and side inputs, then adds bias and
    applies the RELU activation function to the result. Supports both float and
    qint8 data formats. In the case of qint8, the output is clipped to [0..127].

    conv_input: A tensor with format as specified by `data_format` (see below).
    filter: A tensor with format depending on `data_format` as follows:
        "NHWC", "NCHW":
             `float [ filter_height, filter_width, in_channels, out_channels ]`
        "NCHW_VECT_C":
             `qint8 [ out_channels, in_channels, filter_height, filter_width ]`
    bias: 1-D float tensor with size matching the `out_channels` dimension of
        `filter`.
        Note: this tensor is still float, even if other inputs are qint8.
    side_input: A tensor with format as specified by `data_format` (see below).
        This tensor will be ignored and can be [] if side_input_scale == 0.
        Otherwise, the size of each dimension must match the `output` tensor.
    output: A tensor with format as specified by `data_format` (see below).
        The dimension sizes are determined automatically based on other inputs
        and attributes.
    T: The element data type of `conv_input`, `side_input` and `output` tensors.
        Note: must match with the `data_format`.
    Tbias: The element data type of `bias`.
    conv_input_scale: scalar float value to be multiplied by `conv_input`.
        (conceptually.. in reality it is applied after convolution).
    side_input_scale: scalar float value to be multiplied by `side_input`.
    strides: 1-D tensor of length 4.  The stride of the sliding window for each
        dimension of `input`. The dimension order is determined by the value of
        `data_format`, see below for details.
        Note: the stride for batch and channel dimensions must be 1.
    padding: The type of padding algorithm to use.
    data_format: A string specifying the data format of `conv_input`,
        `side_input` and `output` tensors with the following options:
        "NHWC": `float [ batch, height, width, channels ]`
        "NCHW": `float [ batch, channels, height, width ]`
        "NCHW_VECT_C":
            `qint8 [ batch, channels / 4, height, width, channels % 4 ]`
        Note: for "NCHW_VECT_C", `channels` must be a multiple of 4.
    filter_format: A string specifying the data format of `filter`,
        "HWIO": `float [ kernel_height, kernel_width, input_channels,
                         output_channels ]`
        "OIHW_VECT_I":
            `qint8 [ output_channels, input_channels / 4,
                     kernel_height, kernel_width, input_channels % 4 ]`
    activation_mode: The activation applied to the output.
        Currently must be "Relu".
)doc");

}  // namespace tensorflow
