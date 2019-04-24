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

#ifndef TENSORFLOW_COMPILER_XLA_CLIENT_LIB_CONV_OP_HELPERS_H_
#define TENSORFLOW_COMPILER_XLA_CLIENT_LIB_CONV_OP_HELPERS_H_

#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/statusor.h"

// This header exposes utilities for translating TensorFlow convolution ops into
// XLA ops.

namespace xla {

// ConvOpAttrs contains all of the metadata necessary to specify an XLA
// convolution.
struct ConvOpAttrs {
  bool depthwise;
  int num_spatial_dims;
  std::vector<int32> dilations;
  std::vector<int32> strides;
  std::vector<int64> explicit_paddings;
  ConvolutionDimensionNumbers data_format;
};

// Computes the convolution with the given input, filter and attributes. Errors
// returned by this function and the ones below are tagged with "type_string",
// which is the name of the TensorFlow operator using them.
StatusOr<XlaOp> MakeXlaForwardConvOp(
    absl::string_view type_string, XlaOp conv_input, XlaOp filter,
    const ConvOpAttrs& attrs,
    const PrecisionConfig* precision_config = nullptr);
// Computes the gradient with respect to the input, given the output gradient
// and the filter.
StatusOr<XlaOp> MakeXlaBackpropInputConvOp(
    absl::string_view type_string, const Shape& input_shape, XlaOp filter,
    XlaOp out_backprop, const ConvOpAttrs& attrs,
    const PrecisionConfig* precision_config = nullptr);
// Computes the gradient with respect to the filter, given the output gradient
// and the activations.
StatusOr<XlaOp> MakeXlaBackpropFilterConvOp(
    absl::string_view type_string, XlaOp activations, const Shape& filter_shape,
    XlaOp out_backprop, const ConvOpAttrs& attrs,
    const PrecisionConfig* precision_config = nullptr);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_CLIENT_LIB_CONV_OP_HELPERS_H_
