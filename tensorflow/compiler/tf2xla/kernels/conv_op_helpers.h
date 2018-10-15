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

#ifndef TENSORFLOW_COMPILER_TF2XLA_KERNELS_CONV_OP_HELPERS_H_
#define TENSORFLOW_COMPILER_TF2XLA_KERNELS_CONV_OP_HELPERS_H_

#include <vector>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

// This header exposes utilities for translating TensorFlow convolution ops into
// XLA ops.
//
// conv_ops.cc contains lowerings for many of these TF convolution ops (e.g.
// Conv2D, Conv3DBackpropFilterV2), but you might want to use the utilities in
// this header to implement a new and exciting convolution op, for example a
// fused TensorFlow op that contains a convolution and other things.

namespace tensorflow {

// ConvOpAttrs contains all of the metadata necessary to specify a TF or XLA
// convolution.
struct ConvOpAttrs {
  // Constructs a ConvOpAttrs, reading most of the attributes from `ctx`.
  static xla::StatusOr<ConvOpAttrs> Create(int num_spatial_dims, bool depthwise,
                                           OpKernelConstruction* ctx);

  bool depthwise;
  int num_spatial_dims;
  std::vector<int32> dilations;
  std::vector<int32> strides;
  Padding padding;
  TensorFormat data_format;
};

// Creates a new XLA forward or backward convolution with the given inputs and
// attributes.
xla::StatusOr<xla::XlaOp> MakeXlaForwardConvOp(StringPiece type_string,
                                               xla::XlaOp conv_input,
                                               xla::XlaOp filter,
                                               const ConvOpAttrs& attrs);
xla::StatusOr<xla::XlaOp> MakeXlaBackpropInputConvOp(
    StringPiece type_string, const xla::Shape& input_shape, xla::XlaOp filter,
    xla::XlaOp out_backprop, const ConvOpAttrs& attrs);
xla::StatusOr<xla::XlaOp> MakeXlaBackpropFilterConvOp(
    StringPiece type_string, xla::XlaOp activations,
    const xla::Shape& filter_shape, xla::XlaOp gradients,
    const ConvOpAttrs& attrs);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_KERNELS_CONV_OP_HELPERS_H_
