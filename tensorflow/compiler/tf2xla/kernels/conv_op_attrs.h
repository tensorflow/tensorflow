/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_TF2XLA_KERNELS_CONV_OP_ATTRS_H_
#define TENSORFLOW_COMPILER_TF2XLA_KERNELS_CONV_OP_ATTRS_H_

#include "tensorflow/compiler/xla/client/lib/conv_op_helpers.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

// ConvOpAttrs contains all of the metadata necessary to specify a TF or XLA
// convolution.
struct ConvOpAttrs {
  // Constructs a ConvOpAttrs, reading most of the attributes from `ctx`.
  static xla::StatusOr<ConvOpAttrs> Create(
      int num_spatial_dims, bool depthwise,
      tensorflow::OpKernelConstruction* ctx);

  // Converts to the format required by the XLA convolution helpers.
  xla::StatusOr<xla::ConvOpAttrs> ToXla(const TensorShape& input_shape,
                                        const TensorShape& filter_shape) const;

  bool depthwise;
  int num_spatial_dims;
  std::vector<int32> dilations;
  std::vector<int32> strides;
  tensorflow::Padding padding;
  std::vector<int64> explicit_paddings;
  tensorflow::TensorFormat data_format;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_KERNELS_CONV_OP_ATTRS_H_
