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

// See docs in ../ops/nn_ops.cc.

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/conv_grad_ops.h"

#include <algorithm>
#include <vector>

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"

namespace tensorflow {

Status ConvBackpropExtractAndVerifyDimension(
    StringPiece label, const TensorShape& input_shape,
    const TensorShape& filter_shape, const TensorShape& output_shape,
    const std::vector<int32>& strides, Padding padding, int spatial_dim,
    int filter_spatial_dim, ConvBackpropSpatialDimension* dim) {
  dim->input_size = input_shape.dim_size(spatial_dim);
  dim->filter_size = filter_shape.dim_size(filter_spatial_dim);
  dim->output_size = output_shape.dim_size(spatial_dim);
  dim->stride = strides[spatial_dim];
  int64 out_size = 0, pad_size = 0;
  TF_RETURN_IF_ERROR(GetWindowedOutputSize(dim->input_size, dim->filter_size,
                                           dim->stride, padding, &out_size,
                                           &pad_size));
  if (dim->output_size != out_size) {
    return errors::InvalidArgument(
        label, ": Size of out_backprop doesn't match computed: ", "actual = ",
        dim->output_size, ", computed = ", out_size);
  }

  dim->expanded_output_size = (dim->output_size - 1) * dim->stride + 1;
  const auto padded_out_size = dim->input_size + dim->filter_size - 1;
  dim->pad_before = dim->filter_size - 1 - pad_size;
  dim->pad_after =
      padded_out_size - dim->expanded_output_size - dim->pad_before;
  VLOG(2) << label << ": expanded_out = " << dim->expanded_output_size
          << ", filter = " << dim->filter_size
          << ", padded_out = " << padded_out_size
          << ", pad_before = " << dim->pad_before
          << ", pad_after = " << dim->pad_after
          << ", strides = " << dim->stride;
  return Status::OK();
}

Status Conv2DBackpropComputeDimensions(
    StringPiece label, const TensorShape& input_shape,
    const TensorShape& filter_shape, const TensorShape& out_backprop_shape,
    const std::vector<int32>& strides, Padding padding,
    TensorFormat data_format, Conv2DBackpropDimensions* dims) {
  if (input_shape.dims() != 4) {
    return errors::InvalidArgument(label, ": input must be 4-dimensional");
  }
  if (filter_shape.dims() != 4) {
    return errors::InvalidArgument(label, ": filter must be 4-dimensional");
  }
  if (out_backprop_shape.dims() != 4) {
    return errors::InvalidArgument(label,
                                   ": out_backprop must be 4-dimensional");
  }
  dims->batch_size = GetTensorDim(input_shape, data_format, 'N');
  if (dims->batch_size != GetTensorDim(out_backprop_shape, data_format, 'N')) {
    return errors::InvalidArgument(
        label, ": input and out_backprop must have the same batch size");
  }

  dims->in_depth = GetTensorDim(input_shape, data_format, 'C');
  if (dims->in_depth != filter_shape.dim_size(2)) {
    return errors::InvalidArgument(
        label, ": input and filter must have the same depth");
  }
  dims->out_depth = filter_shape.dim_size(3);
  if (dims->out_depth != GetTensorDim(out_backprop_shape, data_format, 'C')) {
    return errors::InvalidArgument(
        label, ": filter and out_backprop must have the same out_depth");
  }

  const int row_dim = GetTensorDimIndex(data_format, 'H');
  const int col_dim = GetTensorDimIndex(data_format, 'W');
  const int filter_row_dim = 0, filter_col_dim = 1;
  TF_RETURN_IF_ERROR(ConvBackpropExtractAndVerifyDimension(
      label, input_shape, filter_shape, out_backprop_shape, strides, padding,
      row_dim, filter_row_dim, &dims->rows));
  TF_RETURN_IF_ERROR(ConvBackpropExtractAndVerifyDimension(
      label, input_shape, filter_shape, out_backprop_shape, strides, padding,
      col_dim, filter_col_dim, &dims->cols));
  return Status::OK();
}

}  // namespace tensorflow
