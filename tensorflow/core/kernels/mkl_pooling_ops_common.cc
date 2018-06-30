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

#ifdef INTEL_MKL

#include "tensorflow/core/kernels/mkl_pooling_ops_common.h"
#include <limits>
#include <vector>
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/kernels/bounds_check.h"

namespace tensorflow {

// Initialization for TensorFlow format
void MklPoolParameters::Init(OpKernelContext* context,
                             const std::vector<int32>& ksize,
                             const std::vector<int32>& stride, Padding padding,
                             TensorFormat data_format,
                             const TensorShape& tensor_in_shape) {
  // For maxpooling, tensor_in should have 4 dimensions.
  OP_REQUIRES(context, tensor_in_shape.dims() == 4,
              errors::InvalidArgument("tensor_in must be 4-dimensional"));

  depth = GetTensorDim(tensor_in_shape, data_format, 'C');
  tensor_in_cols = GetTensorDim(tensor_in_shape, data_format, 'W');
  tensor_in_rows = GetTensorDim(tensor_in_shape, data_format, 'H');
  tensor_in_batch = GetTensorDim(tensor_in_shape, data_format, 'N');

  Init(context, ksize, stride, padding, data_format);
}

#ifdef INTEL_MKL_ML
// Initialization for MKL format
void MklPoolParameters::Init(OpKernelContext* context,
                             const std::vector<int32>& ksize,
                             const std::vector<int32>& stride, Padding padding,
                             TensorFormat data_format,
                             const MklShape* mklInputShape) {
  // Get the input sizes
  depth = mklInputShape->GetSizes()[2];
  tensor_in_cols = mklInputShape->GetSizes()[0];
  tensor_in_rows = mklInputShape->GetSizes()[1];
  tensor_in_batch = mklInputShape->GetSizes()[3];

  Init(context, ksize, stride, padding, data_format);
}
#else
// Initialization for MKL format
void MklPoolParameters::Init(OpKernelContext* context,
                             const std::vector<int32>& ksize,
                             const std::vector<int32>& stride, Padding padding,
                             TensorFormat data_format,
                             const MklDnnShape* mklInputShape) {
  // Get the input sizes
  depth = mklInputShape->GetDimension('C');
  tensor_in_cols = mklInputShape->GetDimension('W');
  tensor_in_rows = mklInputShape->GetDimension('H');
  tensor_in_batch = mklInputShape->GetDimension('N');

  Init(context, ksize, stride, padding, data_format);
}
#endif  // INTEL_MKL_ML
// Common Initialization for TensorFlow and MKL formats
void MklPoolParameters::Init(OpKernelContext* context,
                             const std::vector<int32>& ksize,
                             const std::vector<int32>& stride, Padding padding,
                             TensorFormat data_format) {
  // Get the data format
  this->data_format = data_format;

  // Get the output sizes
  window_rows = GetTensorDim(ksize, data_format, 'H');
  window_cols = GetTensorDim(ksize, data_format, 'W');
  depth_window = GetTensorDim(ksize, data_format, 'C');

  // Get the strides
  row_stride = GetTensorDim(stride, data_format, 'H');
  col_stride = GetTensorDim(stride, data_format, 'W');
  depth_stride = GetTensorDim(stride, data_format, 'C');

  // We only support 2D pooling across width/height and depthwise
  // pooling, not a combination.
  OP_REQUIRES(context,
              (depth_window == 1 || (window_rows == 1 && window_cols == 1)),
              errors::Unimplemented(
                  "MaxPooling supports exactly one of pooling across depth "
                  "or pooling across width/height."));

  if (depth_window == 1) {  // we are pooling in the H and W
    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                tensor_in_rows, window_rows, row_stride,
                                padding, &out_height, &pad_top, &pad_bottom));

    OP_REQUIRES_OK(context, GetWindowedOutputSizeVerbose(
                                tensor_in_cols, window_cols, col_stride,
                                padding, &out_width, &pad_left, &pad_right));
#ifndef INTEL_MKL_ML
    // TF can work with int64, but mkldnn only supports int32
    // Fail if the height or width are greater than MAX_INT

    OP_REQUIRES(context,
                FastBoundsCheck(out_height, std::numeric_limits<int>::max()),
                errors::InvalidArgument("output height is too large"));

    OP_REQUIRES(context,
                FastBoundsCheck(out_width, std::numeric_limits<int>::max()),
                errors::InvalidArgument("output width is too large"));

#endif
    out_depth = depth;  // output will have the same depth as the input
  } else {              // we are pooling in the depth dimension
    // Our current version of depthwise max pooling does not support
    // any padding, and expects the depth_window to equal the depth
    // stride (no overlapping).
    OP_REQUIRES(context, depth % depth_window == 0,
                errors::Unimplemented("Depthwise max pooling requires the"
                                      " depth window to evenly divide the"
                                      " input depth"));
    OP_REQUIRES(context, depth_stride == depth_window,
                errors::Unimplemented("Depthwise max pooling requires the"
                                      " depth window to equal the depth"
                                      " stride"));

    // The current version of depthwise max is only implemented on CPU.
    OP_REQUIRES(context,
                (DeviceType(static_cast<Device*>(context->device())
                                ->attributes()
                                .device_type()) == DeviceType(DEVICE_CPU)),
                errors::Unimplemented("Depthwise max pooling is currently "
                                      "only implemented for CPU devices."));

    out_depth = depth / depth_window;
  }
}

// Transfers the right parameters for pooling to the op parameters
// Updates context->status if there is an invalid input.
void ExtractMklOpParams(OpKernelContext* context, TensorFormat data_format,
                        const MklPoolParameters& params,
                        MklPoolingOpParams* mkl_params) {
  mkl_params->in_sizes[0] = params.tensor_in_cols;
  mkl_params->in_sizes[1] = params.tensor_in_rows;
  mkl_params->in_sizes[2] = params.depth;
  mkl_params->in_sizes[3] = params.tensor_in_batch;

  GetStridesFromSizes(data_format, mkl_params->in_strides,
                      mkl_params->in_sizes);

  mkl_params->out_sizes[0] = params.out_width;
  mkl_params->out_sizes[1] = params.out_height;
  mkl_params->out_sizes[2] = params.depth;
  mkl_params->out_sizes[3] = params.tensor_in_batch;

  GetStridesFromSizes(data_format, mkl_params->out_strides,
                      mkl_params->out_sizes);

  mkl_params->in_offset[0] = -params.pad_left;
  mkl_params->in_offset[1] = -params.pad_top;
  mkl_params->in_offset[2] = -params.pad_right;
  mkl_params->in_offset[3] = -params.pad_bottom;

  mkl_params->kernel_stride[0] = params.col_stride;
  mkl_params->kernel_stride[1] = params.row_stride;

  mkl_params->kernel_size[0] = params.window_cols;
  mkl_params->kernel_size[1] = params.window_rows;
}
}  // namespace tensorflow
#endif  // INTEL_MKL
