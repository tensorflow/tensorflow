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

#ifndef TENSORFLOW_CORE_KERNELS_MKL_POOLING_OPS_COMMON_H_
#define TENSORFLOW_CORE_KERNELS_MKL_POOLING_OPS_COMMON_H_

#ifdef INTEL_MKL
#include <vector>
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/util/padding.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

struct MklPoolParameters {
  int depth;

  int tensor_in_cols;
  int tensor_in_rows;
  int tensor_in_batch;

  int window_rows;
  int window_cols;
  int depth_window;

  int row_stride;
  int col_stride;
  int depth_stride;

  int64 out_height;
  int64 out_width;
  int out_depth;

  int64 pad_left;
  int64 pad_right;
  int64 pad_top;
  int64 pad_bottom;
  int pad_depth;

  TensorFormat data_format;

  // Updates context->status if there is an invalid input.
  void Init(OpKernelContext* context, const std::vector<int32>& ksize,
            const std::vector<int32>& stride, Padding padding,
            TensorFormat data_format, const TensorShape& tensor_in_shape);
  void Init(OpKernelContext* context, const std::vector<int32>& ksize,
            const std::vector<int32>& stride, Padding padding,
            TensorFormat data_format, const MklShape* mkl_in_shape);

 private:
  // Common initialization for TensorFlow and MKL formats
  void Init(OpKernelContext* context, const std::vector<int32>& ksize,
            const std::vector<int32>& stride, Padding padding,
            TensorFormat data_format);
};

//-------------------------------------------------------------------
// Utility functions

typedef struct {
  size_t in_dim;
  size_t in_sizes[4];
  size_t in_strides[4];
  size_t out_sizes[4];
  size_t out_strides[4];
  int in_offset[4];
  size_t kernel_stride[2];
  size_t kernel_size[2];
} MklPoolingOpParams;

// Transfers the right parameters for pooling to the op parameters
// Updates context->status if there is an invalid input.
void ExtractMklOpParams(OpKernelContext* context, TensorFormat data_format,
                        const MklPoolParameters& params,
                        MklPoolingOpParams* mkl_params);
}  // namespace tensorflow

#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_KERNELS_MKL_POOLING_OPS_COMMON_H_
