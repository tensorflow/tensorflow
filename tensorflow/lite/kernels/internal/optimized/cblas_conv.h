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

#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_CBLAS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_CBLAS_CONV_H_

// The Conv implementation based on CBLAS interface. This is only used on iOS
// for now, utilizing Apple's Accelerate framework.

#if TFLITE_USE_APPLE_ACCELERATE_FOR_CONV
#include <Accelerate/Accelerate.h>
#else
#include "tensorflow/lite/kernels/internal/optimized/cblas_reference.h"
#endif

#include "tensorflow/lite/kernels/internal/optimized/multithreaded_conv.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"

namespace tflite {
namespace cblas_ops {

inline void Conv(const ConvParams& params, const RuntimeShape& input_shape,
                 const float* input_data, const RuntimeShape& filter_shape,
                 const float* filter_data, const RuntimeShape& bias_shape,
                 const float* bias_data, const RuntimeShape& output_shape,
                 float* output_data, const RuntimeShape& im2col_shape,
                 float* im2col_data) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const float output_activation_min = params.float_activation_min;
  const float output_activation_max = params.float_activation_max;
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  gemmlowp::ScopedProfilingLabel label("Conv/cblas");

  const float* gemm_input_data = nullptr;
  const RuntimeShape* gemm_input_shape = nullptr;
  const int filter_width = filter_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const bool need_im2col = stride_width != 1 || stride_height != 1 ||
                           filter_width != 1 || filter_height != 1;
  if (need_im2col) {
    TFLITE_DCHECK(im2col_data);
    ConvParams op_params;
    op_params.padding_type = PaddingType::kSame;
    op_params.padding_values.width = pad_width;
    op_params.padding_values.height = pad_height;
    op_params.stride_width = stride_width;
    op_params.stride_height = stride_height;
    op_params.dilation_width_factor = dilation_width_factor;
    op_params.dilation_height_factor = dilation_height_factor;
    optimized_ops::Im2col(op_params, filter_height, filter_width, 0,
                          input_shape, input_data, im2col_shape, im2col_data);

    gemm_input_data = im2col_data;
    gemm_input_shape = &im2col_shape;
  } else {
    TFLITE_DCHECK(!im2col_data);
    gemm_input_data = input_data;
    gemm_input_shape = &input_shape;
  }

  // The following code computes matrix multiplication c = a * transponse(b)
  // with CBLAS, where:
  // * `a` is a matrix with dimensions (m, k).
  // * `b` is a matrix with dimensions (n, k), so transpose(b) is (k, n).
  // * `c` is a matrix with dimensions (m, n).
  // The naming of variables are aligned with CBLAS specification here.
  const float* a = gemm_input_data;
  const float* b = filter_data;
  float* c = output_data;
  const int gemm_input_dims = gemm_input_shape->DimensionsCount();
  int m = FlatSizeSkipDim(*gemm_input_shape, gemm_input_dims - 1);
  int n = output_shape.Dims(3);
  int k = gemm_input_shape->Dims(gemm_input_dims - 1);
  // The stride of matrix a, b and c respectively.
  int stride_a = k;
  int stride_b = k;
  int stride_c = n;

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, a,
              stride_a, b, stride_b, 0.0f, c, stride_c);

  optimized_ops::AddBiasAndEvalActivationFunction(
      output_activation_min, output_activation_max, bias_shape, bias_data,
      output_shape, output_data);
}

}  // namespace cblas_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_CBLAS_CONV_H_
