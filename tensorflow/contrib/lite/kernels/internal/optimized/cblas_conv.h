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

#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_CBLAS_CONV_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_CBLAS_CONV_H_

// The Conv implementation based on CBLAS interface. This is only used on iOS
// for now, utilizing Apple's Accelerate framework.

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include "tensorflow/contrib/lite/kernels/internal/optimized/cblas_reference.h"
#endif  // __APPLE__

#include "tensorflow/contrib/lite/kernels/internal/optimized/multithreaded_conv.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"

namespace tflite {
namespace cblas_ops {

inline void Conv(const float* input_data, const Dims<4>& input_dims,
                 const float* filter_data, const Dims<4>& filter_dims,
                 const float* bias_data, const Dims<4>& bias_dims,
                 int stride_width, int stride_height, int pad_width,
                 int pad_height, float output_activation_min,
                 float output_activation_max, float* output_data,
                 const Dims<4>& output_dims, float* im2col_data,
                 const Dims<4>& im2col_dims) {
  gemmlowp::ScopedProfilingLabel label("Conv/cblas");

  const float* gemm_input_data = nullptr;
  const Dims<4>* gemm_input_dims = nullptr;
  const int filter_width = ArraySize(filter_dims, 1);
  const int filter_height = ArraySize(filter_dims, 2);
  const bool need_im2col = stride_width != 1 || stride_height != 1 ||
                           filter_width != 1 || filter_height != 1;
  if (need_im2col) {
    TFLITE_DCHECK(im2col_data);
    optimized_ops::Im2col(input_data, input_dims, stride_width, stride_height,
                          pad_width, pad_height, filter_height, filter_width, 0,
                          im2col_data, im2col_dims);
    gemm_input_data = im2col_data;
    gemm_input_dims = &im2col_dims;
  } else {
    TFLITE_DCHECK(!im2col_data);
    gemm_input_data = input_data;
    gemm_input_dims = &input_dims;
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
  int m = gemm_input_dims->sizes[1] * gemm_input_dims->sizes[2] *
          gemm_input_dims->sizes[3];
  int n = output_dims.sizes[0];
  int k = gemm_input_dims->sizes[0];
  // The stride of matrix a, b and c respectively.
  int stride_a = k;
  int stride_b = k;
  int stride_c = n;

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, a,
              stride_a, b, stride_b, 0.0f, c, stride_c);

  optimized_ops::AddBiasAndEvalActivationFunction(
      bias_data, bias_dims, output_data, output_dims, output_activation_min,
      output_activation_max);
}

}  // namespace cblas_ops
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_CBLAS_CONV_H_
