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

#ifndef TENSORFLOW_CORE_KERNELS_MKL_QUANTIZED_CONV_OPS_H_
#define TENSORFLOW_CORE_KERNELS_MKL_QUANTIZED_CONV_OPS_H_

#include "tensorflow/core/framework/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#ifdef INTEL_MKL

namespace tensorflow {
template <class T>
float MklFloatForOneQuantizedLevel(float range_min, float range_max) {
  int64 highest = static_cast<int64>(Eigen::NumTraits<T>::highest());
  int64 lowest = static_cast<int64>(Eigen::NumTraits<T>::lowest());

  // Adjusting for having a symmetric range.
  // for example: for 8-bit [-127, 127] as opposed to [-128, 127].
  if (lowest < -highest) ++lowest;

  const float float_for_one_quantized_level =
      (range_max - range_min) / (highest - lowest);
  return float_for_one_quantized_level;
}

template <class T1, class T2, class T3>
void MklQuantizationRangeForMultiplication(float min_a, float max_a,
                                           float min_b, float max_b,
                                           float* min_c, float* max_c) {
  const float a_float_for_one_quant_level =
      MklFloatForOneQuantizedLevel<T1>(min_a, max_a);
  const float b_float_for_one_quant_level =
      MklFloatForOneQuantizedLevel<T2>(min_b, max_b);

  const int64 c_highest = static_cast<int64>(Eigen::NumTraits<T3>::highest());
  const int64 c_lowest = static_cast<int64>(Eigen::NumTraits<T3>::lowest());
  const float c_float_for_one_quant_level =
      a_float_for_one_quant_level * b_float_for_one_quant_level;

  *min_c = c_float_for_one_quant_level * c_lowest;
  *max_c = c_float_for_one_quant_level * c_highest;
}

template <class T1, class T2, class T3>
void MklQuantizationRangeForMultiplication(float min_a, float max_a,
                                           const Tensor& min_b_vector,
                                           const Tensor& max_b_vector,
                                           Tensor** min_c_vector,
                                           Tensor** max_c_vector) {
  CHECK(min_b_vector.NumElements() == (*min_c_vector)->NumElements());
  CHECK(max_b_vector.NumElements() == (*max_c_vector)->NumElements());

  const int64 c_highest = static_cast<int64>(Eigen::NumTraits<T3>::highest());
  const int64 c_lowest = static_cast<int64>(Eigen::NumTraits<T3>::lowest());
  const float* min_b = min_b_vector.flat<float>().data();
  const float* max_b = max_b_vector.flat<float>().data();
  float* min_c = (*min_c_vector)->flat<float>().data();
  float* max_c = (*max_c_vector)->flat<float>().data();
#pragma omp parallel for
  for (size_t n = 0; n < min_b_vector.NumElements(); n++) {
    float a_float_for_one_quant_level =
        MklFloatForOneQuantizedLevel<T1>(min_a, max_a);
    float b_float_for_one_quant_level =
        MklFloatForOneQuantizedLevel<T2>(min_b[n], max_b[n]);
    float c_float_for_one_quant_level =
        a_float_for_one_quant_level * b_float_for_one_quant_level;
    min_c[n] = c_float_for_one_quant_level * c_lowest;
    max_c[n] = c_float_for_one_quant_level * c_highest;
  }
}

}  // namespace tensorflow

#endif  // INTEL_MKL

#endif  // TENSORFLOW_CORE_KERNELS_MKL_QUANTIZED_CONV_OPS_H_
