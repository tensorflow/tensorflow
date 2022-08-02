/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/uniform_quant_ops/math_utils.h"

#include <algorithm>
#include <cmath>

namespace tensorflow {

void AsymmetricQuantize(const Tensor& tensor, int apply_offset, int apply_size,
                        int32_t quantization_min_val,
                        int32_t quantization_max_val, float& scale,
                        int32& zero_point, Tensor& quantized_tensor) {
  Eigen::DSizes<Eigen::Index, 1> apply_offset_array{apply_offset};
  Eigen::DSizes<Eigen::Index, 1> apply_size_array{apply_size};

  auto tensor_slice =
      tensor.flat<float>().slice(apply_offset_array, apply_size_array);
  auto quantized_tensor_slice = quantized_tensor.flat<qint8>().slice(
      apply_offset_array, apply_size_array);

  Eigen::Tensor<float, 0, Eigen::RowMajor> tensor_slice_min =
      tensor_slice.minimum();
  Eigen::Tensor<float, 0, Eigen::RowMajor> tensor_slice_max =
      tensor_slice.maximum();
  const double rmin = static_cast<double>(std::min(0.0f, tensor_slice_min()));
  const double rmax = static_cast<double>(std::max(0.0f, tensor_slice_max()));
  const double qmin_double = quantization_min_val;
  const double qmax_double = quantization_max_val;

  float inv_scale = 0;
  scale = (rmax - rmin) / (qmax_double - qmin_double);
  if (rmax - rmin != 0) {
    // Re-calculate the inverse instead of using (1./scale), to avoid loss of
    // precision.
    inv_scale = (qmax_double - qmin_double) / (rmax - rmin);
  }
  if (scale == 0 || !std::isfinite(inv_scale)) {
    quantized_tensor_slice.setZero();
    scale = 1.0;
    zero_point = 0;
    return;
  }

  const double zero_point_from_min = qmin_double - rmin / scale;
  const double zero_point_from_max = qmax_double - rmax / scale;
  const double zero_point_from_min_error =
      std::abs(qmin_double) + std::abs(rmin / scale);
  const double zero_point_from_max_error =
      std::abs(qmax_double) + std::abs(rmax / scale);
  const double zero_point_double =
      zero_point_from_min_error < zero_point_from_max_error
          ? zero_point_from_min
          : zero_point_from_max;

  int8_t nudged_zero_point = 0;
  if (zero_point_double <= qmin_double) {
    nudged_zero_point = quantization_min_val;
  } else if (zero_point_double >= qmax_double) {
    nudged_zero_point = quantization_max_val;
  } else {
    nudged_zero_point = static_cast<int8_t>(round(zero_point_double));
  }
  zero_point = nudged_zero_point;

  AffineQuantize(tensor_slice, inv_scale, zero_point, quantization_min_val,
                 quantization_max_val, quantized_tensor_slice);
}

}  // namespace tensorflow
