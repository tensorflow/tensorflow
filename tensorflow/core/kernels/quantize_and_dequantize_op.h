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

#ifndef TENSORFLOW_CORE_KERNELS_QUANTIZE_AND_DEQUANTIZE_OP_H_
#define TENSORFLOW_CORE_KERNELS_QUANTIZE_AND_DEQUANTIZE_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct QuantizeAndDequantizeOneScaleFunctor {
  void operator()(const Device& d, typename TTypes<T>::ConstVec input,
                  bool signed_input, int num_bits, bool range_given,
                  Tensor* input_min_tensor, Tensor* input_max_tensor,
                  typename TTypes<T>::Vec out);
};

// The implementation below runs on both CPU and GPU.
template <typename Device, typename T>
struct QuantizeAndDequantizeOneScaleImpl {
  static void Compute(const Device& d, typename TTypes<T>::ConstVec input,
                      bool signed_input, int num_bits, bool range_given,
                      Tensor* input_min_tensor, Tensor* input_max_tensor,
                      typename TTypes<T>::Vec out) {
    T min_range;
    T max_range;
    auto input_min = input_min_tensor->scalar<T>();
    auto input_max = input_max_tensor->scalar<T>();
    if (!range_given) {
      input_min.device(d) = input.minimum();
      input_max.device(d) = input.maximum();
    }
    d.memcpyDeviceToHost(&min_range, input_min.data(), sizeof(T));
    d.memcpyDeviceToHost(&max_range, input_max.data(), sizeof(T));

    // Calculate the range for the simulated integer quantization:
    // e.g. [-128,127] for signed = true, num_bits = 8,
    // or [0, 255] for signed = false, num_bits = 8.
    const int64 min_quantized = signed_input ? -(1ULL << (num_bits - 1)) : 0;
    const int64 max_quantized = min_quantized + ((1ULL << num_bits) - 1);

    // Determine the maximum scaling factor that would scale
    // [min_range, max_range] to not exceed [min_quantized, max_quantized],
    // while keeping 0 unchanged.
    const T scale_from_min_side = (min_quantized * min_range > 0)
                                      ? min_quantized / min_range
                                      : std::numeric_limits<T>::max();
    const T scale_from_max_side = (max_quantized * max_range > 0)
                                      ? max_quantized / max_range
                                      : std::numeric_limits<T>::max();
    auto scale = std::min(scale_from_min_side, scale_from_max_side);

    T inverse_scale = T(1.0) / scale;
    if (range_given) {
      out.device(d) =
          ((input.cwiseMin(max_range).cwiseMax(min_range) - min_range) * scale +
           T(0.5))
                  .floor() *
              inverse_scale +
          min_range;
    } else {
      // No need to compare with min and max as they are measured from the
      // tensor.
      out.device(d) =
          ((input - min_range) * scale + T(0.5)).floor() * inverse_scale +
          min_range;
    }
  }
};

}  // end of namespace functor
}  // end of namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_QUANTIZE_AND_DEQUANTIZE_OP_H_
