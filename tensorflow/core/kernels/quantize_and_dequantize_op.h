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

#ifndef LEARNING_BRAIN_GOOGLE_KERNELS_QUANTIZE_TRAINING_OP_H_
#define LEARNING_BRAIN_GOOGLE_KERNELS_QUANTIZE_TRAINING_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct QuantizeAndDequantizeOneScaleFunctor {
  void operator()(const Device& d, typename TTypes<T>::ConstVec input,
                  bool signed_input, int num_bits, bool range_given,
                  typename TTypes<T>::Scalar input_min,
                  typename TTypes<T>::Scalar input_max,
                  typename TTypes<T>::Vec out);
};

// The implementation below runs on both CPU and GPU.
template <typename Device, typename T>
struct QuantizeAndDequantizeOneScaleImpl {
  static void Compute(const Device& d, typename TTypes<T>::ConstVec input,
                      bool signed_input, int num_bits, bool range_given,
                      typename TTypes<T>::Scalar input_min,
                      typename TTypes<T>::Scalar input_max,
                      typename TTypes<T>::Vec out) {
    if (!range_given) {
      input_min.device(d) = input.minimum();
      input_max.device(d) = input.maximum();
    }

    T min_range = input_min();
    T max_range = input_max();

    // Make sure the range is symmetric for signed quantization, or start from
    // 0 for unsigned quantization.
    max_range = std::max(std::abs(max_range), std::abs(min_range));

    // If both min and max are 0, then the output should be just 0.
    if (max_range == 0) {
      out.device(d) = input * T(0);
      return;
    }

    if (signed_input) {
      min_range = -max_range;

      // If it is signed, we try to keep 0.0 being 0 and drop one bucket. For
      // example, if it is 8 bits, we have the range [-127, 127]. So for input
      // range of [-x, x], the scale should be 254/(2*x).
      T scale = static_cast<T>((uint64_t{1} << (num_bits - 1)) - 1) / max_range;
      T inverse_scale = T(1.0) / scale;
      if (range_given) {
        out.device(d) =
            ((input.cwiseMin(max_range).cwiseMax(min_range) - min_range) *
                 scale +
             T(0.5)).floor() *
                inverse_scale +
            min_range;
      } else {
        // No need to compare with min and max as they are measured from the
        // tensor.
        out.device(d) =
            ((input - min_range) * scale + T(0.5)).floor() * inverse_scale +
            min_range;
      }
    } else {
      min_range = 0;
      // If it is unsigned and num_bits == 8, the range with 8 bits is [0, 255].
      // If the input range is [0, x], then the scale is x/255 instead of 254 as
      // in the case above.
      T scale = static_cast<T>((uint64_t{1} << num_bits) - 1) / max_range;
      T inverse_scale = 1.0 / scale;
      if (range_given) {
        out.device(d) =
            ((input.cwiseMin(max_range).cwiseMax(min_range)) * scale + T(0.5))
                .floor() *
            inverse_scale;
      } else {
        // No need to compare with min and max as they are measured from the
        // tensor.
        out.device(d) = (input * scale + T(0.5)).floor() * inverse_scale;
      }
    }
  }
};

}  // end of namespace functor
}  // end of namespace tensorflow

#endif  // LEARNING_BRAIN_GOOGLE_KERNELS_QUANTIZE_TRAINING_OP_H_
