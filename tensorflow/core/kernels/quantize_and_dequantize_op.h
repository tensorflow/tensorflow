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
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

enum QuantizerRoundMode {
  // Round half up: if the fraction of y is exactly 0.5, then
  // round(y) = y + 0.5
  // E.g., -5.5 gets rounded to -5, -5.4 goes to -5,
  // 5.4 goes to 5, and 5.5 goes to 6.
  ROUND_HALF_UP,
  // Round half to even: if the fraction of y is exactly 0.5, then round(y) is
  // the nearest even integer to y.
  // E.g., 23.5 gets rounded to 24, 24.5 gets rounded to 24, while -23.5 becomes
  // -24, and -24.5 gets rounded to 24.
  ROUND_HALF_TO_EVEN,
};

namespace functor {

// TODO(pauldonnelly): 'signed_input' should really be called 'signed_output'.

template <typename Device, typename T>
struct QuantizeAndDequantizeOneScaleFunctor {
  void operator()(const Device& d, typename TTypes<T>::ConstVec input,
                  bool signed_input, int num_bits, bool range_given,
                  Tensor* input_min_tensor, Tensor* input_max_tensor,
                  QuantizerRoundMode round_mode, bool narrow_range,
                  typename TTypes<T>::Vec output);
};

template <typename Device, typename T>
struct QuantizeAndDequantizePerChannelFunctor {
  void operator()(const Device& d, typename TTypes<T, 3>::ConstTensor input,
                  bool signed_input, int num_bits, bool range_given,
                  Tensor* input_min_tensor, Tensor* input_max_tensor,
                  QuantizerRoundMode round_mode, bool narrow_range,
                  typename TTypes<T, 3>::Tensor output);
};

template <typename Device, typename T>
struct QuantizeAndDequantizeOneScaleGradientFunctor {
  void operator()(const Device& d, typename TTypes<T>::ConstFlat gradient,
                  typename TTypes<T>::ConstFlat input,
                  typename TTypes<T>::ConstScalar input_min,
                  typename TTypes<T>::ConstScalar input_max,
                  typename TTypes<T>::Flat input_backprop,
                  typename TTypes<T>::Scalar input_min_backprop,
                  typename TTypes<T>::Scalar input_max_backprop);
};

template <typename Device, typename T>
struct QuantizeAndDequantizePerChannelGradientFunctor {
  void operator()(const Device& d, typename TTypes<T, 3>::ConstTensor gradient,
                  typename TTypes<T, 3>::ConstTensor input,
                  const Tensor* input_min_tensor,
                  const Tensor* input_max_tensor,
                  typename TTypes<T, 3>::Tensor input_backprop,
                  typename TTypes<T>::Flat input_min_backprop,
                  typename TTypes<T>::Flat input_max_backprop);
};

// The implementation below runs on both CPU and GPU.
template <typename Device, typename T, typename Func,
          typename Vec = typename TTypes<T>::Vec,
          typename ConstVec = typename TTypes<T>::ConstVec>
void ClampScaleAndRound(const Device& d, ConstVec input, T min_range,
                        T max_range, T scale, T inverse_scale, Func round_func,
                        Vec output) {
  output.device(d) = (input.cwiseMin(max_range).cwiseMax(min_range) * scale)
                         .unaryExpr(round_func) *
                     inverse_scale;
}

// The implementation below runs on both CPU and GPU.
template <typename Device, typename T, typename Vec = typename TTypes<T>::Vec,
          typename ConstVec = typename TTypes<T>::ConstVec>
void ClampScaleAndRound(const Device& d, ConstVec input, T min_range,
                        T max_range, T scale, T inverse_scale,
                        QuantizerRoundMode round_mode, Vec output) {
  switch (round_mode) {
    case ROUND_HALF_TO_EVEN:
      ClampScaleAndRound(d, input, min_range, max_range, scale, inverse_scale,
                         Eigen::internal::scalar_round_half_to_even_op<T>(),
                         output);
      break;
    case ROUND_HALF_UP:
      ClampScaleAndRound(d, input, min_range, max_range, scale, inverse_scale,
                         Eigen::internal::scalar_round_up_op<T>(), output);
      break;
  }
}

// The implementation below runs on both CPU and GPU.
template <typename Device, typename T, typename Func,
          typename Vec = typename TTypes<T>::Vec,
          typename ConstVec = typename TTypes<T>::ConstVec>
void ScaleAndRound(const Device& d, ConstVec input, T scale, T inverse_scale,
                   Func round_func, Vec output) {
  output.device(d) = (input * scale).unaryExpr(round_func) * inverse_scale;
}

// The implementation below runs on both CPU and GPU.
template <typename Device, typename T, typename Vec = typename TTypes<T>::Vec,
          typename ConstVec = typename TTypes<T>::ConstVec>
void ScaleAndRound(const Device& d, ConstVec input, T scale, T inverse_scale,
                   QuantizerRoundMode round_mode, Vec output) {
  switch (round_mode) {
    case ROUND_HALF_TO_EVEN:
      ScaleAndRound(d, input, scale, inverse_scale,
                    Eigen::internal::scalar_round_half_to_even_op<T>(), output);
      break;
    case ROUND_HALF_UP:
      ScaleAndRound(d, input, scale, inverse_scale,
                    Eigen::internal::scalar_round_up_op<T>(), output);
      break;
  }
}

template <typename T>
void ComputeQuantizationRange(bool signed_input, int num_bits,
                              QuantizerRoundMode round_mode, bool narrow_range,
                              T* min_range, T* max_range, T* scale,
                              T* inverse_scale) {
  // Calculate the range for the simulated integer quantization:
  // e.g. [-127,127] for signed = true, narrow_range = true, num_bits = 8,
  // or [-128,127] for signed = true, narrow_range = false, num_bits = 8,
  // or [0, 255] for signed = false, num_bits = 8.
  const int64 min_quantized = signed_input ? narrow_range
                                                 ? -(1ULL << (num_bits - 1)) + 1
                                                 : -(1ULL << (num_bits - 1))
                                           : 0;
  const int64 max_quantized =
      signed_input ? (1ULL << (num_bits - 1)) - 1 : (1ULL << num_bits) - 1;
  // Determine the maximum scaling factor that would scale
  // [min_range, max_range] to not exceed [min_quantized, max_quantized],
  // while keeping 0 unchanged.
  const T scale_from_min_side = (min_quantized * *min_range > 0)
                                    ? min_quantized / *min_range
                                    : std::numeric_limits<T>::max();
  const T scale_from_max_side = (max_quantized * *max_range > 0)
                                    ? max_quantized / *max_range
                                    : std::numeric_limits<T>::max();

  // Note: Avoids changing the side of the range that determines scale.
  if (scale_from_min_side < scale_from_max_side) {
    *scale = scale_from_min_side;
    *inverse_scale = *min_range / min_quantized;
    *max_range = max_quantized * *inverse_scale;
  } else {
    *scale = scale_from_max_side;
    *inverse_scale = *max_range / max_quantized;
    *min_range = min_quantized * *inverse_scale;
  }
}

// The implementation below runs on both CPU and GPU.
template <typename Device, typename T>
struct QuantizeAndDequantizeOneScaleImpl {
  static void Compute(const Device& d, typename TTypes<T>::ConstVec input,
                      bool signed_input, int num_bits, bool range_given,
                      Tensor* input_min_tensor, Tensor* input_max_tensor,
                      QuantizerRoundMode round_mode, bool narrow_range,
                      typename TTypes<T>::Vec output) {
    T min_range;
    T max_range;
    auto input_min = input_min_tensor->scalar<T>();
    auto input_max = input_max_tensor->scalar<T>();
    if (!range_given) {
      input_min.device(d) = input.minimum();
      input_max.device(d) = input.maximum();
      d.memcpyDeviceToHost(&min_range, input_min.data(), sizeof(T));
      d.memcpyDeviceToHost(&max_range, input_max.data(), sizeof(T));
    } else {
      // Copy the range values from their respective tensors on the host.
      min_range = input_min_tensor->scalar<T>()();
      max_range = input_max_tensor->scalar<T>()();
    }

    T scale, inverse_scale;
    ComputeQuantizationRange(signed_input, num_bits, round_mode, narrow_range,
                             &min_range, &max_range, &scale, &inverse_scale);

    if (range_given) {
      // Note: The clamping here is to avoid overflow in the quantized type.
      // The semantics of the op does not guarantee to clamp to the specified
      // min_range and max_range - because we may have changed either min_range
      // or max_range.
      ClampScaleAndRound(d, input, min_range, max_range, scale, inverse_scale,
                         round_mode, output);
    } else {
      ScaleAndRound(d, input, scale, inverse_scale, round_mode, output);
    }
  }
};

// The implementation below runs on both CPU and GPU.

template <typename Device, typename T>
struct QuantizeAndDequantizePerChannelImpl {
  static void Compute(const Device& d, typename TTypes<T, 3>::ConstTensor input,
                      bool signed_input, int num_bits, bool range_given,
                      Tensor* input_min_tensor, Tensor* input_max_tensor,
                      QuantizerRoundMode round_mode, bool narrow_range,
                      typename TTypes<T, 3>::Tensor output) {
    using Index = typename tensorflow::TTypes<T>::ConstTensor::Index;
    int num_channels = input.dimension(1);
    auto input_min = input_min_tensor->vec<T>();
    auto input_max = input_max_tensor->vec<T>();
    std::vector<T> min_range(num_channels);
    std::vector<T> max_range(num_channels);

    if (!range_given) {
#if !defined(EIGEN_HAS_INDEX_LIST)
      Eigen::array<int, 2> reduce_dims{{0, 2}};
#else
      Eigen::IndexList<Eigen::type2index<0>, Eigen::type2index<2> > reduce_dims;
#endif
      input_min.device(d) = input.minimum(reduce_dims);
      input_max.device(d) = input.maximum(reduce_dims);
      d.memcpyDeviceToHost(min_range.data(), input_min.data(),
                           num_channels * sizeof(T));
      d.memcpyDeviceToHost(max_range.data(), input_max.data(),
                           num_channels * sizeof(T));
    } else {
      // Copy the range values from their respective tensors on the host.
      std::memcpy(min_range.data(), input_min_tensor->vec<T>().data(),
                  num_channels * sizeof(T));
      std::memcpy(max_range.data(), input_max_tensor->vec<T>().data(),
                  num_channels * sizeof(T));
    }

    for (Index i = 0; i < num_channels; ++i) {
      const auto input_chip = input.template chip<1>(i);
      auto output_chip = output.template chip<1>(i);

      T scale, inverse_scale;
      ComputeQuantizationRange(signed_input, num_bits, round_mode, narrow_range,
                               &min_range[i], &max_range[i], &scale,
                               &inverse_scale);
      if (range_given) {
        ClampScaleAndRound(d, input_chip, min_range[i], max_range[i], scale,
                           inverse_scale, round_mode, output_chip);
      } else {
        ScaleAndRound(d, input_chip, scale, inverse_scale, round_mode,
                      output_chip);
      }
    }
  }
};

template <typename Device, typename T>
struct QuantizeAndDequantizeOneScaleGradientImpl {
  static void Compute(const Device& d, typename TTypes<T>::ConstFlat gradient,
                      typename TTypes<T>::ConstFlat input,
                      typename TTypes<T>::ConstScalar input_min,
                      typename TTypes<T>::ConstScalar input_max,
                      typename TTypes<T>::Flat input_backprop,
                      typename TTypes<T>::Scalar input_min_backprop,
                      typename TTypes<T>::Scalar input_max_backprop) {
    const T min_val = input_min();
    const T max_val = input_max();
    const auto in_range =
        (input >= min_val && input <= max_val)
            .select(input.constant(1.0f), input.constant(0.0f));
    input_backprop.device(d) = gradient * in_range;
    input_min_backprop.device(d) = input_min_backprop.constant(0.0f);
    input_max_backprop.device(d) = input_max_backprop.constant(0.0f);
  }
};

template <typename Device, typename T>
struct QuantizeAndDequantizePerChannelGradientImpl {
  static void Compute(const Device& d,
                      typename TTypes<T, 3>::ConstTensor gradient,
                      typename TTypes<T, 3>::ConstTensor input,
                      const Tensor* input_min_tensor,
                      const Tensor* input_max_tensor,
                      typename TTypes<T, 3>::Tensor input_backprop,
                      typename TTypes<T>::Flat input_min_backprop,
                      typename TTypes<T>::Flat input_max_backprop) {
    using Index = typename tensorflow::TTypes<T>::ConstTensor::Index;
    auto input_min = input_min_tensor->vec<T>();
    auto input_max = input_max_tensor->vec<T>();
    int num_channels = input.dimension(1);
    for (Index i = 0; i < num_channels; ++i) {
      const auto gradient_chip = gradient.template chip<1>(i);
      const auto input_chip = input.template chip<1>(i);
      const T min_val = input_min(i);
      const T max_val = input_max(i);
      const auto in_range =
          (input_chip >= min_val && input_chip <= max_val)
              .select(input_chip.constant(1.0f), input_chip.constant(0.0f));
      input_backprop.template chip<1>(i).device(d) = gradient_chip * in_range;
    }
    input_min_backprop.device(d) = input_min_backprop.constant(0.0f);
    input_max_backprop.device(d) = input_max_backprop.constant(0.0f);
  }
};

}  // end of namespace functor
}  // end of namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_QUANTIZE_AND_DEQUANTIZE_OP_H_
