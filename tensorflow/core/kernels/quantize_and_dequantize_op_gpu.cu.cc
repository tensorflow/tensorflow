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

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/quantize_and_dequantize_op.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {
template <typename T>
struct QuantizeAndDequantizeOneScaleFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::ConstVec input,
                  bool signed_input, int num_bits, bool range_given,
                  Tensor* input_min_tensor, Tensor* input_max_tensor,
                  QuantizerRoundMode round_mode, bool narrow_range,
                  typename TTypes<T>::Vec output) {
    QuantizeAndDequantizeOneScaleImpl<GPUDevice, T>::Compute(
        d, input, signed_input, num_bits, range_given, input_min_tensor,
        input_max_tensor, round_mode, narrow_range, output);
  }
};

template <typename T>
struct QuantizeAndDequantizePerChannelFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T, 3>::ConstTensor input,
                  bool signed_input, int num_bits, bool range_given,
                  Tensor* input_min_tensor, Tensor* input_max_tensor,
                  QuantizerRoundMode round_mode, bool narrow_range,
                  typename TTypes<T, 3>::Tensor output) {
    QuantizeAndDequantizePerChannelImpl<GPUDevice, T>::Compute(
        d, input, signed_input, num_bits, range_given, input_min_tensor,
        input_max_tensor, round_mode, narrow_range, output);
  }
};

template <typename T>
struct QuantizeAndDequantizeOneScaleGradientFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T>::ConstFlat gradient,
                  typename TTypes<T>::ConstFlat input,
                  typename TTypes<T>::ConstScalar input_min_tensor,
                  typename TTypes<T>::ConstScalar input_max_tensor,
                  typename TTypes<T>::Flat input_backprop,
                  typename TTypes<T>::Scalar input_min_backprop,
                  typename TTypes<T>::Scalar input_max_backprop) {
    QuantizeAndDequantizeOneScaleGradientImpl<GPUDevice, T>::Compute(
        d, gradient, input, input_min_tensor, input_max_tensor, input_backprop,
        input_min_backprop, input_max_backprop);
  }
};

template <typename T>
struct QuantizeAndDequantizePerChannelGradientFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d,
                  typename TTypes<T, 3>::ConstTensor gradient,
                  typename TTypes<T, 3>::ConstTensor input,
                  const Tensor* input_min_tensor,
                  const Tensor* input_max_tensor,
                  typename TTypes<T, 3>::Tensor input_backprop,
                  typename TTypes<T>::Flat input_min_backprop,
                  typename TTypes<T>::Flat input_max_backprop) {
    QuantizeAndDequantizePerChannelGradientImpl<GPUDevice, T>::Compute(
        d, gradient, input, input_min_tensor, input_max_tensor, input_backprop,
        input_min_backprop, input_max_backprop);
  }
};

}  // end namespace functor

// Instantiate the GPU implementation for float and double.
template struct functor::QuantizeAndDequantizeOneScaleFunctor<GPUDevice, float>;
template struct functor::QuantizeAndDequantizeOneScaleFunctor<GPUDevice,
                                                              double>;

template struct functor::QuantizeAndDequantizePerChannelFunctor<GPUDevice,
                                                                float>;
template struct functor::QuantizeAndDequantizePerChannelFunctor<GPUDevice,
                                                                double>;

template struct functor::QuantizeAndDequantizeOneScaleGradientFunctor<GPUDevice,
                                                                      float>;
template struct functor::QuantizeAndDequantizeOneScaleGradientFunctor<GPUDevice,
                                                                      double>;
template struct functor::QuantizeAndDequantizePerChannelGradientFunctor<
    GPUDevice, float>;
template struct functor::QuantizeAndDequantizePerChannelGradientFunctor<
    GPUDevice, double>;

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
