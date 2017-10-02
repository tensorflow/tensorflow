/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/contrib/rnn/kernels/lstm_ops.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// TODO(b/63339763): Provide an alternative implementation for
// LSTMBlockCell{F,B}prop that doesn't rely on Eigen.
#define DEFINE_GPU_SPECS(T)                                                    \
  template struct TensorZero<GPUDevice, T>;                                    \
  template struct TensorUnalignedZero<GPUDevice, T>;                           \
  template struct TensorCopy<GPUDevice, T>;                                    \
  template struct TensorCopyUnaligned<GPUDevice, T>;                           \
  template struct TensorCopyToUnaligned<GPUDevice, T>;                         \
  template struct TensorAdd<GPUDevice, T>;                                     \
  template <>                                                                  \
  void LSTMBlockCellFprop<GPUDevice, T, true /* USE_CUBLAS */>::operator()(    \
      OpKernelContext* ctx, const GPUDevice& d, const T forget_bias,           \
      const T cell_clip, bool use_peephole, typename TTypes<T>::ConstMatrix x, \
      typename TTypes<T>::ConstMatrix cs_prev,                                 \
      typename TTypes<T>::ConstMatrix h_prev,                                  \
      typename TTypes<T>::ConstMatrix w, typename TTypes<T>::ConstVec wci,     \
      typename TTypes<T>::ConstVec wcf, typename TTypes<T>::ConstVec wco,      \
      typename TTypes<T>::ConstVec b, typename TTypes<T>::Matrix xh,           \
      typename TTypes<T>::Matrix i, typename TTypes<T>::Matrix cs,             \
      typename TTypes<T>::Matrix f, typename TTypes<T>::Matrix o,              \
      typename TTypes<T>::Matrix ci, typename TTypes<T>::Matrix co,            \
      typename TTypes<T>::Matrix icfo, typename TTypes<T>::Matrix h) {         \
    LSTMBlockCellFpropWithEigen<GPUDevice, T, true /* USE_CUBLAS */>(          \
        *this, ctx, d, forget_bias, cell_clip, use_peephole, x, cs_prev,       \
        h_prev, w, wci, wcf, wco, b, xh, i, cs, f, o, ci, co, icfo, h);        \
  }                                                                            \
  template <>                                                                  \
  void LSTMBlockCellBprop<GPUDevice, T, true /* USE_CUBLAS */>::operator()(    \
      OpKernelContext* ctx, const GPUDevice& d, bool use_peephole,             \
      typename TTypes<T>::ConstMatrix x,                                       \
      typename TTypes<T>::ConstMatrix cs_prev,                                 \
      typename TTypes<T>::ConstMatrix h_prev,                                  \
      typename TTypes<T>::ConstMatrix w, typename TTypes<T>::ConstVec wci,     \
      typename TTypes<T>::ConstVec wcf, typename TTypes<T>::ConstVec wco,      \
      typename TTypes<T>::ConstVec b, typename TTypes<T>::ConstMatrix i,       \
      typename TTypes<T>::ConstMatrix cs, typename TTypes<T>::ConstMatrix f,   \
      typename TTypes<T>::ConstMatrix o, typename TTypes<T>::ConstMatrix ci,   \
      typename TTypes<T>::ConstMatrix co,                                      \
      typename TTypes<T>::ConstMatrix cs_grad,                                 \
      typename TTypes<T>::ConstMatrix h_grad, typename TTypes<T>::Matrix do_,  \
      typename TTypes<T>::Matrix dcs, typename TTypes<T>::Matrix dci,          \
      typename TTypes<T>::Matrix df, typename TTypes<T>::Matrix di,            \
      typename TTypes<T>::Matrix dicfo,                                        \
      typename TTypes<T>::Matrix cs_prev_grad,                                 \
      typename TTypes<T>::Vec wci_grad, typename TTypes<T>::Vec wcf_grad,      \
      typename TTypes<T>::Vec wco_grad) {                                      \
    LSTMBlockCellBpropWithEigen<GPUDevice, T, true /* USE_CUBLAS */>(          \
        *this, ctx, d, use_peephole, x, cs_prev, h_prev, w, wci, wcf, wco, b,  \
        i, cs, f, o, ci, co, cs_grad, h_grad, do_, dcs, dci, df, di, dicfo,    \
        cs_prev_grad, wci_grad, wcf_grad, wco_grad);                           \
  }                                                                            \
  template struct LSTMBlockCellFprop<GPUDevice, T, true /* USE_CUBLAS */>;     \
  template struct LSTMBlockCellBprop<GPUDevice, T, true /* USE_CUBLAS */>;     \
  template struct BlockLSTMBprop<GPUDevice, T, true /* USE_CUBLAS */>;

DEFINE_GPU_SPECS(float);
// DEFINE_GPU_SPECS(double);
#undef DEFINE_GPU_SPECS

}  // end namespace functor
}  // end namespace tensorflow
#endif  // GOOGLE_CUDA
