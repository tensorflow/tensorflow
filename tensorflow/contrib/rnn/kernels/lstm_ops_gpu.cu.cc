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

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

#define DEFINE_GPU_SPECS(T)                               \
  template struct TensorZero<GPUDevice, T>;               \
  template struct TensorUnalignedZero<GPUDevice, T>;      \
  template struct TensorCopy<GPUDevice, T>;               \
  template struct TensorCopyUnaligned<GPUDevice, T>;      \
  template struct TensorCopyToUnaligned<GPUDevice, T>;    \
  template struct TensorAdd<GPUDevice, T>;                \
  template struct LSTMBlockCellFprop<GPUDevice, T, true>; \
  template struct LSTMBlockCellBprop<GPUDevice, T, true>; \
  template struct BlockLSTMBprop<GPUDevice, T, true>;

DEFINE_GPU_SPECS(float);
// DEFINE_GPU_SPECS(double);
#undef DEFINE_GPU_SPECS

}  // end namespace functor
}  // end namespace tensorflow
#endif  // GOOGLE_CUDA
