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

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

#define FAKE_QUANT_NO_DEBUG

#define EIGEN_USE_GPU
#include "tensorflow/core/kernels/fake_quant_ops_functor.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// Just instantiate GPU functor implementations.
template struct FakeQuantWithMinMaxArgsFunctor<GPUDevice>;
template struct FakeQuantWithMinMaxArgsGradientFunctor<GPUDevice>;
template struct FakeQuantWithMinMaxVarsFunctor<GPUDevice>;
template struct FakeQuantWithMinMaxVarsGradientFunctor<GPUDevice>;
template struct FakeQuantWithMinMaxVarsPerChannelFunctor<GPUDevice>;
template struct FakeQuantWithMinMaxVarsPerChannelGradientFunctor<GPUDevice>;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
