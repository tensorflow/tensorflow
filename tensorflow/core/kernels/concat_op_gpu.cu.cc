/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include <stdio.h>

#include <memory>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
void ConcatGPU(const GPUDevice& d,
               const std::vector<
                   std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>& inputs,
               typename TTypes<T, 2>::Matrix* output) {
  Eigen::array<int32, 2> offset{0, 0};
  for (int i = 0; i < inputs.size(); ++i) {
    Eigen::array<int32_t, 2> size;
    size[0] = inputs[i]->dimension(0);
    size[1] = inputs[i]->dimension(1);
    To32Bit(*output).slice(offset, size).device(d) = To32Bit(*inputs[i]);
    offset[1] += size[1];
  }
}

#define REGISTER_GPU(T)                                                       \
  template void ConcatGPU<T>(                                                 \
      const GPUDevice& d,                                                     \
      const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>& \
          inputs,                                                             \
      typename TTypes<T, 2>::Matrix* output);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);
REGISTER_GPU(bfloat16);
#undef REGISTER_GPU

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
