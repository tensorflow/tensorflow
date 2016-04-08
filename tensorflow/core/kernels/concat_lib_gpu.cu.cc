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
#include <vector>

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/concat_lib.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
void ConcatGPU32(
    const GPUDevice& d,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs,
    typename TTypes<T, 2>::Matrix* output) {
  Eigen::array<int32, 2> offset{0, 0};
  for (int i = 0; i < inputs.size(); ++i) {
    Eigen::array<int32, 2> size;
    size[0] = inputs[i]->dimension(0);
    size[1] = inputs[i]->dimension(1);
    To32Bit(*output).slice(offset, size).device(d) = To32Bit(*inputs[i]);
    offset[1] += size[1];
  }
}

template <typename T>
void ConcatGPU64(
    const GPUDevice& d,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs,
    typename TTypes<T, 2>::Matrix* output) {
  Eigen::array<int64, 2> offset{0, 0};
  for (int i = 0; i < inputs.size(); ++i) {
    Eigen::array<int64, 2> size;
    size[0] = inputs[i]->dimension(0);
    size[1] = inputs[i]->dimension(1);
    output->slice(offset, size).device(d) = *inputs[i];
    offset[1] += size[1];
  }
}

#define REGISTER_GPU32(T)                                                     \
  template void ConcatGPU32<T>(                                               \
      const GPUDevice& d,                                                     \
      const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>& \
          inputs,                                                             \
      typename TTypes<T, 2>::Matrix* output);

#define REGISTER_GPU64(T)                                                     \
  template void ConcatGPU64<T>(                                               \
      const GPUDevice& d,                                                     \
      const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>& \
          inputs,                                                             \
      typename TTypes<T, 2>::Matrix* output);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU32);
REGISTER_GPU32(bfloat16);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU64);
REGISTER_GPU64(bfloat16);

#undef REGISTER_GPU32
#undef REGISTER_GPU64

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
