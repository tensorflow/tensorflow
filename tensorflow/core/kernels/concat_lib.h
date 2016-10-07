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

#ifndef TENSORFLOW_KERNELS_CONCAT_LIB_H_
#define TENSORFLOW_KERNELS_CONCAT_LIB_H_

#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/device_base.h"

namespace tensorflow {

// Assumes all inputs are nonempty
template <typename T>
void ConcatCPU(DeviceBase* d,
               const std::vector<
                   std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>& inputs,
               typename TTypes<T, 2>::Matrix* output);
#if GOOGLE_CUDA
template <typename T>
void ConcatGPU(
    OpKernelContext* c,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs_flat,
    Tensor* output, typename TTypes<T, 2>::Tensor* output_flat);

#endif  // GOOGLE_CUDA
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_CONCAT_LIB_H_
