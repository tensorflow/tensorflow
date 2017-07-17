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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow{

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

template <typename T>
void RepeatGPUCall(const Eigen::GpuDevice& d,
                   const typename TTypes<T,3>::ConstTensor& input_flat,
                   const typename TTypes<int32>::ConstFlat& repeats_flat,
                   int axis, typename TTypes<T,3>::Tensor* output_flat);

template <typename T>
void RepeatGPUImpl(const Eigen::GpuDevice& d, const Tensor& input,
                   const typename TTypes<int32>::ConstFlat& repeats_flat,
                   int axis, Tensor* output) {
  auto input_flat = input.flat_inner_outer_dims<T>(axis);
  auto output_flat = output->flat_inner_outer_dims<T>(axis);
                                          
  RepeatGPUCall<T>(d, input_flat, repeats_flat, axis, &output_flat);
  
}

#define REGISTER(T)                                          \
  template void RepeatGPUImpl<T>(const Eigen::GpuDevice& d,  \
      const Tensor& input,                                   \
      const typename TTypes<int32>::ConstFlat& repeats_flat, \
      int axis, Tensor* output);
  
TF_CALL_GPU_NUMBER_TYPES(REGISTER);

#undef REGISTER

#endif // GOOGLE_CUDA

} // end namespace tensorflow
