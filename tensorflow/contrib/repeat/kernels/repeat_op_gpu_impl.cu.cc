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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow{

template <typename T>
void RepeatGPUCall(const Eigen::GpuDevice& d,
                   const typename TTypes<T, 3>::ConstTensor& input_flat,
                   const typename TTypes<int32>::ConstFlat& repeats_flat,
                   int axis, typename TTypes<T, 3>::Tensor* output_flat) {  
  if (repeats_flat.size() == 1) {
    int32 repeat = repeats_flat(0);
    Eigen::array<int64, 3> input_offsets = {0, 0, 0};
    Eigen::array<int64, 3> output_offsets = {0, 0, 0};
    Eigen::array<int64, 3> input_extents = {
        input_flat.dimension(0), 1, input_flat.dimension(2)};
    Eigen::array<int64, 3> output_extents = {
        input_flat.dimension(0), repeat, input_flat.dimension(2)};
    Eigen::array<int64, 3> broadcast_array = {1, repeat, 1};
    for (int64 i = 0; i < input_flat.dimension(1); ++i) {   
      output_flat->slice(output_offsets, output_extents).device(d) = 
          input_flat.slice(input_offsets, input_extents).broadcast(broadcast_array);
      output_offsets[1] += repeat;
      input_offsets[1]++;
    }
  } else {
    int64 input_offset = 0;
    int64 output_offset = 0;  
    for (int64 i = 0; i < repeats_flat.size(); i++) {
      for (int64 j = 0; j < repeats_flat(i); j++) {
        output_flat->chip(output_offset, 1).device(d) = 
            input_flat.chip(input_offset, 1);
        output_offset++;
      }
      input_offset++;
    }
  } 
}

#define REGISTER(T)                                          \
  template void RepeatGPUCall<T>(const Eigen::GpuDevice& d,  \
      const typename TTypes<T, 3>::ConstTensor& input_flat,  \
      const typename TTypes<int32>::ConstFlat& repeats_flat, \
      int axis, typename TTypes<T, 3>::Tensor* output_flat);
  
TF_CALL_GPU_NUMBER_TYPES(REGISTER);

#undef REGISTER

} // end namespace tensorflow

#endif // GOOGLE_CUDA
