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

// See docs in ../ops/array_ops.cc.

#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"

#if GOOGLE_CUDA

#include "tensorflow/core/kernels/cuda_device_array.h"

namespace tensorflow {

template <typename T, typename IntType>
void ConcatGPUSlice(
    const Eigen::GpuDevice& gpu_device,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs_flat,
    typename TTypes<T, 2>::Matrix* output);

template <typename T, typename IntType>
void ConcatGPUImpl(const Eigen::GpuDevice& d,
                   const CudaDeviceArrayStruct<const T*>& input_ptrs,
                   const CudaDeviceArrayStruct<IntType>& ptr_offsets,
                   bool same_size, int slice_size,
                   typename TTypes<T, 2>::Matrix* output);

namespace {

template <typename T, typename IntType>
void ConcatGPUCall(
    OpKernelContext* c,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs_flat,
    typename TTypes<T, 2>::Tensor* output_flat) {
  CudaDeviceArrayOnHost<const T*> input_ptrs(c, inputs_flat.size());
  OP_REQUIRES_OK(c, input_ptrs.Init());
  for (int i = 0; i < inputs_flat.size(); ++i) {
    input_ptrs.Set(i, inputs_flat[i]->data());
  }
  OP_REQUIRES_OK(c, input_ptrs.Finalize());

  CudaDeviceArrayOnHost<IntType> output_scan(c, inputs_flat.size() + 1);
  OP_REQUIRES_OK(c, output_scan.Init());
  IntType scan = 0;
  output_scan.Set(0, scan);
  bool one_size_input = true;
  for (int i = 0; i < inputs_flat.size(); ++i) {
    if (one_size_input && i < inputs_flat.size() - 1 &&
        inputs_flat[i]->dimension(1) != inputs_flat[i + 1]->dimension(1)) {
      one_size_input = false;
    }
    scan += inputs_flat[i]->dimension(1);
    output_scan.Set(i + 1, scan);
  }
  if (!one_size_input) OP_REQUIRES_OK(c, output_scan.Finalize());

  ConcatGPUImpl<T, IntType>(c->eigen_gpu_device(), input_ptrs.data(),
                            output_scan.data(), one_size_input,
                            inputs_flat[0]->dimension(1), output_flat);
}

}  // end namespace

template <typename T>
void ConcatGPU(
    OpKernelContext* c,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs_flat,
    Tensor* output, typename TTypes<T, 2>::Tensor* output_flat) {
  if (inputs_flat.size() < 16) {
    if (output->NumElements() < std::numeric_limits<int32>::max()) {
      ConcatGPUSlice<T, int32>(c->eigen_gpu_device(), inputs_flat, output_flat);
    } else {
      ConcatGPUSlice<T, int64>(c->eigen_gpu_device(), inputs_flat, output_flat);
    }
  } else {
    // Switching indexing to int64 might cause performance issues.
    // Hence, we keep int32 indexing in the GPU kernel unless we need to
    // switch to int64.
    if (output->NumElements() < std::numeric_limits<int32>::max()) {
      ConcatGPUCall<T, int32>(c, inputs_flat, output_flat);
    } else {
      ConcatGPUCall<T, int64>(c, inputs_flat, output_flat);
    }
  }
}

#define REGISTER(T)                                                           \
  template void ConcatGPU<T>(                                                 \
      OpKernelContext * c,                                                    \
      const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>& \
          inputs_flat,                                                        \
      Tensor* output, typename TTypes<T, 2>::Tensor* output_flat);

TF_CALL_GPU_NUMBER_TYPES(REGISTER);
TF_CALL_complex64(REGISTER);
TF_CALL_complex128(REGISTER);
TF_CALL_int64(REGISTER);
TF_CALL_bfloat16(REGISTER);
TF_CALL_bool(REGISTER);
TF_CALL_uint8(REGISTER);

#undef REGISTER

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
