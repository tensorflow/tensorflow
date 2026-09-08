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

// GPU-side implementation of the PackScalarsOnGPU helper for PackOp.
//
// This file is compiled by NVCC / hipcc (i.e. as a .cu.cc source) which
// defines EIGEN_USE_GPU before including Eigen headers.  That is required for
// the Eigen device() assignment used below; doing the same in the plain .cc
// file (where EIGEN_USE_GPU is NOT defined) would trigger:
//
//   static assertion failed: "Default executor instantiated with non-default
//   device.  You must #define EIGEN_USE_THREADS, EIGEN_USE_GPU or
//   EIGEN_USE_SYCL before including Eigen headers."
//
// pack_op.cc forward-declares PackScalarsOnGPU<T> and calls it for the GPU
// device path; the linker resolves the call to the explicit instantiations
// at the bottom of this file.

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

// Packs `num` 0-D scalar input tensors (c->input(0) … c->input(num-1)) into
// a pre-allocated 1-D output tensor `output` of shape [num].
// Each scalar element i is written to output[i] via an Eigen GPU chip
// assignment so that the operation is correctly dispatched to the GPU stream.
template <typename T>
void PackScalarsOnGPU(OpKernelContext* c, int num, Tensor* output) {
  const Tensor& first_input = c->input(0);
  auto output_vec = output->flat<T>();
  const Eigen::GpuDevice& d = c->eigen_device<Eigen::GpuDevice>();
  for (int i = 0; i < num; ++i) {
    const Tensor& input = c->input(i);
    OP_REQUIRES(c, first_input.shape().IsSameSize(input.shape()),
                absl::InvalidArgumentError(absl::StrCat(
                    "Shapes of all inputs must match: values[0].shape = ",
                    first_input.shape().DebugString(), " != values[", i,
                    "].shape = ", input.shape().DebugString())));
    output_vec.template chip<0>(i).device(d) = input.scalar<T>();
  }
}

// Explicit instantiations for all types registered with REGISTER_GPU in
// pack_op.cc.  Must match TF_CALL_GPU_ALL_TYPES + the extra integer types.
#define INSTANTIATE(T) template void PackScalarsOnGPU<T>(OpKernelContext*, int, Tensor*);

TF_CALL_int64(INSTANTIATE);
TF_CALL_int16(INSTANTIATE);
TF_CALL_uint32(INSTANTIATE);
TF_CALL_uint64(INSTANTIATE);
TF_CALL_GPU_ALL_TYPES(INSTANTIATE);
TF_CALL_float8_e5m2(INSTANTIATE);
TF_CALL_float8_e4m3fn(INSTANTIATE);

#undef INSTANTIATE

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
