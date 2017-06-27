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

#include "tensorflow/core/kernels/slice_op.h"

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace internal {

template <typename Device, typename T>
void SliceSimple(const Device& d, Tensor* out, const Tensor& in,
                 const gtl::ArraySlice<int64>& slice_indices) {
  const int ndims = in.dims();
  const int64 nelem = out->NumElements();
  gtl::InlinedVector<int64, 8> in_strides = ComputeStride<int64>(in.shape());
  gtl::InlinedVector<int64, 8> out_strides = ComputeStride<int64>(out->shape());
  const T* p = in.flat<T>().data();
  T* q = out->flat<T>().data();

  for (int64 o_idx = 0; o_idx < nelem; ++o_idx) {
    int64 i_idx = 0;
    int64 t = o_idx;
    for (int i = 0; i < ndims; ++i) {
      i_idx += (t / out_strides[i] + slice_indices[i]) * in_strides[i];
      t %= out_strides[i];
    }
    q[o_idx] = p[i_idx];
  }
}

} // namespace internel

using CpuDevice = Eigen::ThreadPoolDevice;

#define DEFINE_CPU_KERNELS(T) template struct functor::Slice<CpuDevice, T>;

TF_CALL_ALL_TYPES(DEFINE_CPU_KERNELS);
DEFINE_CPU_KERNELS(bfloat16);

#undef DEFINE_CPU_KERNELS

#ifdef TENSORFLOW_USE_SYCL
using SyclDevice = Eigen::SyclDevice;

#define DEFINE_SYCL_KERNELS(T) template struct functor::Slice<SyclDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_SYCL_KERNELS);
DEFINE_SYCL_KERNELS(int32);

#undef DEFINE_SYCL_KERNELS
#endif // TENSORFLOW_USE_SYCL

} // namespace tensorflow
