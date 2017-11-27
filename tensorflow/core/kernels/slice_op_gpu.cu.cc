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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/slice_op.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {
namespace internal {

template <typename T>
__global__ void SliceKernel(int nthreads, const T* src, const int32* buf,
                            const int32 ndims, T* dst) {
  const int32* in_strides = buf;
  const int32* out_strides = buf + ndims;
  const int32* slice_indices = buf + ndims * 2;
  CUDA_1D_KERNEL_LOOP(o_idx, nthreads) {
    int32 i_idx = 0;
    int32 t = o_idx;
    for (int i = 0; i < ndims; ++i) {
      i_idx += (t / out_strides[i] + slice_indices[i]) * in_strides[i];
      t %= out_strides[i];
    }
    dst[o_idx] = ldg(src + i_idx);
  }
}

template <typename Device, typename T>
void SliceSimpleGpu(const Device& d, Tensor* out, const Tensor& in,
                 const gtl::ArraySlice<int64>& slice_indices) {
  // Ensures we can use 32-bit index.
  const int64 in_nelem = in.NumElements();
  CHECK_LT(in_nelem, kint32max) << "Tensor too large to transpose on GPU";
  const int64 out_nelem = out->NumElements();
  CHECK_LT(out_nelem, kint32max) << "Tensor too large to transpose on GPU";
  // Pack strides and slice indices sizes into one buffer.
  const int32 ndims = in.dims();
  gtl::InlinedVector<int32, 24> host_buf(ndims * 3);
  gtl::InlinedVector<int32, 8> in_strides = ComputeStride<int32>(in.shape());
  gtl::InlinedVector<int32, 8> out_strides = ComputeStride<int32>(out->shape());
  for (int i = 0; i < ndims; ++i) {
    host_buf[i] = in_strides[i];
    host_buf[ndims + i] = out_strides[i];
    host_buf[ndims * 2 + i] = slice_indices[i];
  }
  auto num_bytes = sizeof(int64) * host_buf.size();
  auto dev_buf = d.allocate(num_bytes);
  // NOTE: host_buf is not allocated by CudaHostAllocator, and
  // therefore we are doing a sync copy effectively.
  d.memcpyHostToDevice(dev_buf, host_buf.data(), num_bytes);
  // Launch kernel to q[...] = p[...].
  const T* p = in.flat<T>().data();
  T* q = out->flat<T>().data();
  CudaLaunchConfig cfg = GetCudaLaunchConfig(out_nelem, d);
  SliceKernel<<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(
      cfg.virtual_thread_count, p, reinterpret_cast<const int32*>(dev_buf),
      ndims, q);
  // Safe to deallocate immediately after the kernel launch.
  d.deallocate(dev_buf);
}

} // namespace internal

typedef Eigen::GpuDevice GPUDevice;

#define DEFINE_GPU_KERNELS(T)                      \
  template struct functor::Slice<GPUDevice, T, 1>; \
  template struct functor::Slice<GPUDevice, T, 2>; \
  template struct functor::Slice<GPUDevice, T, 3>; \
  template struct functor::Slice<GPUDevice, T, 4>; \
  template struct functor::Slice<GPUDevice, T, 5>; \
  template struct functor::Slice<GPUDevice, T, 6>; \
  template struct functor::Slice<GPUDevice, T, 7>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_KERNELS);
TF_CALL_complex64(DEFINE_GPU_KERNELS);
TF_CALL_complex128(DEFINE_GPU_KERNELS);
DEFINE_GPU_KERNELS(int32);

#undef DEFINE_GPU_KERNELS

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
