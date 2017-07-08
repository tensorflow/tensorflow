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
#include "tensorflow/core/kernels/tile_functor.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {
namespace internal {

template <typename T>
__global__ void TileKernel(int nthreads, const T* src, const int32* buf,
                           const int32 ndims, T* dst) {
  const int32* in_strides = buf;
  const int32* out_strides = buf + ndims;
  const int32* in_dim_sizes = buf + ndims * 2;
  CUDA_1D_KERNEL_LOOP(o_idx, nthreads) {
    int32 i_idx = 0;
    int32 t = o_idx;
    for (int i = 0; i < ndims; ++i) {
      i_idx += t / out_strides[i] % in_dim_sizes[i] * in_strides[i];
      t %= out_strides[i];
    }
    dst[o_idx] = ldg(src + i_idx);
  }
}

template <typename Device, typename T>
void TileSimple(const Device& d, Tensor* out, const Tensor& in) {
  // Ensures we can use 32-bit index.
  const int64 in_nelem = in.NumElements();
  CHECK_LT(in_nelem, kint32max) << "Tensor too large to transpose on GPU";
  const int64 out_nelem = out->NumElements();
  CHECK_LT(out_nelem, kint32max) << "Tensor too large to transpose on GPU";
  // Pack strides and input dimension sizes into one buffer.
  const int32 ndims = in.dims();
  gtl::InlinedVector<int32, 24> host_buf(ndims * 3);
  gtl::InlinedVector<int32, 8> in_strides = ComputeStride<int32>(in.shape());
  gtl::InlinedVector<int32, 8> out_strides = ComputeStride<int32>(out->shape());
  for (int i = 0; i < ndims; ++i) {
    host_buf[i] = in_strides[i];
    host_buf[ndims + i] = out_strides[i];
    host_buf[ndims * 2 + i] = in.dim_size(i);
  }
  // Copies the input strides, output strides and input dimension sizes to the device.
  auto num_bytes = sizeof(int64) * host_buf.size();
  auto dev_buf = d.allocate(num_bytes);
  // NOTE: host_buf is not allocated by CudaHostAllocator, and
  // therefore we are doing a sync copy effectively.
  d.memcpyHostToDevice(dev_buf, host_buf.data(), num_bytes);
  // Launch kernel to q[...] = p[...].
  const T* p = in.flat<T>().data();
  T* q = out->flat<T>().data();
  CudaLaunchConfig cfg = GetCudaLaunchConfig(out_nelem, d);
  TileKernel<<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(
      cfg.virtual_thread_count, p, reinterpret_cast<const int32*>(dev_buf),
      ndims, q);
  // Safe to deallocate immediately after the kernel launch.
  d.deallocate(dev_buf);
}

}  // end namespace internal

namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// Register functors used for Tile functor.
#define DEFINE_TYPE(T) template struct Tile<GPUDevice, T>;

TF_CALL_int16(DEFINE_TYPE);
TF_CALL_int32(DEFINE_TYPE);
TF_CALL_int64(DEFINE_TYPE);
TF_CALL_float(DEFINE_TYPE);
TF_CALL_double(DEFINE_TYPE);
TF_CALL_half(DEFINE_TYPE);
TF_CALL_complex64(DEFINE_TYPE);
TF_CALL_complex128(DEFINE_TYPE);

#undef DEFINE_TYPE

}  // end namespace functor
}  // namespace tensorflow
#endif  // GOOGLE_CUDA
