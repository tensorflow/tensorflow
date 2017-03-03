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
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {
namespace internal {

template <typename T>
__global__ void TransposeKernel(int nthreads, const T* src, const int32* buf,
                                const int32 ndims, T* dst) {
  const int32* in_strides = buf;
  const int32* out_strides = buf + ndims;
  const int32* perm = buf + ndims * 2;
  CUDA_1D_KERNEL_LOOP(o_idx, nthreads) {
    int32 i_idx = 0;
    int32 t = o_idx;
    for (int i = 0; i < ndims; ++i) {
      i_idx += (t / out_strides[i]) * in_strides[perm[i]];
      t = t % out_strides[i];
    }
    dst[o_idx] = ldg(src + i_idx);
  }
}

template <typename Device, typename T>
void TransposeSimple(const Device& d, const Tensor& in,
                     const gtl::ArraySlice<int32> perm, Tensor* out) {
  // Ensures we can use 32-bit index.
  const int64 nelem = in.NumElements();
  CHECK_LT(nelem, kint32max) << "Tensor too large to transpose on GPU";
  // Pack strides and permutation into one buffer.
  const int32 ndims = in.dims();
  gtl::InlinedVector<int32, 16> host_buf(ndims * 3);
  // Input strides.
  ComputeStride(in.shape(), &host_buf[0]);
  // Output strides.
  ComputeStride(out->shape(), &host_buf[ndims]);
  // Dimension permutation.
  for (int i = 0; i < ndims; ++i) {
    host_buf[ndims * 2 + i] = perm[i];
  }
  // Copies the input strides, output strides and permutation to the device.
  auto num_bytes = sizeof(int64) * host_buf.size();
  auto dev_buf = d.allocate(num_bytes);
  // NOTE: host_buf is not allocated by CudaHostAllocator, and
  // therefore we are doing a sync copy effectively.
  d.memcpyHostToDevice(dev_buf, host_buf.data(), num_bytes);
  // Launch kernel to q[...] = p[...].
  const T* p = reinterpret_cast<const T*>(in.tensor_data().data());
  T* q = reinterpret_cast<T*>(const_cast<char*>((out->tensor_data().data())));
  CudaLaunchConfig cfg = GetCudaLaunchConfig(nelem, d);
  TransposeKernel<<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(
      cfg.virtual_thread_count, p, reinterpret_cast<const int32*>(dev_buf),
      ndims, q);
  // Safe to deallocate immediately after the kernel launch.
  d.deallocate(dev_buf);
}

template <typename Device, typename T, int NDIMS>
void TransposeUsingEigen(const Device& d, const Tensor& in,
                         const gtl::ArraySlice<int32> perm, Tensor* out) {
  Eigen::array<int, NDIMS> p;
  for (int i = 0; i < NDIMS; ++i) p[i] = perm[i];
  auto x = typename TTypes<T, NDIMS>::ConstTensor(
      reinterpret_cast<const T*>(in.tensor_data().data()),
      in.shape().AsEigenDSizes<NDIMS>());
  auto y = typename TTypes<T, NDIMS>::Tensor(
      reinterpret_cast<T*>(const_cast<char*>(out->tensor_data().data())),
      out->shape().AsEigenDSizes<NDIMS>());
  y.device(d) = x.shuffle(p);
}

}  // end namespace internal

typedef Eigen::GpuDevice Device;

template <>
Status DoTranspose<Device>(const Device& d, const Tensor& in,
                           const gtl::ArraySlice<int32> perm, Tensor* out) {
  CHECK_GE(in.dims(), 2);
  CHECK_EQ(in.dims(), out->dims());
  CHECK_EQ(in.dims(), perm.size());
  CHECK_EQ(in.dtype(), out->dtype());
  switch (in.dtype()) {
    case DT_BOOL:
    case DT_INT8:
    case DT_QINT8:
    case DT_QUINT8:
    case DT_UINT8:
      internal::Transpose<Device, uint8>(d, in, perm, out);
      break;

    case DT_BFLOAT16:
    case DT_HALF:
    case DT_INT16:
    case DT_QINT16:
    case DT_QUINT16:
    case DT_UINT16:
      internal::Transpose<Device, uint16>(d, in, perm, out);
      break;

    case DT_FLOAT:
    case DT_INT32:
    case DT_QINT32:
      internal::Transpose<Device, uint32>(d, in, perm, out);
      break;

    case DT_COMPLEX64:
    case DT_DOUBLE:
    case DT_INT64:
      internal::Transpose<Device, uint64>(d, in, perm, out);
      break;

    case DT_COMPLEX128:
      internal::Transpose<Device, float4>(d, in, perm, out);
      break;

    default:
      return errors::Unimplemented("Unsupported dtype on GPU: ", in.dtype());
  }
  return Status::OK();
}

}  // namespace tensorflow
#endif  // GOOGLE_CUDA
