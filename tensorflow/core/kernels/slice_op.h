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

#ifndef TENSORFLOW_KERNELS_SLICE_OP_H_
#define TENSORFLOW_KERNELS_SLICE_OP_H_

// Functor definition for SliceOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/ops_util.h"

namespace tensorflow {

namespace internal {

#if GOOGLE_CUDA

#include "tensorflow/core/util/cuda_kernel_helper.h"

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
void SliceSimple(const Device& d, Tensor* out, const Tensor& in,
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
#else
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
#endif

template <typename Device, typename T, int NDIMS>
void SliceUsingEigen(const Device& d, Tensor* out, const Tensor& in,
                 const gtl::ArraySlice<int64>& slice_indices,
                 const gtl::ArraySlice<int64>& slice_sizes) {
  auto input = in.tensor<T, NDIMS>();
  auto output = out->tensor<T, NDIMS>();
  Eigen::DSizes<int, NDIMS> indices;
  for (int i = 0; i < NDIMS; ++i) {
    indices[i] = slice_indices[i];
  }
  Eigen::DSizes<int, NDIMS> sizes;
  for (int i = 0; i < NDIMS; ++i) {
    sizes[i] = slice_sizes[i];
  }
  bool use_64bit = (input.size() > Eigen::NumTraits<int>::highest());
  if (!use_64bit &&
      Eigen::internal::is_same<Device, Eigen::GpuDevice>::value) {
    To32Bit(output).device(d) = To32Bit(input).slice(indices, sizes);
  } else {
    output.device(d) = input.slice(indices, sizes);
  }
}

} // namespace internal

namespace functor {

template <typename Device, typename T>
struct Slice {
  void operator()(const Device& d, Tensor* out, const Tensor& in,
                  const gtl::ArraySlice<int64>& slice_indices,
                  const gtl::ArraySlice<int64>& slice_sizes) {
    switch (in.dims()) {
      case 1:
        internal::SliceUsingEigen<Device, T, 1>(d, out, in, slice_indices, slice_sizes);
        break;
      case 2:
        internal::SliceUsingEigen<Device, T, 2>(d, out, in, slice_indices, slice_sizes);
        break;
      case 3:
        internal::SliceUsingEigen<Device, T, 3>(d, out, in, slice_indices, slice_sizes);
        break;
      case 4:
        internal::SliceUsingEigen<Device, T, 4>(d, out, in, slice_indices, slice_sizes);
        break;
      case 5:
        internal::SliceUsingEigen<Device, T, 5>(d, out, in, slice_indices, slice_sizes);
        break;
      case 6:
        internal::SliceUsingEigen<Device, T, 6>(d, out, in, slice_indices, slice_sizes);
        break;
      case 7:
        internal::SliceUsingEigen<Device, T, 7>(d, out, in, slice_indices, slice_sizes);
        break;
      default:
        internal::SliceSimple<Device, T>(d, out, in, slice_indices);
        break;
    }
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_SLICE_OP_H_
