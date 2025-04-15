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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/kernels/inplace_ops_functor.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice Device;

template <typename T>
__global__ void DoParallelConcatOpKernel(int nthreads, const int64 rows,
                                         const int64 cols, int32 loc,
                                         const T* __restrict__ src,
                                         T* __restrict__ dst) {
  GPU_1D_KERNEL_LOOP(idx, nthreads) {
    int64 c = idx % cols;
    int64 r = (loc % rows + rows) % rows;  // Guard index range.
    T* p = dst + r * cols + c;
    const T* q = src + idx;
    *p = ldg(q);
  }
}

template <typename T>
Status DoParallelConcatUpdate(const Device& d, const Tensor& value, int32 loc,
                              Tensor* output) {
  const int64 nelem = value.NumElements();
  GpuLaunchConfig cfg = GetGpuLaunchConfig(nelem, d);
  auto Toutput = output->flat_outer_dims<T>();
  const int64 nrows = Toutput.dimension(0);
  const int64 ncols = Toutput.dimension(1);
  const T* src = value.flat<T>().data();
  T* dst = output->flat<T>().data();
  TF_CHECK_OK(GpuLaunchKernel(
      DoParallelConcatOpKernel<T>, cfg.block_count, cfg.thread_per_block, 0,
      d.stream(), cfg.virtual_thread_count, nrows, ncols, loc, src, dst));
  return OkStatus();
}

template <>
Status DoParallelConcat(const Device& d, const Tensor& value, int32 loc,
                        Tensor* output) {
  CHECK_EQ(value.dtype(), output->dtype());
  switch (value.dtype()) {
#define CASE(type)                                              \
  case DataTypeToEnum<type>::value:                             \
    return DoParallelConcatUpdate<type>(d, value, loc, output); \
    break;

    CASE(float)
    CASE(double)
    CASE(Eigen::half)
    CASE(Eigen::bfloat16)
// Using TF_CALL_GPU_NUMBER_TYPES(CASE) results in the compiler complaining
// that CASE is not defined...hence the above construction
#undef CASE
    default:
      return errors::InvalidArgument("Unsupported data type: ",
                                     DataTypeString(value.dtype()));
  }
  return OkStatus();
}

template <typename T, InplaceOpType op>
__global__ void DoInplaceOpKernel(int nthreads, const int64 rows,
                                  const int64 cols, const int64 n,
                                  const T* __restrict__ src,
                                  const int32* __restrict__ rowids,
                                  T* __restrict__ dst) {
  GPU_1D_KERNEL_LOOP(idx, nthreads) {
    int64 r = idx / cols;
    int64 c = idx % cols;
    r = (rowids[r] % rows + rows) % rows;  // Guard index range.
    T* p = dst + r * cols + c;
    const T* q = src + idx;
    switch (op) {
      case I_UPDATE:
        *p = ldg(q);
        break;
      case I_ADD:
        *p += ldg(q);
        break;
      case I_SUB:
        *p -= ldg(q);
        break;
    }
  }
}

template <typename T>
void DoInplaceOp(const Device& d, InplaceOpType op, const Tensor& i,
                 const Tensor& v, Tensor* y) {
  const int64 nelem = v.NumElements();
  GpuLaunchConfig cfg = GetGpuLaunchConfig(nelem, d);
  auto Ty = y->flat_outer_dims<T>();
  const int64 nrows = Ty.dimension(0);
  const int64 ncols = Ty.dimension(1);
  const int64 n = i.NumElements();
  const T* src = v.flat<T>().data();
  // TODO(sjhwang): Check that first dimension fits in int32 range.
  const int32* rowids = i.flat<int32>().data();
  T* dst = y->flat<T>().data();
  switch (op) {
    case I_UPDATE:
      TF_CHECK_OK(GpuLaunchKernel(DoInplaceOpKernel<T, I_UPDATE>,
                                  cfg.block_count, cfg.thread_per_block, 0,
                                  d.stream(), cfg.virtual_thread_count, nrows,
                                  ncols, n, src, rowids, dst));
      break;
    case I_ADD:
      TF_CHECK_OK(GpuLaunchKernel(DoInplaceOpKernel<T, I_ADD>, cfg.block_count,
                                  cfg.thread_per_block, 0, d.stream(),
                                  cfg.virtual_thread_count, nrows, ncols, n,
                                  src, rowids, dst));
      break;
    case I_SUB:
      TF_CHECK_OK(GpuLaunchKernel(DoInplaceOpKernel<T, I_SUB>, cfg.block_count,
                                  cfg.thread_per_block, 0, d.stream(),
                                  cfg.virtual_thread_count, nrows, ncols, n,
                                  src, rowids, dst));
      break;
  }
}

template <bool>
void DoInplaceOp(const Device& d, InplaceOpType op, const Tensor& i,
                 const Tensor& v, Tensor* y) {
  const int64 nelem = v.NumElements();
  GpuLaunchConfig cfg = GetGpuLaunchConfig(nelem, d);
  auto Ty = y->flat_outer_dims<bool>();
  const int64 nrows = Ty.dimension(0);
  const int64 ncols = Ty.dimension(1);
  const int64 n = i.NumElements();
  const bool* src = v.flat<bool>().data();
  // TODO(sjhwang): Check that first dimension fits in int32 range.
  const int32* rowids = i.flat<int32>().data();
  bool* dst = y->flat<bool>().data();
  if (op == I_UPDATE) {
    TF_CHECK_OK(GpuLaunchKernel(DoInplaceOpKernel<bool, I_UPDATE>,
                                cfg.block_count, cfg.thread_per_block, 0,
                                d.stream(), cfg.virtual_thread_count, nrows,
                                ncols, n, src, rowids, dst));
  }
}

template <>
Status DoInplace(const Device& d, InplaceOpType op, const Tensor& i,
                 const Tensor& v, Tensor* y) {
  CHECK_EQ(v.dtype(), y->dtype());
  switch (v.dtype()) {
#define CASE(type)                     \
  case DataTypeToEnum<type>::value:    \
    DoInplaceOp<type>(d, op, i, v, y); \
    break;

    CASE(bool)
    CASE(float)
    CASE(double)
    CASE(Eigen::half)
    CASE(Eigen::bfloat16)
    CASE(uint8_t)
    CASE(int8_t)
    CASE(int64_t)
    CASE(uint64_t)
#undef CASE
    default:
      return errors::InvalidArgument("Unsupported data type from DoInplace: ",
                                     DataTypeString(v.dtype()));
  }
  return OkStatus();
}

template <>
Status DoCopy(const Device& d, const Tensor& x, Tensor* y) {
  CHECK_EQ(x.dtype(), y->dtype());
  switch (x.dtype()) {
#define CASE(type)                              \
  case DataTypeToEnum<type>::value:             \
    y->flat<type>().device(d) = x.flat<type>(); \
    break;

    CASE(bool)
    CASE(float)
    CASE(double)
    CASE(Eigen::half)
    CASE(Eigen::bfloat16)
    CASE(complex64)
    CASE(complex128)
    CASE(uint8_t)
    CASE(int8_t)
    CASE(int64_t)
    CASE(uint64_t)
#undef CASE
    default:
      return errors::InvalidArgument("Unsupported dtype from DoCopy: ",
                                     DataTypeString(x.dtype()));
  }
  return OkStatus();
}

}  // end namespace functor
}  // namespace tensorflow
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
