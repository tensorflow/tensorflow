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
#include "tensorflow/core/kernels/inplace_ops_functor.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice Device;

template <typename T, InplaceOpType op>
__global__ void DoInplaceOpKernel(int nthreads, const int64 rows,
                                  const int64 cols, const int64 n, const T* src,
                                  const int64* rowids, T* dst) {
  CUDA_1D_KERNEL_LOOP(idx, nthreads) {
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
Status DoInplaceUpdate(const Device& d, InplaceOpType op, const Tensor& value,
                       const Tensor& loc, Tensor* output) {
  const int64 nelem = value.NumElements();
  CudaLaunchConfig cfg = GetCudaLaunchConfig(nelem, d);
  auto Toutput = output->flat_outer_dims<T>();
  const int64 nrows = Toutput.dimension(0);
  const int64 ncols = Toutput.dimension(1);
  const int64 n = loc.NumElements();
  const T* src = value.flat<T>().data();
  const int64* rowids = loc.flat<int64>().data();
  T* dst = output->flat<T>().data();
  switch (op) {
    case I_UPDATE:
      DoInplaceOpKernel<T, I_UPDATE>
          <<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(
              cfg.virtual_thread_count, nrows, ncols, n, src, rowids, dst);
      break;
    case I_ADD:
      DoInplaceOpKernel<T, I_ADD>
          <<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(
              cfg.virtual_thread_count, nrows, ncols, n, src, rowids, dst);
      break;
    case I_SUB:
      DoInplaceOpKernel<T, I_SUB>
          <<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(
              cfg.virtual_thread_count, nrows, ncols, n, src, rowids, dst);
      break;
    default:
      return errors::InvalidArgument("Unsupported operation type", op);
  }
  return Status::OK();
}

template <>
Status DoInplace(const Device& d, InplaceOpType op, const Tensor& value,
                 const Tensor& loc, Tensor* output) {
  CHECK_EQ(value.dtype(), output->dtype());
  switch (value.dtype()) {
#define CASE(type)                                           \
  case DataTypeToEnum<type>::value:                          \
    return DoInplaceUpdate<type>(d, op, value, loc, output); \
    break;

    CASE(float)
    CASE(double)
    CASE(Eigen::half)
// Using TF_CALL_GPU_NUMBER_TYPES(CASE) results in the compiler complaining
// that CASE is not defined...hence the above construction
#undef CASE
    default:
      return errors::InvalidArgument("Unsupported data type: ", value.dtype());
  }
  return Status::OK();
}

}  // end namespace functor
}  // namespace tensorflow
#endif  // GOOGLE_CUDA
