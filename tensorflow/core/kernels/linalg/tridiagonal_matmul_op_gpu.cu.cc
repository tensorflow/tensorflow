/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/linalg_ops.cc.

#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/cuda_solvers.h"
#include "tensorflow/core/util/cuda_sparse.h"
#include "tensorflow/core/util/gpu_device_functions.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"

namespace tensorflow {

template <typename Scalar>
__global__ void TridiagonalMatMulKernel(int batch_size, int m, int n,
                                        const Scalar* __restrict__ superdiag,
                                        const Scalar* __restrict__ maindiag,
                                        const Scalar* __restrict__ subdiag,
                                        const Scalar* __restrict__ rhs,
                                        Scalar* __restrict__ product) {
  for (int i : CudaGridRangeX(batch_size * m * n)) {
    int row_id = i / n;
    Scalar result = maindiag[row_id] * rhs[i];
    if (row_id % m != 0) {
      result = result + subdiag[row_id] * rhs[i - n];
    }
    if ((row_id + 1) % m != 0) {
      result = result + superdiag[row_id] * rhs[i + n];
    }
    product[i] = result;
  }
}

template <typename Scalar>
class TridiagonalMatMulOpGpu : public OpKernel {
 public:
  explicit TridiagonalMatMulOpGpu(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) final {
    const Tensor& superdiag = context->input(0);
    const Tensor& maindiag = context->input(1);
    const Tensor& subdiag = context->input(2);
    const Tensor& rhs = context->input(3);

    const int ndims = rhs.dims();
    int64 batch_size = 1;
    for (int i = 0; i < ndims - 2; i++) {
      batch_size *= rhs.dim_size(i);
    }
    const int m = rhs.dim_size(ndims - 2);
    const int n = rhs.dim_size(ndims - 1);

    // Allocate output.
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, rhs.shape(), &output));

    const Eigen::GpuDevice& device = context->eigen_device<Eigen::GpuDevice>();
    GpuLaunchConfig cfg = GetGpuLaunchConfig(1, device);
    TF_CHECK_OK(GpuLaunchKernel(
        TridiagonalMatMulKernel<Scalar>, cfg.block_count, cfg.thread_per_block,
        0, device.stream(), batch_size, m, n, superdiag.flat<Scalar>().data(),
        maindiag.flat<Scalar>().data(), subdiag.flat<Scalar>().data(),
        rhs.flat<Scalar>().data(), output->flat<Scalar>().data()));
  }
};

REGISTER_LINALG_OP_GPU("TridiagonalMatMul", (TridiagonalMatMulOpGpu<float>),
                       float);
REGISTER_LINALG_OP_GPU("TridiagonalMatMul", (TridiagonalMatMulOpGpu<double>),
                       double);
REGISTER_LINALG_OP_GPU("TridiagonalMatMul", (TridiagonalMatMulOpGpu<complex64>),
                       complex64);
REGISTER_LINALG_OP_GPU("TridiagonalMatMul",
                       (TridiagonalMatMulOpGpu<complex128>), complex128);
}  // namespace tensorflow

#endif  // GOOGLE_CUDA
