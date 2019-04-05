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

#include "tensorflow/core/common_runtime/gpu/gpu_event_mgr.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/cuda_solvers.h"
#include "tensorflow/core/kernels/cuda_sparse.h"
#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/cuda_device_functions.h"
#include "tensorflow/core/util/cuda_launch_config.h"

namespace tensorflow {

static const char kNotInvertibleMsg[] = "The matrix is not invertible.";

static const char kNotInvertibleScalarMsg[] =
    "The matrix is not invertible: it is a scalar with value zero.";

template <typename Scalar>
__global__ void SolveForSizeOneOrTwoKernel(const int m, const Scalar* diags,
                                           const Scalar* rhs, const int num_rhs,
                                           Scalar* x, bool* not_invertible) {
  if (m == 1) {
    if (diags[1] == Scalar(0)) {
      *not_invertible = true;
      return;
    }
    for (int i : CudaGridRangeX(num_rhs)) {
      x[i] = rhs[i] / diags[1];
    }
  } else {
    Scalar det = diags[2] * diags[3] - diags[0] * diags[5];
    if (det == Scalar(0)) {
      *not_invertible = true;
      return;
    }
    for (int i : CudaGridRangeX(num_rhs)) {
      x[i] = (diags[3] * rhs[i] - diags[0] * rhs[i + num_rhs]) / det;
      x[i + num_rhs] = (diags[2] * rhs[i + num_rhs] - diags[5] * rhs[i]) / det;
    }
  }
}

template <typename Scalar>
class TridiagonalSolveOpGpu : public LinearAlgebraOp<Scalar> {
 public:
  INHERIT_LINALG_TYPEDEFS(Scalar);

  explicit TridiagonalSolveOpGpu(OpKernelConstruction* context)
      : Base(context) {}

  void ValidateInputMatrixShapes(
      OpKernelContext* context,
      const TensorShapes& input_matrix_shapes) const final {
    auto num_inputs = input_matrix_shapes.size();
    OP_REQUIRES(context, num_inputs == 2,
                errors::InvalidArgument("Expected two input matrices, got ",
                                        num_inputs, "."));

    auto num_diags = input_matrix_shapes[0].dim_size(0);
    OP_REQUIRES(
        context, num_diags == 3,
        errors::InvalidArgument("Expected diagonals to be provided as a "
                                "matrix with 3 columns, got ",
                                num_diags, " columns."));

    auto num_rows1 = input_matrix_shapes[0].dim_size(1);
    auto num_rows2 = input_matrix_shapes[1].dim_size(0);
    OP_REQUIRES(context, num_rows1 == num_rows2,
                errors::InvalidArgument("Expected same number of rows in both "
                                        "arguments, got ",
                                        num_rows1, " and ", num_rows2, "."));
  }

  bool EnableInputForwarding() const final { return false; }

  TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shapes) const final {
    return TensorShapes({input_matrix_shapes[1]});
  }

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
    const auto diagonals = inputs[0];
    // Subdiagonal elements, first is ignored.
    const auto& superdiag = diagonals.row(0);
    // Diagonal elements.
    const auto& diag = diagonals.row(1);
    // Superdiagonal elements, last is ignored.
    const auto& subdiag = diagonals.row(2);
    // Right-hand sides.
    const auto& rhs = inputs[1];
    MatrixMap& x = outputs->at(0);
    const int m = diag.size();
    const int k = rhs.cols();

    if (m == 0) {
      return;
    }
    if (m < 3) {
      // Cusparse gtsv routine requires m >= 3. Solving manually for m < 3.
      SolveForSizeOneOrTwo(context, diagonals.data(), rhs.data(), x.data(), m,
                           k);
      return;
    }
    std::unique_ptr<CudaSparse> cusparse_solver(new CudaSparse(context));
    OP_REQUIRES_OK(context, cusparse_solver->Initialize());
    if (k == 1) {
      // rhs is copied into x, then gtsv replaces x with solution.
      CopyDeviceToDevice(context, rhs.data(), x.data(), m);
      SolveWithGtsv(context, cusparse_solver, superdiag.data(), diag.data(),
                    subdiag.data(), x.data(), m, 1);
    } else {
      // Gtsv expects rhs in column-major form, so we have to transpose.
      // rhs is transposed into temp, gtsv replaces temp with solution, then
      // temp is transposed into x.
      std::unique_ptr<CudaSolver> cublas_solver(new CudaSolver(context));
      Tensor temp;
      TensorShape temp_shape({k, m});
      OP_REQUIRES_OK(context,
                     cublas_solver->allocate_scoped_tensor(
                         DataTypeToEnum<Scalar>::value, temp_shape, &temp));
      TransposeWithGeam(context, cublas_solver, rhs.data(),
                        temp.flat<Scalar>().data(), m, k);
      SolveWithGtsv(context, cusparse_solver, superdiag.data(), diag.data(),
                    subdiag.data(), temp.flat<Scalar>().data(), m, k);
      TransposeWithGeam(context, cublas_solver, temp.flat<Scalar>().data(),
                        x.data(), k, m);
    }
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TridiagonalSolveOpGpu);

  void TransposeWithGeam(OpKernelContext* context,
                         const std::unique_ptr<CudaSolver>& cublas_solver,
                         const Scalar* src, Scalar* dst, const int src_rows,
                         const int src_cols) const {
    const Scalar zero(0), one(1);
    OP_REQUIRES_OK(context,
                   cublas_solver->Geam(CUBLAS_OP_T, CUBLAS_OP_N, src_rows,
                                       src_cols, &one, src, src_cols, &zero,
                                       static_cast<const Scalar*>(nullptr),
                                       src_rows, dst, src_rows));
  }

  void SolveWithGtsv(OpKernelContext* context,
                     std::unique_ptr<CudaSparse>& cusparse_solver,
                     const Scalar* superdiag, const Scalar* diag,
                     const Scalar* subdiag, Scalar* rhs, const int num_eqs,
                     const int num_rhs) const {
    OP_REQUIRES_OK(context,
                   cusparse_solver->Gtsv(num_eqs, num_rhs, subdiag, diag,
                                         superdiag, rhs, num_eqs));
  }

  void CopyDeviceToDevice(OpKernelContext* context, const Scalar* src,
                          Scalar* dst, const int num_elements) const {
    auto src_device_mem = AsDeviceMemory(src);
    auto dst_device_mem = AsDeviceMemory(dst);
    auto* stream = context->op_device_context()->stream();
    bool copy_status = stream
                           ->ThenMemcpyD2D(&dst_device_mem, src_device_mem,
                                           sizeof(Scalar) * num_elements)
                           .ok();
    if (!copy_status) {
      context->SetStatus(errors::Internal("Copying device-to-device failed."));
    }
  }

  se::DeviceMemory<Scalar> AsDeviceMemory(const Scalar* cuda_memory) const {
    se::DeviceMemoryBase wrapped(const_cast<Scalar*>(cuda_memory));
    se::DeviceMemory<Scalar> typed(wrapped);
    return typed;
  }

  void SolveForSizeOneOrTwo(OpKernelContext* context, const Scalar* diagonals,
                            const Scalar* rhs, Scalar* output, int m, int k) {
    const Eigen::GpuDevice& device = context->eigen_device<Eigen::GpuDevice>();
    CudaLaunchConfig cfg = GetCudaLaunchConfig(1, device);
    bool* not_invertible_dev;
    cudaMalloc(&not_invertible_dev, sizeof(bool));
    TF_CHECK_OK(CudaLaunchKernel(SolveForSizeOneOrTwoKernel<Scalar>,
                                 cfg.block_count, cfg.thread_per_block, 0,
                                 device.stream(), m, diagonals, rhs, k, output,
                                 not_invertible_dev));
    bool not_invertible_host;
    cudaMemcpy(&not_invertible_host, not_invertible_dev, sizeof(bool),
               cudaMemcpyDeviceToHost);
    cudaFree(not_invertible_dev);
    OP_REQUIRES(context, !not_invertible_host,
                errors::InvalidArgument(m == 1 ? kNotInvertibleScalarMsg
                                               : kNotInvertibleMsg));
  }
};

REGISTER_LINALG_OP_GPU("TridiagonalSolve", (TridiagonalSolveOpGpu<float>),
                       float);
REGISTER_LINALG_OP_GPU("TridiagonalSolve", (TridiagonalSolveOpGpu<double>),
                       double);
REGISTER_LINALG_OP_GPU("TridiagonalSolve", (TridiagonalSolveOpGpu<complex64>),
                       complex64);
REGISTER_LINALG_OP_GPU("TridiagonalSolve", (TridiagonalSolveOpGpu<complex128>),
                       complex128);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
