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

// See docs in ../ops/linalg_ops.cc.
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif

#include <numeric>

#include "Eigen/Core"  // from @eigen_archive
#include "Eigen/LU"  // from @eigen_archive
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/util/gpu_solvers.h"
#endif

namespace tensorflow {

static const char kErrMsg[] = "Input matrix is not invertible.";

template <class Scalar>
class MatrixSolveOp : public LinearAlgebraOp<Scalar> {
 public:
  INHERIT_LINALG_TYPEDEFS(Scalar);

  explicit MatrixSolveOp(OpKernelConstruction* context) : Base(context) {
    OP_REQUIRES_OK(context, context->GetAttr("adjoint", &adjoint_));
  }

  void ValidateInputMatrixShapes(
      OpKernelContext* context,
      const TensorShapes& input_matrix_shapes) const final {
    Base::ValidateSquareSolver(context, input_matrix_shapes);
  }

  TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shapes) const final {
    return TensorShapes({TensorShape({input_matrix_shapes[0].dim_size(1),
                                      input_matrix_shapes[1].dim_size(1)})});
  }

  int64_t GetCostPerUnit(const TensorShapes& input_matrix_shapes) const final {
    double rows = static_cast<double>(input_matrix_shapes[0].dim_size(0));
    double num_rhss = static_cast<double>(input_matrix_shapes[1].dim_size(1));
    double cost = rows * rows * (rows + num_rhss);
    return cost >= static_cast<double>(kint64max) ? kint64max
                                                  : static_cast<int64_t>(cost);
  }

  bool EnableInputForwarding() const final { return false; }

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
    const ConstMatrixMap& matrix = inputs[0];
    const ConstMatrixMap& rhs = inputs[1];
    if (matrix.rows() == 0 || matrix.cols() == 0 || rhs.cols() == 0) {
      // To be consistent with the MatrixInverse op, we define the solution for
      // an empty set of equation as the empty matrix.
      return;
    }
    Eigen::PartialPivLU<Matrix> lu_decomposition(matrix.rows());
    if (adjoint_) {
      // TODO(rmlarsen): For Eigen 3.2, this creates a temporary copy.
      // Make sure to backport: https://bitbucket.org/eigen/eigen/commits/
      // bd2219a74c96dfe3f6bc2c23588749e36d2d8173
      lu_decomposition.compute(matrix.adjoint());
    } else {
      lu_decomposition.compute(matrix);
    }

    // PartialPivLU cannot give strong guarantees on invertibility,
    // but we can at least guard against exact zero pivots. This can occur as
    // a result of basic user mistakes such providing integer valued
    // matrices that are exactly singular, or due to underflow if this
    // code is run with denormals being flushed to zero.
    const RealScalar min_abs_pivot =
        lu_decomposition.matrixLU().diagonal().cwiseAbs().minCoeff();
    OP_REQUIRES(context, min_abs_pivot > RealScalar(0),
                errors::InvalidArgument(kErrMsg));

    // TODO(rmlarsen): Add check based on condition number estimation.
    // The necessary changes to Eigen are in
    // https://bitbucket.org/eigen/eigen/pull-requests/174/
    // add-matrix-condition-number-estimation/diff
    outputs->at(0) = lu_decomposition.solve(rhs);
  }

 private:
  bool adjoint_;

  MatrixSolveOp(const MatrixSolveOp&) = delete;
  void operator=(const MatrixSolveOp&) = delete;
};

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
typedef Eigen::GpuDevice GPUDevice;

template <class Scalar>
class MatrixSolveOpGpu : public AsyncOpKernel {
 public:
  explicit MatrixSolveOpGpu(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("adjoint", &adjoint_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) final {
    const Tensor& input = context->input(0);
    const Tensor& rhs = context->input(1);
    const int ndims = input.dims();
    const int64_t n = input.dim_size(ndims - 1);
    const int64_t nrhs = rhs.dim_size(ndims - 1);
    // Validate inputs.
    OP_REQUIRES_ASYNC(
        context, ndims >= 2,
        errors::InvalidArgument("Input must have rank >= 2, got ", ndims),
        done);
    OP_REQUIRES_ASYNC(context, rhs.dims() == ndims,
                      errors::InvalidArgument(
                          "Input and right-hand side must have same rank, got ",
                          ndims, " != ", rhs.dims()),
                      done);
    OP_REQUIRES_ASYNC(
        context, input.dim_size(ndims - 2) == n,
        errors::InvalidArgument("Input matrices must be squares, got ",
                                input.dim_size(ndims - 2), " != ", n),
        done);
    OP_REQUIRES_ASYNC(context, rhs.dim_size(ndims - 2) == n,
                      errors::InvalidArgument(
                          "Input matrix and right-hand side must have the "
                          "same number of rows, got ",
                          n, " != ", rhs.dim_size(ndims - 2)),
                      done);
    for (int dim = 0; dim < ndims - 2; dim++) {
      OP_REQUIRES_ASYNC(
          context, input.dim_size(dim) == rhs.dim_size(dim),
          errors::InvalidArgument(
              "All input tensors must have the same outer dimensions."),
          done);
    }

    // Allocate output.
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->forward_input_or_allocate_output({1}, 0, rhs.shape(), &output),
        done);

    // To be consistent with the MatrixInverse op, we define the solution for
    // an empty set of equations as the empty matrix.
    if (input.NumElements() == 0 || rhs.NumElements() == 0) {
      done();
      return;
    }

    // TODO(rmlarsen): Convert to std::make_unique when available.
    std::unique_ptr<GpuSolver> solver(new GpuSolver(context));

    // Make a copy of the input for the factorization step, or, if adjoint_ is
    // false, try to reuse the input buffer if this op owns it exclusively.
    Tensor input_copy;
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    if (adjoint_) {
      // For the adjoint case, it is simpler to always make a transposed copy up
      // front.
      OP_REQUIRES_OK_ASYNC(
          context,
          solver->allocate_scoped_tensor(DataTypeToEnum<Scalar>::value,
                                         input.shape(), &input_copy),
          done);
      OP_REQUIRES_OK_ASYNC(context,
                           DoMatrixTranspose(device, input, &input_copy), done);
    } else {
      OP_REQUIRES_OK_ASYNC(
          context,
          solver->forward_input_or_allocate_scoped_tensor(
              {0}, DataTypeToEnum<Scalar>::value, input.shape(), &input_copy),
          done);
      if (!input.SharesBufferWith(input_copy)) {
        device.memcpy(input_copy.flat<Scalar>().data(),
                      input.flat<Scalar>().data(),
                      input.NumElements() * sizeof(Scalar));
      }
    }
    auto input_copy_reshaped = input_copy.template flat_inner_dims<Scalar, 3>();
    const int64_t batch_size = input_copy_reshaped.dimension(0);

    // Allocate pivots on the device.
    Tensor pivots;
    OP_REQUIRES_OK_ASYNC(
        context,
        solver->allocate_scoped_tensor(DataTypeToEnum<int>::value,
                                       TensorShape{batch_size, n}, &pivots),
        done);
    auto pivots_mat = pivots.template matrix<int>();

    // 1. Compute the partially pivoted LU factorization(s) of the
    // matrix/matrices.
    std::vector<DeviceLapackInfo> dev_info;
    auto input_copy_ptrs = solver->GetScratchSpace<uint8>(
        sizeof(Scalar*) * batch_size, "input_copt_ptrs",
        /* on_host */ true);
    const int kMaxMatrixSizeToBatchSizeRatio = 128;
    const bool use_batched_solver =
        n <= kMaxMatrixSizeToBatchSizeRatio * batch_size;
    if (use_batched_solver) {
      // For small matrices or large batch sizes, we use the batched interface
      // from cuBlas/rocmSolver.
#if GOOGLE_CUDA
      const Scalar** input_copy_ptrs_base =
          reinterpret_cast<const Scalar**>(input_copy_ptrs.mutable_data());
#else  // TENSORFLOW_USE_ROCM
      Scalar** input_copy_ptrs_base =
          reinterpret_cast<Scalar**>(input_copy_ptrs.mutable_data());
#endif
      for (int batch = 0; batch < batch_size; ++batch) {
        input_copy_ptrs_base[batch] = &input_copy_reshaped(batch, 0, 0);
      }
      dev_info.push_back(
          solver->GetDeviceLapackInfo(batch_size, "getrfBatched"));
      OP_REQUIRES_OK_ASYNC(
          context,
          solver->GetrfBatched(n, input_copy_ptrs_base, n, pivots_mat.data(),
                               &dev_info.back(), batch_size),
          done);
    } else {
      // For small batch sizes or large matrices, we use the non-batched
      // interface from cuSolver, which is much faster for large matrices.
      dev_info.push_back(solver->GetDeviceLapackInfo(batch_size, "getrf"));
      for (int batch = 0; batch < batch_size; ++batch) {
        OP_REQUIRES_OK_ASYNC(
            context,
            solver->Getrf(n, n, &input_copy_reshaped(batch, 0, 0), n,
                          &pivots_mat(batch, 0), &dev_info.back()(batch)),
            done);
      }
    }

    // 2. Make a transposed copy of the right-hand sides. This is necessary
    // because cuBLAS/rocSolver assumes column-major storage while TensorFlow TF
    // uses row-major.
    TensorShape transposed_rhs_shape(rhs.shape());
    transposed_rhs_shape.RemoveLastDims(2);
    transposed_rhs_shape.AddDim(nrhs);
    transposed_rhs_shape.AddDim(n);
    Tensor transposed_rhs;
    OP_REQUIRES_OK_ASYNC(
        context,
        solver->allocate_scoped_tensor(DataTypeToEnum<Scalar>::value,
                                       transposed_rhs_shape, &transposed_rhs),
        done);
    if (nrhs > 1) {
      OP_REQUIRES_OK_ASYNC(
          context, DoMatrixTranspose(device, rhs, &transposed_rhs), done);
    } else {
      device.memcpy(transposed_rhs.flat<Scalar>().data(),
                    rhs.flat<Scalar>().data(),
                    rhs.NumElements() * sizeof(Scalar));
    }
#if GOOGLE_CUDA
    auto op_t = adjoint_ ? CUBLAS_OP_C : CUBLAS_OP_T;
    auto opbatch_t = op_t;
#else  // TENSORFLOW_USE_ROCM
    auto opbatch_t = adjoint_ ? rocblas_operation_conjugate_transpose
                              : rocblas_operation_transpose;
#if TF_ROCM_VERSION >= 40500
    auto op_t = adjoint_ ? HIPSOLVER_OP_C : HIPSOLVER_OP_T;
#else
    auto op_t = opbatch_t;
#endif
#endif

    // 3. Solve op(A) X = B (in column major form).
    // We use a trick here: If adjoint_ is true, we converted A to column major
    // form above. If adjoint is false then I leave A in row-major form and use
    // trans_a = CUBLAS_OP_T to effectively transform it to column-major on the
    // fly. (This means that we actually use the LU-factorization of A^T in that
    // case, but that is equally good for solving AX=B). This way we save an
    // explicit transpose in the more common case of adjoint_ == false.
    auto input_copy_ptr_array = solver->GetScratchSpace<uint8>(
        sizeof(Scalar*) * batch_size, "input_copy_ptr_array",
        /* on_host */ true);
    auto transposed_rhs_ptr_array = solver->GetScratchSpace<uint8>(
        sizeof(Scalar*) * batch_size, "transposed_rhs_ptr_array",
        /* on_host */ true);
    auto transposed_rhs_reshaped =
        transposed_rhs.template flat_inner_dims<Scalar, 3>();
    if (use_batched_solver) {
#if GOOGLE_CUDA
      const Scalar** input_copy_ptrs_base =
          reinterpret_cast<const Scalar**>(input_copy_ptr_array.mutable_data());
      const Scalar** transposed_rhs_ptrs_base =
          reinterpret_cast<const Scalar**>(
              transposed_rhs_ptr_array.mutable_data());
#else  // TENSORFLOW_USE_ROCM
      Scalar** input_copy_ptrs_base =
          reinterpret_cast<Scalar**>(input_copy_ptr_array.mutable_data());
      Scalar** transposed_rhs_ptrs_base =
          reinterpret_cast<Scalar**>(transposed_rhs_ptr_array.mutable_data());
#endif
      for (int batch = 0; batch < batch_size; ++batch) {
        input_copy_ptrs_base[batch] = &input_copy_reshaped(batch, 0, 0);
        transposed_rhs_ptrs_base[batch] = &transposed_rhs_reshaped(batch, 0, 0);
      }
      int host_info = 0;
      OP_REQUIRES_OK_ASYNC(
          context,
          solver->GetrsBatched(opbatch_t, n, nrhs, input_copy_ptrs_base, n,
                               pivots_mat.data(), transposed_rhs_ptrs_base, n,
                               &host_info, batch_size),
          done);
      OP_REQUIRES_ASYNC(
          context, host_info == 0,
          errors::InvalidArgument("The ", -host_info,
                                  "'th argument to cublas*getrsBatched had "
                                  "an illegal value."),
          done);
    } else {
      dev_info.push_back(solver->GetDeviceLapackInfo(batch_size, "getrs"));
      for (int batch = 0; batch < batch_size; ++batch) {
        OP_REQUIRES_OK_ASYNC(
            context,
            solver->Getrs(op_t, n, nrhs, &input_copy_reshaped(batch, 0, 0), n,
                          &pivots_mat(batch, 0),
                          &transposed_rhs_reshaped(batch, 0, 0), n,
                          &dev_info.back()(batch)),
            done);
      }
    }

    // 4. Transpose X to get the final result in row-major form.
    if (nrhs > 1) {
      OP_REQUIRES_OK_ASYNC(
          context, DoMatrixTranspose(device, transposed_rhs, output), done);
    } else {
      device.memcpy(output->flat<Scalar>().data(),
                    transposed_rhs.flat<Scalar>().data(),
                    transposed_rhs.NumElements() * sizeof(Scalar));
    }

    // Callback for checking info after kernels finish. Also capture the
    // temporary Tensors/ScratchSpace so they don't get deallocated before the
    // kernels run. TODO(rmlarsen): Use move capture once C++14 becomes
    // available.
    auto info_checker = [context, done, dev_info](
                            const Status& status,
                            const std::vector<HostLapackInfo>& host_infos) {
      if (!status.ok() && errors::IsInvalidArgument(status) &&
          !host_infos.empty()) {
        for (int i = 0; i < host_infos[0].size(); ++i) {
          // Match the CPU error message for singular matrices. Otherwise
          // just print the original error message from the status below.
          OP_REQUIRES_ASYNC(context, host_infos[0].data()[i] <= 0,
                            errors::InvalidArgument(kErrMsg), done);
        }
      }
      OP_REQUIRES_OK_ASYNC(context, status, done);
      done();
    };
    GpuSolver::CheckLapackInfoAndDeleteSolverAsync(std::move(solver), dev_info,
                                                   std::move(info_checker));
  }

 private:
  bool adjoint_;
};

REGISTER_LINALG_OP_GPU("MatrixSolve", (MatrixSolveOpGpu<float>), float);
REGISTER_LINALG_OP_GPU("MatrixSolve", (MatrixSolveOpGpu<double>), double);
REGISTER_LINALG_OP_GPU("MatrixSolve", (MatrixSolveOpGpu<complex64>), complex64);
REGISTER_LINALG_OP_GPU("MatrixSolve", (MatrixSolveOpGpu<complex128>),
                       complex128);

#endif  // GOOGLE_CUDA

REGISTER_LINALG_OP("MatrixSolve", (MatrixSolveOp<float>), float);
REGISTER_LINALG_OP("MatrixSolve", (MatrixSolveOp<double>), double);
REGISTER_LINALG_OP("MatrixSolve", (MatrixSolveOp<complex64>), complex64);
REGISTER_LINALG_OP("MatrixSolve", (MatrixSolveOp<complex128>), complex128);
REGISTER_LINALG_OP("BatchMatrixSolve", (MatrixSolveOp<float>), float);
REGISTER_LINALG_OP("BatchMatrixSolve", (MatrixSolveOp<double>), double);
REGISTER_LINALG_OP("BatchMatrixSolve", (MatrixSolveOp<complex64>), complex64);
REGISTER_LINALG_OP("BatchMatrixSolve", (MatrixSolveOp<complex128>), complex128);
}  // namespace tensorflow
