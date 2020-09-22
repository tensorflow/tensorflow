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

#include <cmath>

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/linalg/determinant_op.h"
#endif

#include "third_party/eigen3/Eigen/LU"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/util/cuda_solvers.h"
#endif

namespace tensorflow {

// A helper function to compute the sign and absolute value of the log of the
// determinant of inputs via a partially pivoted LU
// factorization.
//
// Returns the log of the absolute value of the determinant, and its sign in
// 'sign'.
template <class Scalar>
static typename Eigen::NumTraits<Scalar>::Real SLogDet(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& inputs,
    Scalar* sign) {
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
  RealScalar log_abs_det = 0;
  *sign = 1;
  // An empty matrix' determinant is defined to be 1.
  // (https://en.wikipedia.org/wiki/Determinant)
  if (inputs.size() > 0) {
    // Compute the log determinant through a Partially Pivoted LU decomposition
    using Eigen::Dynamic;
    Eigen::PartialPivLU<Eigen::Matrix<Scalar, Dynamic, Dynamic>> lu(inputs);
    Eigen::Matrix<Scalar, Dynamic, Dynamic> LU = lu.matrixLU();
    *sign = lu.permutationP().determinant();
    auto diag = LU.diagonal().array().eval();
    auto abs_diag = diag.cwiseAbs().eval();
    log_abs_det += abs_diag.log().sum();
    *sign *= (diag / abs_diag).prod();
  }
  if (!Eigen::numext::isfinite(log_abs_det)) {
    *sign = 0;
    log_abs_det =
        log_abs_det > 0 ? -std::log(RealScalar(0)) : std::log(RealScalar(0));
  }
  return log_abs_det;
}

template <class Scalar>
class LogDeterminantOp : public LinearAlgebraOp<Scalar> {
 public:
  INHERIT_LINALG_TYPEDEFS(Scalar);

  explicit LogDeterminantOp(OpKernelConstruction* context) : Base(context) {}

  TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shapes) const final {
    return TensorShapes({TensorShape({}), TensorShape({})});
  }

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
    Scalar sign;
    const RealScalar log_abs_det = SLogDet(
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(inputs[0]),
        &sign);

    outputs->at(0)(0, 0) = sign;
    outputs->at(1)(0, 0) = log_abs_det;
  }
};

template <class Scalar>
class DeterminantOp : public LinearAlgebraOp<Scalar> {
 public:
  INHERIT_LINALG_TYPEDEFS(Scalar);

  explicit DeterminantOp(OpKernelConstruction* context) : Base(context) {}

  TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shape) const final {
    return TensorShapes({TensorShape({})});
  }

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
    Scalar sign;
    const RealScalar log_abs_det = SLogDet(
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>(inputs[0]),
        &sign);
    outputs->at(0)(0, 0) = sign * std::exp(log_abs_det);
  }
};

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

template <class Scalar>
class DeterminantOpGpu : public AsyncOpKernel {
 public:
  explicit DeterminantOpGpu(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) final {
    const Tensor& input = context->input(0);
    const int ndims = input.dims();
    const int64 n = input.dim_size(ndims - 1);
    // Validate inputs.
    OP_REQUIRES_ASYNC(
        context, ndims >= 2,
        errors::InvalidArgument("Input must have rank >= 2, got ", ndims),
        done);
    OP_REQUIRES_ASYNC(
        context, input.dim_size(ndims - 2) == n,
        errors::InvalidArgument("Input matrices must be square, got",
                                input.dim_size(ndims - 2), " != ", n),
        done);

    // Allocate output.
    TensorShape out_shape;
    for (int dim = 0; dim < ndims - 2; ++dim) {
      out_shape.AddDim(input.dim_size(dim));
    }
    out_shape.AppendShape(TensorShape({}));
    Tensor* out;
    OP_REQUIRES_OK_ASYNC(context, context->allocate_output(0, out_shape, &out),
                         done);

    // By definition, the determinant of an empty matrix is equal to one.
    const GPUDevice& d = context->eigen_device<GPUDevice>();
    if (input.NumElements() == 0) {
      functor::SetOneFunctor<GPUDevice, Scalar> f;
      f(d, out->template flat<Scalar>());
      done();
      return;
    }

    // TODO(rmlarsen): Convert to absl::make_unique when available.
    std::unique_ptr<CudaSolver> solver(new CudaSolver(context));

    // Reuse the input buffer or make a copy for the factorization step,
    // depending on whether this ops owns it exclusively.
    Tensor input_copy;
    OP_REQUIRES_OK_ASYNC(
        context,
        solver->forward_input_or_allocate_scoped_tensor(
            {0}, DataTypeToEnum<Scalar>::value, input.shape(), &input_copy),
        done);
    if (!input.SharesBufferWith(input_copy)) {
      d.memcpy(input_copy.flat<Scalar>().data(), input.flat<Scalar>().data(),
               input.NumElements() * sizeof(Scalar));
    }
    auto input_copy_reshaped = input_copy.template flat_inner_dims<Scalar, 3>();
    const int64 batch_size = input_copy_reshaped.dimension(0);

    // Allocate pivots on the device.
    Tensor pivots;
    OP_REQUIRES_OK_ASYNC(
        context,
        solver->allocate_scoped_tensor(DataTypeToEnum<int>::value,
                                       TensorShape{batch_size, n}, &pivots),
        done);
    auto pivots_mat = pivots.template matrix<int>();

    // Prepare pointer arrays for cuBlas' batch interface.
    // TODO(rmlarsen): Find a way to encode pointer arrays in pinned host memory
    // without the ugly casting.
    auto input_copy_ptrs = solver->GetScratchSpace<uint8>(
        sizeof(Scalar*) * batch_size, "input_copy_ptrs",
        /* on_host */ true);
    auto output_reshaped = out->template flat_inner_dims<Scalar, 1>();

    // Compute the partially pivoted LU factorization(s) of the matrix/matrices.
    std::vector<DeviceLapackInfo> dev_info;
    if (n / batch_size <= 128) {
      // For small matrices or large batch sizes, we use the batched interface
      // from cuBlas.
      const Scalar** input_copy_ptrs_base =
          reinterpret_cast<const Scalar**>(input_copy_ptrs.mutable_data());
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
      // For small batch sizes we use the non-batched interface from cuSolver,
      // which is much faster for large matrices.
      dev_info.push_back(solver->GetDeviceLapackInfo(batch_size, "getrf"));
      for (int batch = 0; batch < batch_size; ++batch) {
        OP_REQUIRES_OK_ASYNC(
            context,
            solver->Getrf(n, n, &input_copy_reshaped(batch, 0, 0), n,
                          &pivots_mat(batch, 0), &dev_info.back()(batch)),
            done);
      }
    }

    // Compute the determinant for each batch as (-1)^s * prod(diag(U)),
    // where s is the order of the permutation encoded in pivots and U is the
    // upper triangular factor of the LU factorization, which is written to
    // input_copy by the Getrf{Batched} kernel.
    functor::DeterminantFromPivotedLUFunctor<GPUDevice, Scalar> functor;
    functor(d,
            const_cast<const Tensor*>(&input_copy)
                ->template flat_inner_dims<Scalar, 3>(),
            pivots_mat.data(), output_reshaped, dev_info.back().mutable_data());

    // Register callback to check info after kernels finish.
    auto info_checker = [context, done](
                            const Status& status,
                            const std::vector<HostLapackInfo>& host_infos) {
      if (!status.ok() && errors::IsInvalidArgument(status) &&
          !host_infos.empty()) {
        for (int i = 0; i < host_infos[0].size(); ++i) {
          // It is OK for a matrix to be singular (signaled by info > 0),
          // corresponding to determinant of zero, but we do want to catch
          // invalid arguments to Getrf{Batched}.
          OP_REQUIRES_ASYNC(
              context, host_infos[0](i) >= 0,
              errors::InvalidArgument("Invalid input argument no. ",
                                      host_infos[0].data()[i],
                                      " for batch index ", i, "."),
              done);
        }
      }
      done();
    };
    CudaSolver::CheckLapackInfoAndDeleteSolverAsync(std::move(solver), dev_info,
                                                    std::move(info_checker));
  }
};

template <class Scalar>
class LogDeterminantOpGpu : public AsyncOpKernel {
 public:
  explicit LogDeterminantOpGpu(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) final {
    const Tensor& input = context->input(0);
    const int ndims = input.dims();
    const int64 n = input.dim_size(ndims - 1);
    // Validate inputs.
    OP_REQUIRES_ASYNC(
        context, ndims >= 2,
        errors::InvalidArgument("Input must have rank >= 2, got ", ndims),
        done);
    OP_REQUIRES_ASYNC(
        context, input.dim_size(ndims - 2) == n,
        errors::InvalidArgument("Input matrices must be square, got",
                                input.dim_size(ndims - 2), " != ", n),
        done);

    // Allocate output.
    TensorShape out_shape;
    for (int dim = 0; dim < ndims - 2; ++dim) {
      out_shape.AddDim(input.dim_size(dim));
    }
    out_shape.AppendShape(TensorShape({}));
    Tensor* sign;
    OP_REQUIRES_OK_ASYNC(context, context->allocate_output(0, out_shape, &sign),
                         done);
    Tensor* log_abs_det;
    OP_REQUIRES_OK_ASYNC(
        context, context->allocate_output(1, out_shape, &log_abs_det), done);

    // By definition, the determinant of an empty matrix is equal to one.
    const GPUDevice& d = context->eigen_device<GPUDevice>();
    if (input.NumElements() == 0) {
      functor::SetOneFunctor<GPUDevice, Scalar> one_func;
      one_func(d, sign->template flat<Scalar>());
      functor::SetZeroFunctor<GPUDevice, Scalar> zero_func;
      zero_func(d, log_abs_det->template flat<Scalar>());
      done();
      return;
    }

    // TODO(rmlarsen): Convert to absl::make_unique when available.
    std::unique_ptr<CudaSolver> solver(new CudaSolver(context));

    // Reuse the input buffer or make a copy for the factorization step,
    // depending on whether this ops owns it exclusively.
    Tensor input_copy;
    OP_REQUIRES_OK_ASYNC(
        context,
        solver->forward_input_or_allocate_scoped_tensor(
            {0}, DataTypeToEnum<Scalar>::value, input.shape(), &input_copy),
        done);
    if (!input.SharesBufferWith(input_copy)) {
      d.memcpy(input_copy.flat<Scalar>().data(), input.flat<Scalar>().data(),
               input.NumElements() * sizeof(Scalar));
    }
    auto input_copy_reshaped = input_copy.template flat_inner_dims<Scalar, 3>();
    const int64 batch_size = input_copy_reshaped.dimension(0);

    // Allocate pivots on the device.
    Tensor pivots;
    OP_REQUIRES_OK_ASYNC(
        context,
        solver->allocate_scoped_tensor(DataTypeToEnum<int>::value,
                                       TensorShape{batch_size, n}, &pivots),
        done);
    auto pivots_mat = pivots.template matrix<int>();

    // Prepare pointer arrays for cuBlas' batch interface.
    // TODO(rmlarsen): Find a way to encode pointer arrays in pinned host memory
    // without the ugly casting.
    auto input_copy_ptrs = solver->GetScratchSpace<uint8>(
        sizeof(Scalar*) * batch_size, "input_copy_ptrs",
        /* on_host */ true);

    // Compute the partially pivoted LU factorization(s) of the matrix/matrices.
    std::vector<DeviceLapackInfo> dev_info;
    if (n / batch_size <= 128) {
      // For small matrices or large batch sizes, we use the batched interface
      // from cuBlas.
      const Scalar** input_copy_ptrs_base =
          reinterpret_cast<const Scalar**>(input_copy_ptrs.mutable_data());
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
      // For large matrices or small batch sizes we use the non-batched
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

    auto input_copy_reshaped_const =
        const_cast<const Tensor*>(&input_copy)
            ->template flat_inner_dims<Scalar, 3>();
    auto sign_reshaped = sign->flat<Scalar>();
    auto log_abs_det_reshaped = log_abs_det->flat<Scalar>();
    // Compute the determinant for each batch as (-1)^s * prod(diag(U)),
    // where s is the order of the permutation encoded in pivots and U is the
    // upper triangular factor of the LU factorization, which is written to
    // input_copy by the Getrf{Batched} kernel.
    functor::LogDeterminantFromPivotedLUFunctor<GPUDevice, Scalar> functor;
    functor(d, input_copy_reshaped_const, pivots_mat.data(), sign_reshaped,
            log_abs_det_reshaped);

    // Register callback to check info after kernels finish.
    auto info_checker = [context, done](
                            const Status& status,
                            const std::vector<HostLapackInfo>& host_infos) {
      if (!status.ok() && errors::IsInvalidArgument(status) &&
          !host_infos.empty()) {
        for (int i = 0; i < host_infos[0].size(); ++i) {
          // It is OK for a matrix to be singular (signaled by info > 0),
          // corresponding to determinant of zero, but we do want to catch
          // invalid arguments to Getrf{Batched}.
          OP_REQUIRES_ASYNC(
              context, host_infos[0](i) >= 0,
              errors::InvalidArgument("Invalid input argument no. ",
                                      host_infos[0].data()[i],
                                      " for batch index ", i, "."),
              done);
        }
      }
      done();
    };
    CudaSolver::CheckLapackInfoAndDeleteSolverAsync(std::move(solver), dev_info,
                                                    std::move(info_checker));
  }
};

REGISTER_LINALG_OP_GPU("MatrixDeterminant", (DeterminantOpGpu<float>), float);
REGISTER_LINALG_OP_GPU("MatrixDeterminant", (DeterminantOpGpu<double>), double);
REGISTER_LINALG_OP_GPU("MatrixDeterminant", (DeterminantOpGpu<complex64>),
                       complex64);
REGISTER_LINALG_OP_GPU("MatrixDeterminant", (DeterminantOpGpu<complex128>),
                       complex128);

REGISTER_LINALG_OP_GPU("LogMatrixDeterminant", (LogDeterminantOpGpu<float>),
                       float);
REGISTER_LINALG_OP_GPU("LogMatrixDeterminant", (LogDeterminantOpGpu<double>),
                       double);
REGISTER_LINALG_OP_GPU("LogMatrixDeterminant", (LogDeterminantOpGpu<complex64>),
                       complex64);
REGISTER_LINALG_OP_GPU("LogMatrixDeterminant",
                       (LogDeterminantOpGpu<complex128>), complex128);
#endif  // GOOGLE_CUDA

REGISTER_LINALG_OP("MatrixDeterminant", (DeterminantOp<float>), float);
REGISTER_LINALG_OP("MatrixDeterminant", (DeterminantOp<double>), double);
REGISTER_LINALG_OP("MatrixDeterminant", (DeterminantOp<complex64>), complex64);
REGISTER_LINALG_OP("MatrixDeterminant", (DeterminantOp<complex128>),
                   complex128);
REGISTER_LINALG_OP("BatchMatrixDeterminant", (DeterminantOp<float>), float);
REGISTER_LINALG_OP("BatchMatrixDeterminant", (DeterminantOp<double>), double);
REGISTER_LINALG_OP("BatchMatrixDeterminant", (DeterminantOp<complex64>),
                   complex64);
REGISTER_LINALG_OP("BatchMatrixDeterminant", (DeterminantOp<complex128>),
                   complex128);

REGISTER_LINALG_OP("LogMatrixDeterminant", (LogDeterminantOp<float>), float);
REGISTER_LINALG_OP("LogMatrixDeterminant", (LogDeterminantOp<double>), double);
REGISTER_LINALG_OP("LogMatrixDeterminant", (LogDeterminantOp<complex64>),
                   complex64);
REGISTER_LINALG_OP("LogMatrixDeterminant", (LogDeterminantOp<complex128>),
                   complex128);
}  // namespace tensorflow
