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

#ifndef TENSORFLOW_CORE_KERNELS_LINALG_QR_OP_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_LINALG_QR_OP_IMPL_H_

// See docs in ../ops/linalg_ops.cc.
//
// This header file is used by the individual qr_*op*.cc files for registering
// individual kernels. A separate file is used for each instantiated kernel to
// improve compilation times.
#include <algorithm>
#include <numeric>

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include "third_party/eigen3/Eigen/QR"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

#if GOOGLE_CUDA
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/kernels/linalg/eye_functor.h"
#include "tensorflow/core/kernels/linalg/matrix_band_part_op.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/util/gpu_solvers.h"
#endif

namespace tensorflow {

template <class Scalar>
class QrOp : public LinearAlgebraOp<Scalar> {
 public:
  typedef LinearAlgebraOp<Scalar> Base;

  explicit QrOp(OpKernelConstruction* context) : Base(context) {
    OP_REQUIRES_OK(context, context->GetAttr("full_matrices", &full_matrices_));
  }

  using TensorShapes = typename Base::TensorShapes;

  void ValidateInputMatrixShapes(
      OpKernelContext* context,
      const TensorShapes& input_matrix_shapes) const final {
    Base::ValidateSingleMatrix(context, input_matrix_shapes);
  }

  TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shapes) const final {
    int64_t m = input_matrix_shapes[0].dim_size(0);
    int64_t n = input_matrix_shapes[0].dim_size(1);
    int64_t min_size = std::min(m, n);
    if (full_matrices_) {
      return TensorShapes({TensorShape({m, m}), TensorShape({m, n})});
    } else {
      return TensorShapes(
          {TensorShape({m, min_size}), TensorShape({min_size, n})});
    }
  }

  int64_t GetCostPerUnit(const TensorShapes& input_matrix_shapes) const final {
    double m = static_cast<double>(input_matrix_shapes[0].dim_size(0));
    double n = static_cast<double>(input_matrix_shapes[0].dim_size(1));
    double max_size = std::max(m, n);
    double min_size = std::min(m, n);
    double cost = 2 * max_size * min_size * min_size -
                  2 * min_size * min_size * min_size / 3.;
    // TODO(jpoulson): Increase the cost if full_matrices is true in a manner
    // that reflects the algorithm used for the expansion.
    return cost >= static_cast<double>(kint64max) ? kint64max
                                                  : static_cast<int64_t>(cost);
  }

  using Matrix = typename Base::Matrix;
  using MatrixMaps = typename Base::MatrixMaps;
  using ConstMatrixMap = typename Base::ConstMatrixMap;
  using ConstMatrixMaps = typename Base::ConstMatrixMaps;

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
    Eigen::HouseholderQR<Matrix> qr(inputs[0]);
    const int m = inputs[0].rows();
    const int n = inputs[0].cols();
    const int min_size = std::min(m, n);

    if (full_matrices_) {
      outputs->at(0) = qr.householderQ();
      outputs->at(1) = qr.matrixQR().template triangularView<Eigen::Upper>();
    } else {
      // TODO(jpoulson): Exploit the fact that Householder transformations can
      // be expanded faster than they can be applied to an arbitrary matrix
      // (Cf. LAPACK's DORGQR).
      Matrix tmp = Matrix::Identity(m, min_size);
      outputs->at(0) = qr.householderQ() * tmp;
      auto qr_top = qr.matrixQR().block(0, 0, min_size, n);
      outputs->at(1) = qr_top.template triangularView<Eigen::Upper>();
    }
  }

 private:
  bool full_matrices_;

  TF_DISALLOW_COPY_AND_ASSIGN(QrOp);
};

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

template <class Scalar>
class QrOpGpu : public AsyncOpKernel {
 public:
  explicit QrOpGpu(OpKernelConstruction* context) : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("full_matrices", &full_matrices_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) final {
    const Tensor& input = context->input(0);
    const int ndims = input.dims();
    const int64_t m = input.dim_size(ndims - 2);
    const int64_t n = input.dim_size(ndims - 1);
    const int64_t min_size = std::min(m, n);
    const int64_t batch_size =
        input.template flat_inner_dims<Scalar, 3>().dimension(0);

    // Validate inputs.
    OP_REQUIRES_ASYNC(
        context, ndims >= 2,
        errors::InvalidArgument("Input must have rank >= 2, got ", ndims),
        done);

    // Allocate output.
    // If full_matrices_ is true then Q is m x m and R is m x n.
    // Otherwise, Q is m x min(m, n), and R is min(m, n) x n.
    Tensor* q;
    TensorShape q_shape = input.shape();
    q_shape.set_dim(ndims - 1, full_matrices_ ? m : min_size);
    OP_REQUIRES_OK_ASYNC(context, context->allocate_output(0, q_shape, &q),
                         done);
    Tensor* r;
    TensorShape r_shape = input.shape();
    r_shape.set_dim(ndims - 2, full_matrices_ ? m : min_size);
    OP_REQUIRES_OK_ASYNC(context, context->allocate_output(1, r_shape, &r),
                         done);

    if (input.NumElements() == 0) {
      done();
      return;
    }

    // TODO(rmlarsen): Convert to std::make_unique when available.
    std::unique_ptr<GpuSolver> solver(new GpuSolver(context));

    // Allocate temporaries.
    Tensor input_transposed;
    TensorShape transposed_shape = input.shape();
    transposed_shape.set_dim(ndims - 2, input.dim_size(ndims - 1));
    transposed_shape.set_dim(ndims - 1, input.dim_size(ndims - 2));

    OP_REQUIRES_OK_ASYNC(
        context,
        solver->allocate_scoped_tensor(DataTypeToEnum<Scalar>::value,
                                       transposed_shape, &input_transposed),
        done);

    Tensor tau;
    OP_REQUIRES_OK_ASYNC(context,
                         solver->allocate_scoped_tensor(
                             DataTypeToEnum<Scalar>::value,
                             TensorShape({batch_size, min_size}), &tau),
                         done);

    // Transpose input, since cuSolver uses column-major, while TensorFlow uses
    // row-major storage.
    const GPUDevice& device = context->eigen_device<GPUDevice>();
    OP_REQUIRES_OK_ASYNC(
        context, DoMatrixTranspose(device, input, &input_transposed), done);

    // Compute QR decomposition in-place in input_transposed.
    std::vector<DeviceLapackInfo> dev_info;
    dev_info.push_back(solver->GetDeviceLapackInfo(batch_size, "geqrf"));
    auto input_transposed_reshaped =
        input_transposed.flat_inner_dims<Scalar, 3>();
    auto tau_matrix = tau.matrix<Scalar>();
    auto r_reshaped = r->flat_inner_dims<Scalar, 3>();
    for (int batch = 0; batch < batch_size; ++batch) {
      OP_REQUIRES_OK_ASYNC(
          context,
          solver->Geqrf(m, n, &input_transposed_reshaped(batch, 0, 0), m,
                        &tau_matrix(batch, 0),
                        dev_info.back().mutable_data() + batch),
          done);
    }

    // Generate R. R is equal to the upper triangle of the decomposition
    // stored in input_transposed. Crop, transpose (to get back to row-major)
    // and copy it to the output buffer.
    if (full_matrices_ || m == n) {
      OP_REQUIRES_OK_ASYNC(
          context, DoMatrixTranspose(device, input_transposed, r), done);
    } else {
      const Scalar alpha(1);
      const Scalar beta(0);
      const Scalar* dummy = nullptr;
      for (int batch = 0; batch < batch_size; ++batch) {
        OP_REQUIRES_OK_ASYNC(
            context,
            solver->Geam(CUBLAS_OP_T, CUBLAS_OP_N, n,
                         full_matrices_ ? m : min_size, &alpha,
                         &input_transposed_reshaped(batch, 0, 0), m, &beta,
                         dummy, n, &r_reshaped(batch, 0, 0), n),
            done);
      }
    }
    // Extract the upper triangle of r (i.e. zero out the strictly lower
    // triangle).
    functor::MatrixBandPartFunctor<GPUDevice, Scalar> band_part;
    auto r_reshaped_const =
        const_cast<const Tensor*>(r)->flat_inner_dims<Scalar, 3>();
    band_part(context, device, 0 /* num_lower_diags */,
              -1 /* num_upper_diags */, r_reshaped_const, r_reshaped);

    // Generate Q from the decomposition in input_transposed.
    if (m != n && (full_matrices_ || m < n)) {
      // Generate full m x m matrix Q by computing the product Q^T * I,
      // where the transpose is to get back to row-major form.
      // In the complex case we actually form Q^H * I and conjugate it
      // to get Q in row-major form.
      functor::EyeFunctor<GPUDevice, Scalar> eye;
      auto q_reshaped = q->flat_inner_dims<Scalar, 3>();
      eye(device, q_reshaped);
      for (int batch = 0; batch < batch_size; ++batch) {
        // Notice: It appears that Unmqr does not write a zero into *info upon
        // success (probably a bug), so we simply re-use the info array already
        // zeroed by Geqrf above.
        OP_REQUIRES_OK_ASYNC(
            context,
            solver->Unmqr(CUBLAS_SIDE_LEFT, CublasAdjointOp<Scalar>(), m, m,
                          min_size, &input_transposed_reshaped(batch, 0, 0), m,
                          &tau_matrix(batch, 0), &q_reshaped(batch, 0, 0), m,
                          dev_info.back().mutable_data() + batch),
            done);
      }
      if (Eigen::NumTraits<Scalar>::IsComplex) {
        functor::UnaryFunctor<GPUDevice, functor::conj<Scalar>> conj;
        conj(device, q->flat<Scalar>() /*out*/,
             const_cast<const Tensor*>(q)->flat<Scalar>() /*in*/);
      }
    } else {
      // Generate m x n matrix Q. In this case we can use the more efficient
      // algorithm in Ungqr to generate Q in place.
      dev_info.push_back(solver->GetDeviceLapackInfo(batch_size, "orgqr"));
      for (int batch = 0; batch < batch_size; ++batch) {
        OP_REQUIRES_OK_ASYNC(
            context,
            solver->Ungqr(
                m, n, min_size, &input_transposed_reshaped(batch, 0, 0), m,
                &tau_matrix(batch, 0), dev_info.back().mutable_data() + batch),
            done);
      }
      OP_REQUIRES_OK_ASYNC(
          context, DoMatrixTranspose(device, input_transposed, q), done);
    }

    // Asynchronously check return status from cuSolver kernels.
    GpuSolver::CheckLapackInfoAndDeleteSolverAsync(std::move(solver), dev_info,
                                                   std::move(done));
  }

 private:
  bool full_matrices_;

  TF_DISALLOW_COPY_AND_ASSIGN(QrOpGpu);
};

#endif  // GOOGLE_CUDA

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_LINALG_QR_OP_IMPL_H_
