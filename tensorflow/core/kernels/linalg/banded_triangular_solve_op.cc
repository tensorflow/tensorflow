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

#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/matmul_bcast.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Scalar>
Scalar eigen_conj(const Scalar& scalar) {
  return Eigen::numext::conj<Scalar>(scalar);
}

// Sequential batch matrix triangular solve kernel that calls Eigen's
// matrix triangular solve.
template <typename Scalar>
struct SequentialBandedTriangularSolveKernel {
  using Matrix =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ConstMatrixMap = Eigen::Map<const Matrix>;
  using MatrixMap = Eigen::Map<Matrix>;
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

  static ConstMatrixMap ConstTensorSliceToEigenMatrix(const Tensor& t,
                                                      int slice) {
    return ConstMatrixMap(
        t.flat<Scalar>().data() + slice * t.dim_size(1) * t.dim_size(2),
        t.dim_size(1), t.dim_size(2));
  }

  static MatrixMap TensorSliceToEigenMatrix(Tensor* t, int slice) {
    return MatrixMap(
        t->flat<Scalar>().data() + slice * t->dim_size(1) * t->dim_size(2),
        t->dim_size(1), t->dim_size(2));
  }

  static void Run(const Tensor& in_x, const Tensor& in_y, bool lower,
                  bool adjoint, const MatMulBCast& bcast, Tensor* out,
                  int start, int limit) {
    const bool should_bcast = bcast.IsBroadcastingRequired();
    const auto& x_batch_indices = bcast.x_batch_indices();
    const auto& y_batch_indices = bcast.y_batch_indices();
    int num_bands = in_x.dim_size(1);
    int matrix_size = in_x.dim_size(2);

    for (int64_t i = start; i < limit; ++i) {
      const int64_t x_batch_index = should_bcast ? x_batch_indices[i] : i;
      const int64_t y_batch_index = should_bcast ? y_batch_indices[i] : i;
      auto matrix = ConstTensorSliceToEigenMatrix(in_x, x_batch_index);
      auto rhs = ConstTensorSliceToEigenMatrix(in_y, y_batch_index);
      auto output = TensorSliceToEigenMatrix(out, i);
      // Below, we use the standard algorithm for computing a triangular solve,
      // except we band limit it.
      // Given A x = b, where A is lower triangular,
      // x_i = (b_i - sum a_ij * x_j) / a_ii, where the sum is from
      // j = 0 to i - 1.
      //
      // Now, in a banded triangular matrix, when i exceeds the band size,
      // then the sum goes from j = i - band_size to i - 1, since the other
      // elements are zero.
      //
      // Finally, given the band storage format, we'll need to change the
      // indexing.
      if (lower) {
        if (!adjoint) {
          output.row(0) = rhs.row(0) / matrix(0, 0);
          for (int i = 1; i < matrix_size; ++i) {
            if (i < num_bands) {
              output.row(i).noalias() =
                  (rhs.row(i) - matrix.block(1, i, i, 1).reverse().transpose() *
                                    output.topRows(i)) /
                  matrix(0, i);
            } else {
              output.row(i).noalias() =
                  (rhs.row(i) -
                   matrix.block(1, i, num_bands - 1, 1).reverse().transpose() *
                       output.middleRows(i - (num_bands - 1), num_bands - 1)) /
                  matrix(0, i);
            }
          }
        } else {
          // In the adjoint case, here and below, we now have an upper (lower)
          // triangular matrix, and thus need to work through with the other
          // case. We can't simply conjugate `matrix` and use the upper (lower)
          // algorithm because the band storage format for upper and lower
          // triangular matrices are different (in the lower case, we pad
          // entries on the left, and in the upper case we pad entries on the
          // right.
          output.row(matrix_size - 1) =
              rhs.row(matrix_size - 1) / eigen_conj(matrix(0, matrix_size - 1));
          for (int i = matrix_size - 1; i >= 0; --i) {
            output.row(i).noalias() = rhs.row(i);
            for (int j = i + 1; j < std::min(matrix_size, i + num_bands); ++j) {
              output.row(i).noalias() -=
                  eigen_conj(matrix(j - i, j)) * output.row(j);
            }
            output.row(i) /= eigen_conj(matrix(0, i));
          }
        }
      } else {
        if (!adjoint) {
          output.row(matrix_size - 1) =
              rhs.row(matrix_size - 1) / matrix(num_bands - 1, matrix_size - 1);
          for (int i = 1; i < matrix_size; ++i) {
            int k = matrix_size - 1 - i;
            if (i < num_bands) {
              output.row(k).noalias() =
                  (rhs.row(k) - matrix.block(num_bands - 1 - i, k, i, 1)
                                        .reverse()
                                        .transpose() *
                                    output.bottomRows(i)) /
                  matrix(num_bands - 1, k);
            } else {
              output.row(k).noalias() =
                  (rhs.row(k) -
                   matrix.block(0, k, num_bands - 1, 1).reverse().transpose() *
                       output.middleRows(k + 1, num_bands - 1)) /
                  matrix(num_bands - 1, k);
            }
          }
        } else {
          output.row(0) = rhs.row(0) / eigen_conj(matrix(num_bands - 1, 0));
          for (int i = 1; i < matrix_size; ++i) {
            output.row(i).noalias() = rhs.row(i);
            for (int j = std::max(0, i - (num_bands - 1)); j < i; ++j) {
              output.row(i).noalias() -=
                  eigen_conj(matrix(num_bands - 1 - (i - j), j)) *
                  output.row(j);
            }
            output.row(i) /= eigen_conj(matrix(num_bands - 1, i));
          }
        }
      }
    }
  }
};

template <typename Scalar>
struct LaunchBatchBandedTriangularSolve;

template <typename Scalar>
struct LaunchBatchBandedTriangularSolve {
  static void Launch(OpKernelContext* context, const Tensor& in_x,
                     const Tensor& in_y, bool adjoint, bool lower,
                     const MatMulBCast& bcast, Tensor* out) {
    // Number of banded matrix triangular solves i.e. size of the batch.
    const int64_t batch_size = bcast.output_batch_size();
    const int64_t cost_per_unit =
        in_x.dim_size(1) * in_x.dim_size(2) * in_y.dim_size(2);
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());

    using Matrix =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using ConstMatrixMap = Eigen::Map<const Matrix>;
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    // Check diagonal before doing any solves. This is the first row in the
    // lower case and else is the last row.
    auto matrix = ConstMatrixMap(in_x.flat<Scalar>().data(), in_x.dim_size(1),
                                 in_x.dim_size(2));
    RealScalar min_abs_pivot;
    if (lower) {
      min_abs_pivot = matrix.row(0).cwiseAbs().minCoeff();
    } else {
      min_abs_pivot = matrix.row(in_x.dim_size(1) - 1).cwiseAbs().minCoeff();
    }
    OP_REQUIRES(context, min_abs_pivot > RealScalar(0),
                errors::InvalidArgument("Input matrix is not invertible."));

    Shard(worker_threads.num_threads, worker_threads.workers, batch_size,
          cost_per_unit,
          [&in_x, &in_y, adjoint, lower, &bcast, out](int64_t start,
                                                      int64_t limit) {
            SequentialBandedTriangularSolveKernel<Scalar>::Run(
                in_x, in_y, lower, adjoint, bcast, out, start, limit);
          });
  }
};

template <typename Scalar>
class BandedTriangularSolveOpCpu : public OpKernel {
 public:
  explicit BandedTriangularSolveOpCpu(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("lower", &lower_));
    OP_REQUIRES_OK(context, context->GetAttr("adjoint", &adjoint_));
  }

  ~BandedTriangularSolveOpCpu() override {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    ValidateInputTensors(ctx, in0, in1);
    if (!ctx->status().ok()) return;

    MatMulBCast bcast(in0.shape().dim_sizes(), in1.shape().dim_sizes());
    OP_REQUIRES(
        ctx, bcast.IsValid(),
        errors::InvalidArgument(
            "In[0] and In[1] must have compatible batch dimensions: ",
            in0.shape().DebugString(), " vs. ", in1.shape().DebugString()));

    TensorShape out_shape = bcast.output_batch_shape();
    auto batch_size = bcast.output_batch_size();
    auto d0 = in0.dim_size(in0.dims() - 2);  // Band size.
    auto d1 = in0.dim_size(in0.dims() - 1);
    Tensor in0_reshaped;
    OP_REQUIRES(
        ctx,
        in0_reshaped.CopyFrom(in0, TensorShape({bcast.x_batch_size(), d0, d1})),
        errors::Internal("Failed to reshape In[0] from ",
                         in0.shape().DebugString()));
    auto d2 = in1.dim_size(in1.dims() - 2);
    auto d3 = in1.dim_size(in1.dims() - 1);
    Tensor in1_reshaped;
    OP_REQUIRES(
        ctx,
        in1_reshaped.CopyFrom(in1, TensorShape({bcast.y_batch_size(), d2, d3})),
        errors::Internal("Failed to reshape In[1] from ",
                         in1.shape().DebugString()));
    OP_REQUIRES(ctx, d1 == d2,
                errors::InvalidArgument(
                    "In[0] mismatch In[1] shape: ", d1, " vs. ", d2, ": ",
                    in0.shape().DebugString(), " ", in1.shape().DebugString(),
                    " ", lower_, " ", adjoint_));
    out_shape.AddDim(d1);
    out_shape.AddDim(d3);
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) {
      return;
    }
    Tensor out_reshaped;
    OP_REQUIRES(ctx,
                out_reshaped.CopyFrom(*out, TensorShape({batch_size, d1, d3})),
                errors::Internal("Failed to reshape output from ",
                                 out->shape().DebugString()));
    LaunchBatchBandedTriangularSolve<Scalar>::Launch(
        ctx, in0_reshaped, in1_reshaped, adjoint_, lower_, bcast,
        &out_reshaped);
  }

 private:
  void ValidateInputTensors(OpKernelContext* ctx, const Tensor& in0,
                            const Tensor& in1) {
    OP_REQUIRES(
        ctx, in0.dims() >= 2,
        errors::InvalidArgument("In[0] ndims must be >= 2: ", in0.dims()));

    OP_REQUIRES(
        ctx, in1.dims() >= 2,
        errors::InvalidArgument("In[1] ndims must be >= 2: ", in1.dims()));

    OP_REQUIRES(ctx, in0.NumElements() > 0,
                errors::InvalidArgument("In[0] must not be an empty tensor: ",
                                        in0.DebugString()));

    OP_REQUIRES(ctx, in1.NumElements() > 0,
                errors::InvalidArgument("In[1] must not be an empty tensor: ",
                                        in1.DebugString()));
  }
  bool lower_;
  bool adjoint_;
};

#define REGISTER_BANDED_TRIANGULAR_SOLVE_CPU(TYPE)        \
  REGISTER_KERNEL_BUILDER(Name("BandedTriangularSolve")   \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<TYPE>("T"), \
                          BandedTriangularSolveOpCpu<TYPE>);

REGISTER_BANDED_TRIANGULAR_SOLVE_CPU(float);
REGISTER_BANDED_TRIANGULAR_SOLVE_CPU(double);
REGISTER_BANDED_TRIANGULAR_SOLVE_CPU(complex64);
REGISTER_BANDED_TRIANGULAR_SOLVE_CPU(complex128);

}  // namespace tensorflow
