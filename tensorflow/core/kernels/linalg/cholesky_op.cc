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

#include "third_party/eigen3/Eigen/Cholesky"
#include "third_party/eigen3/Eigen/Core"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

<<<<<<< HEAD
#if GOOGLE_CUDA
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/linalg/matrix_band_part_op.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/gpu_solvers.h"
#endif

=======
>>>>>>> google_upstream/master
namespace tensorflow {

template <class Scalar>
class CholeskyOp : public LinearAlgebraOp<Scalar> {
 public:
  INHERIT_LINALG_TYPEDEFS(Scalar);

  explicit CholeskyOp(OpKernelConstruction* context) : Base(context) {}

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
    const ConstMatrixMap& input = inputs[0];
    if (input.rows() == 0) {
      // If X is an empty matrix (0 rows, 0 col), X * X' == X.
      // Therefore, we return X.
      return;
    }
    // Perform the actual LL^T Cholesky decomposition. This will only use
    // the lower triangular part of data_in by default. The upper triangular
    // part of the matrix will not be read.
    Eigen::LLT<
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        llt_decomposition(input);

<<<<<<< HEAD
    OP_REQUIRES(context, llt_decomposition.info() == Eigen::Success,
                errors::InvalidArgument(kErrMsg));

    // Output the lower triangular in a dense form.
    outputs->at(0) = llt_decomposition.matrixL();
  }
};

#if GOOGLE_CUDA
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
#define DECLARE_GPU_SPEC(T)                                            \
  template <>                                                          \
  struct MatrixBandPartFunctor<GPUDevice, T> {                         \
    void operator()(OpKernelContext* context, const GPUDevice& device, \
                    int num_upper_diags, int num_lower_diags,          \
                    typename TTypes<T, 3>::ConstTensor input,          \
                    typename TTypes<T, 3>::Tensor output);             \
  };                                                                   \
  extern template struct MatrixBandPartFunctor<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
TF_CALL_COMPLEX_TYPES(DECLARE_GPU_SPEC);
}  // namespace functor

template <class Scalar>
class CholeskyOpGpu : public AsyncOpKernel {
 public:
  explicit CholeskyOpGpu(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) final {
    const Tensor& input = context->input(0);
    const int ndims = input.dims();
    const int64_t n = input.dim_size(ndims - 1);
    // Validate inputs.
    OP_REQUIRES_ASYNC(
        context, ndims >= 2,
        errors::InvalidArgument("Input must have rank >= 2, got ", ndims),
        done);
    OP_REQUIRES_ASYNC(
        context, input.dim_size(ndims - 2) == n,
        errors::InvalidArgument("Input matrices must be squares, got",
                                input.dim_size(ndims - 2), " != ", n),
        done);

    if (input.NumElements() == 0) {
      // If X is an empty matrix (0 rows, 0 col), X * X' == X.
      // Therefore, we return X.
      context->set_output(0, input);
      done();
      return;
    }

    // Allocate output.
    // TODO(rmlarsen): Convert to std::make_unique when available.
    std::unique_ptr<GpuSolver> solver(new GpuSolver(context));
    Tensor* output;
    OP_REQUIRES_OK_ASYNC(context,
                         context->forward_input_or_allocate_output(
                             {0}, 0, input.shape(), &output),
                         done);

    // Copy the lower triangular part of the input matrices to the output and
    // set the strictly upper triangular part to zero. We use a pre-existing
    // kernel MatrixBandPart to do this for all matrices in the batch at once,
    // before we launch each of the Cholesky factorization kernels.
    auto input_reshaped = input.template flat_inner_dims<Scalar, 3>();
    auto output_reshaped = output->template flat_inner_dims<Scalar, 3>();
    functor::MatrixBandPartFunctor<GPUDevice, Scalar> band_part;
    band_part(context, context->eigen_device<GPUDevice>(),
              n /* num_lower_diags */, 0 /* num_upper_diags */, input_reshaped,
              output_reshaped);

    // Launch a Cholesky kernel for each matrix in the batch.
    const int64_t batch_size = input_reshaped.dimension(0);
    std::vector<DeviceLapackInfo> dev_info;

#if CUDA_VERSION >= 9020
    // Decide whether to use the batched API.
    // TODO(rmlarsen): The value 128 was found to be optimal for the equivalent
    // split in matrix_solve_op. Tune this heuristic.
    constexpr int kMaxMatrixSizeToBatchSizeRatio = 128;
    const bool use_batched_solver =
        n <= kMaxMatrixSizeToBatchSizeRatio * batch_size;
    if (use_batched_solver) {
      // For small matrices or large batch sizes, we use the batched interface
      // from cuSolver.
      auto output_reshaped_ptrs = solver->GetScratchSpace<uint8>(
          sizeof(Scalar*) * batch_size, "input_copt_ptrs",
          /* on_host */ true);
      const Scalar** output_reshaped_ptrs_base =
          reinterpret_cast<const Scalar**>(output_reshaped_ptrs.mutable_data());
      for (int batch = 0; batch < batch_size; ++batch) {
        output_reshaped_ptrs_base[batch] = &output_reshaped(batch, 0, 0);
      }
      dev_info.push_back(
          solver->GetDeviceLapackInfo(batch_size, "potrfBatched"));
      OP_REQUIRES_OK_ASYNC(context,
                           solver->PotrfBatched(CUBLAS_FILL_MODE_UPPER, n,
                                                output_reshaped_ptrs_base, n,
                                                &dev_info.back(), batch_size),
                           done);
      // TODO(rmlarsen): We have to clear the upper triangle of the output
      // due to a bug in potrfBatched. Remove this workaround once the bug
      // is fixed.
      auto input_reshaped = const_cast<const Tensor*>(output)
                                ->template flat_inner_dims<Scalar, 3>();
      auto output_reshaped = output->template flat_inner_dims<Scalar, 3>();
      functor::MatrixBandPartFunctor<GPUDevice, Scalar> band_part;
      band_part(context, context->eigen_device<GPUDevice>(),
                n /* num_lower_diags */, 0 /* num_upper_diags */,
                input_reshaped, output_reshaped);
=======
    // If decomposition fails, fill output with NaNs so the failure can
    // be detected at runtime.
    if (!TF_PREDICT_TRUE(llt_decomposition.info() == Eigen::Success)) {
      LOG(WARNING) << "Cholesky decomposition was not successful. "
                      "Eigen::LLT failed with error code "
                   << llt_decomposition.info()
                   << ". Filling lower-triangular output with NaNs.";
      outputs->at(0).template triangularView<Eigen::Lower>().fill(
          Eigen::NumTraits<Scalar>::quiet_NaN());
>>>>>>> google_upstream/master
    } else {
      // Output the lower triangular in a dense form.
      outputs->at(0) = llt_decomposition.matrixL();
    }
<<<<<<< HEAD
#endif

    // Register callback to check info after kernels finish.
    auto info_checker = [context, done](
                            const Status& status,
                            const std::vector<HostLapackInfo>& /* unused */) {
      OP_REQUIRES_ASYNC(context, status.ok(), errors::InvalidArgument(kErrMsg),
                        done);
      done();
    };
    GpuSolver::CheckLapackInfoAndDeleteSolverAsync(std::move(solver), dev_info,
                                                    std::move(info_checker));
=======
>>>>>>> google_upstream/master
  }
};

REGISTER_LINALG_OP("Cholesky", (CholeskyOp<float>), float);
REGISTER_LINALG_OP("Cholesky", (CholeskyOp<double>), double);
REGISTER_LINALG_OP("Cholesky", (CholeskyOp<complex64>), complex64);
REGISTER_LINALG_OP("Cholesky", (CholeskyOp<complex128>), complex128);
REGISTER_LINALG_OP("BatchCholesky", (CholeskyOp<float>), float);
REGISTER_LINALG_OP("BatchCholesky", (CholeskyOp<double>), double);

}  // namespace tensorflow
