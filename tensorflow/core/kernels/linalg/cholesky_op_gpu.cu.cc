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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"
#include "tensorflow/core/kernels/linalg/matrix_band_part_op.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_solvers.h"

namespace tensorflow {

namespace functor {
namespace {

template <typename Scalar>
__global__ void MatrixBandFillKernel(const int num_threads,
                                     const int batch_size, const int m,
                                     const int n, const int num_lower_diags,
                                     const int num_upper_diags,
                                     const Scalar value,
                                     Scalar* __restrict__ output_ptr) {
  GPU_1D_KERNEL_LOOP(index, num_threads) {
    const int col = index % n;
    const int row = (index / n) % m;
    const int band_start = (num_lower_diags < 0 ? 0 : row - num_lower_diags);
    const int band_end = (num_upper_diags < 0 ? n : row + num_upper_diags + 1);
    if (col < band_start || col >= band_end) {
      output_ptr[index] = Scalar(0);
    } else {
      output_ptr[index] = value;
    }
  }
}

}  // namespace

// Fills a banded matrix with a constant value.
template <typename Device, typename Scalar>
struct MatrixBandFillFunctor;

typedef Eigen::GpuDevice GPUDevice;

template <typename Scalar>
struct MatrixBandFillFunctor<GPUDevice, Scalar> {
  void operator()(OpKernelContext* context, const GPUDevice& device,
                  int num_lower_diags, int num_upper_diags, const Scalar& value,
                  typename TTypes<Scalar, 3>::Tensor output) {
    const int batch_size = output.dimension(0);
    const int m = output.dimension(1);
    const int n = output.dimension(2);
    GpuLaunchConfig config = GetGpuLaunchConfig(batch_size * m * n, device);
    TF_CHECK_OK(GpuLaunchKernel(MatrixBandFillKernel<Scalar>,
                                config.block_count, config.thread_per_block, 0,
                                device.stream(), config.virtual_thread_count,
                                batch_size, m, n, num_lower_diags,
                                num_upper_diags, value, output.data()));
  }
};

}  // namespace functor

typedef Eigen::GpuDevice GPUDevice;

template <class Scalar>
class CholeskyOpGpu : public AsyncOpKernel {
 public:
  explicit CholeskyOpGpu(OpKernelConstruction* context)
      : AsyncOpKernel(context) {}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) final {
    const Tensor& input = context->input(0);
    const int ndims = input.dims();
    const int64_t n = input.dim_size(ndims - 1);
#if GOOGLE_CUDA
    cublasFillMode_t fill = CUBLAS_FILL_MODE_UPPER;
#elif TENSORFLOW_USE_ROCM
#if TF_ROCM_VERSION >= 40500
    hipsolverFillMode_t fill = HIPSOLVER_FILL_MODE_UPPER;
#else
    rocblas_fill fill = rocblas_fill_upper;
#endif
#endif
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
    std::unique_ptr<GpuSolver> solver = std::make_unique<GpuSolver>(context);
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
      OP_REQUIRES_OK_ASYNC(
          context,
          solver->PotrfBatched(fill, n, output_reshaped_ptrs_base, n,
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
    } else {
      dev_info.push_back(solver->GetDeviceLapackInfo(batch_size, "potrf"));
      for (int batch = 0; batch < batch_size; ++batch) {
        OP_REQUIRES_OK_ASYNC(
            context,
            solver->Potrf(fill, n, &output_reshaped(batch, 0, 0), n,
                          &dev_info.back()(batch)),
            done);
      }
    }

    // Register callback to check info after kernels finish.
    auto info_checker = [context, done, n](
                            const Status& status,
                            const std::vector<HostLapackInfo>& host_infos) {
      if (!status.ok() && errors::IsInvalidArgument(status) &&
          !host_infos.empty()) {
        Tensor* output = context->mutable_output(0);
        auto output_reshaped = output->template flat_inner_dims<Scalar, 3>();

        for (int i = 0; i < host_infos[0].size(); ++i) {
          if (host_infos[0](i) > 0) {
            LOG(WARNING) << "Cholesky decomposition was not successful for "
                            "batch "
                         << i
                         << ". The input might not be valid. "
                            "Filling lower-triangular output with NaNs.";
            typename TTypes<Scalar, 3>::Tensor output_batch(
                &output_reshaped(i, 0, 0), 1, n, n);
            functor::MatrixBandFillFunctor<GPUDevice, Scalar> band_fill;
            band_fill(context, context->eigen_device<GPUDevice>(),
                      /*num_lower_diags=*/n, /*num_upper_diags=*/0,
                      /*value=*/Eigen::NumTraits<Scalar>::quiet_NaN(),
                      output_batch);
          }
        }
      }
      done();
    };
    GpuSolver::CheckLapackInfoAndDeleteSolverAsync(std::move(solver), dev_info,
                                                   std::move(info_checker));
  }
};

REGISTER_LINALG_OP_GPU("Cholesky", (CholeskyOpGpu<float>), float);
REGISTER_LINALG_OP_GPU("Cholesky", (CholeskyOpGpu<double>), double);
REGISTER_LINALG_OP_GPU("Cholesky", (CholeskyOpGpu<complex64>), complex64);
REGISTER_LINALG_OP_GPU("Cholesky", (CholeskyOpGpu<complex128>), complex128);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
