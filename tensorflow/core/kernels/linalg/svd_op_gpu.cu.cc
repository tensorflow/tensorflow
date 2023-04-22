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
// TODO(shamanDevel): Enable complex inputs. This will require a specialization
//                    of Gesvd for complex inputs as well as a new kernel
//                    definition to output the singular values as reals
//                    instead of complex values. The current CPU implementation
//                    outputs the singular values as complex values and then
//                    casts them to reals in the python wrapper.
// TODO(rmlarsen/shamanDevel): This could use a bit of cleanup. We don't need to
// pass quite as many raw pointers around. Would also be nice to reduce code
// duplication.

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <algorithm>
#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/linalg/eye_functor.h"
#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_solvers.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

static const char kErrMsg[] =
    "Singular Value Decomposition was not successful. The input might not be "
    "valid.";

typedef Eigen::GpuDevice GPUDevice;

namespace {
// This kernel computes the reduction
// V' = sum_i (M_i * U_i,1 * S_i).
// The result is stored in V[batch] and has the same sign as the
// real value of V (which should be computed)
template <class Scalar>
__global__ void ComputeValueOfVKernel(Gpu2DLaunchConfig config, int64 m,
                                      int64 ldu, const Scalar* __restrict__ M,
                                      const Scalar* __restrict__ U,
                                      const Scalar* __restrict__ S,
                                      Scalar* __restrict__ V) {
  GPU_AXIS_KERNEL_LOOP(batch, config.virtual_thread_count.x, X) {
    GPU_AXIS_KERNEL_LOOP(i, config.virtual_thread_count.y, Y) {
      Scalar v = M[i + m * batch] * U[ldu * (i + m * batch)] * S[batch];
      GpuAtomicAdd(V + batch, v);
    }
  }
}

// Extracts the sign of V
// V[i] = V[i]>=0 ? 1 : 0
template <class Scalar>
__global__ void ExtractSignOfVKernel(GpuLaunchConfig config,
                                     Scalar* __restrict__ V) {
  GPU_1D_KERNEL_LOOP(i, config.virtual_thread_count) {
    V[i] = V[i] >= 0 ? Scalar(1) : Scalar(-1);
  }
}
}  // namespace

// Scalar: The input scalar type (can be complex)
template <class Scalar>
class SvdOpGpu : public AsyncOpKernel {
 public:
  using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

  explicit SvdOpGpu(OpKernelConstruction* context) : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("compute_uv", &compute_uv_));
    OP_REQUIRES_OK(context, context->GetAttr("full_matrices", &full_matrices_));
  }

  void RunSVD(OpKernelContext* context, DoneCallback done, int64 m, int64 n,
              int64 p, Tensor& M_copy, Tensor* S, Tensor* U, Tensor* V,
              std::unique_ptr<CudaSolver> solver) {
    // Compute U S V* = M.
    // 1. cuSolver works in column-major rather than row-major.
    // 2. Gesvd returns V*. GesvdjBatched returns V.
    // 3. Hence M should be transposed before input and
    //    a) U (rather than V) should be transposed on output with Gesvd.
    //    b) U and V should be transposed on output with GesvdjBatched.

    // get the pointers to input data
    Scalar* input_ptr;
    RealScalar* outputS_ptr;
    auto input_reshaped = M_copy.template flat_inner_dims<Scalar, 3>();
    input_ptr = input_reshaped.data();
    const int64 batch_size =
        M_copy.dims() > 2 ? input_reshaped.dimension(0) : 1;
    // Gesvdjbatched handles matrices up to 32x32.
    // TODO(jamessspencer): if not full_matrices, compute full U and V matrices
    // using Gesvdjbatched and return slices.
    const bool batched =
        m <= 32 && n <= 32 && batch_size > 1 && (full_matrices_ || m == n);

    // Copies of U and V if required so can take transposes after SVD.
    Tensor u_copy, v_copy;
    Scalar* outputU_ptr = NULL;
    Scalar* outputV_ptr = NULL;
    if (compute_uv_ || batched) {
      TensorShape u_shape, v_shape;
      if (batched) {
        // Gesvdjbatched seems to require U and V matrices even if the vectors
        // aren't computed.
        TensorShape shapeRaw = M_copy.shape();
        shapeRaw.RemoveLastDims(2);
        u_shape = shapeRaw;
        u_shape.AddDim(m);
        u_shape.AddDim(m);
        v_shape = shapeRaw;
        v_shape.AddDim(n);
        v_shape.AddDim(n);
      } else if (full_matrices_) {
        u_shape = U->shape();
        v_shape = V->shape();
      } else {
        TensorShape shapeRaw = M_copy.shape();
        shapeRaw.RemoveLastDims(2);
        u_shape = shapeRaw;
        u_shape.AddDim(p);
        u_shape.AddDim(m);
        v_shape = shapeRaw;
        v_shape.AddDim(p);
        v_shape.AddDim(n);
      }
      OP_REQUIRES_OK_ASYNC(
          context, solver->allocate_scoped_tensor(U->dtype(), u_shape, &u_copy),
          done);
      if (batched) {
        OP_REQUIRES_OK_ASYNC(
            context,
            solver->allocate_scoped_tensor(V->dtype(), v_shape, &v_copy), done);
      }
      outputU_ptr = u_copy.template flat_inner_dims<Scalar, 3>().data();
      if (batched) {
        outputV_ptr = v_copy.template flat_inner_dims<Scalar, 3>().data();
      } else {
        outputV_ptr = V->template flat_inner_dims<Scalar, 3>().data();
      }
    }

    outputS_ptr = S->template flat_inner_dims<RealScalar, 2>().data();
    std::vector<DeviceLapackInfo> dev_info;
    dev_info.push_back(solver->GetDeviceLapackInfo(batch_size, "gesvd"));
    int* dev_info_ptr = dev_info.back().mutable_data();

    // Save the input matrix
    // Needed for the n=1 fix, see below, since SVD destroys the input
    Tensor input_copy;
    if (compute_uv_ && n == 1) {
      OP_REQUIRES_OK_ASYNC(context,
                           solver->allocate_scoped_tensor(
                               DataTypeToEnum<Scalar>::v(),
                               TensorShape({batch_size, m}), &input_copy),
                           done);
      const GPUDevice& d = context->eigen_device<GPUDevice>();
      d.memcpy(input_copy.flat<Scalar>().data(), input_ptr,
               batch_size * m * sizeof(Scalar));
    }

    if (batched) {
      cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_NOVECTOR;
      if (compute_uv_) jobz = CUSOLVER_EIG_MODE_VECTOR;
      OP_REQUIRES_OK_ASYNC(
          context,
          solver->GesvdjBatched(jobz, m, n, input_ptr, m, outputS_ptr,
                                outputU_ptr, m, outputV_ptr, n, dev_info_ptr,
                                batch_size),
          done);
    } else {
      for (int64 batch = 0; batch < batch_size; ++batch) {
        Scalar* input = input_ptr + batch * m * n;
        RealScalar* outputS = outputS_ptr + batch * p;
        Scalar* outputU = NULL;
        Scalar* outputVT = NULL;
        char jobu = 'N';
        char jobvt = 'N';

        if (compute_uv_) {
          if (full_matrices_) {
            outputU = outputU_ptr + batch * m * m;
            outputVT = outputV_ptr + batch * n * n;
            jobu = 'A';
            jobvt = 'A';
          } else {
            outputU = outputU_ptr + batch * m * p;
            outputVT = outputV_ptr + batch * n * p;
            jobu = 'S';
            jobvt = 'S';
          }
        }

        OP_REQUIRES_OK_ASYNC(
            context,
            solver->Gesvd(jobu, jobvt, m, n, input, m, outputS, outputU, m,
                          outputVT, n, dev_info_ptr + batch),
            done);
      }
    }

    // This is a bug in cuSolver:
    // If n is one, then outputVT only contains zeros instead of ones.
    // Hence, I need to fill outputVT manually
    // The question is: +1 or -1?
    // -> Compute U*S and compare sign against M
    // But because S is zero except for the first entry, the multiplication
    // simplifies a lot.
    // However, what happens if M contains zeros? At these indices, it is
    // impossible to determine the value of V.
    // -> Compute V for all rows in M to cope for zeros.
    // 1. V' = sum_i (M_i * U_i,1 * S_i)
    // 2. V = {1, V'>=0, -1, V'<0}
    // TODO: what is with complex values?
    if (compute_uv_ && n == 1) {
      // 1. compute the (batched) sum
      const GPUDevice& d = context->eigen_device<GPUDevice>();
      d.memset(outputV_ptr, 0, batch_size * sizeof(Scalar));
      Gpu2DLaunchConfig cfg2D = GetGpu2DLaunchConfig(batch_size, m, d);
      TF_CHECK_OK(GpuLaunchKernel(ComputeValueOfVKernel<Scalar>,
                                  cfg2D.block_count, cfg2D.thread_per_block, 0,
                                  d.stream(), cfg2D, m, full_matrices_ ? m : p,
                                  input_copy.flat<Scalar>().data(), outputU_ptr,
                                  outputS_ptr, outputV_ptr));
      // 2. clamp V to -1 or +1
      GpuLaunchConfig cfg1D = GetGpuLaunchConfig(batch_size, d);
      TF_CHECK_OK(GpuLaunchKernel(ExtractSignOfVKernel<Scalar>,
                                  cfg1D.block_count, cfg1D.thread_per_block, 0,
                                  d.stream(), cfg1D, outputV_ptr));
    }

    if (compute_uv_) {
      auto device = context->eigen_device<GPUDevice>();
      OP_REQUIRES_OK_ASYNC(context, DoMatrixTranspose(device, u_copy, U), done);
      if (batched) {
        OP_REQUIRES_OK_ASYNC(context, DoMatrixTranspose(device, v_copy, V),
                             done);
      }
    }

    CheckResult(context, std::move(done), dev_info, std::move(solver));
  }

  void CheckResult(OpKernelContext* context, DoneCallback done,
                   const std::vector<DeviceLapackInfo>& dev_info,
                   std::unique_ptr<CudaSolver> solver) {
    auto info_checker = [context, done](
                            const Status& status,
                            const std::vector<HostLapackInfo>& /* unused */) {
      Status full_status = status;
      if (!full_status.ok()) {
        full_status.Update(errors::InvalidArgument(kErrMsg));
      }
      OP_REQUIRES_OK_ASYNC(context, full_status, done);
      done();
    };

    CudaSolver::CheckLapackInfoAndDeleteSolverAsync(std::move(solver), dev_info,
                                                    std::move(info_checker));
  }

  // The SVD if m >= n
  // TODO: can the two cases (MgeqN and MlessN) be simplified,
  //   common boilerplate be reduced, or even combined in one method?
  void PerformSVD_MgeqN(OpKernelContext* context, DoneCallback done, int64 m,
                        int64 n, int64 p, const Tensor& M, Tensor* S, Tensor* U,
                        Tensor* V) {
    // Transpose M, because cuSolver expects it to be column-major
    TensorShape shapeRaw = M.shape();
    shapeRaw.RemoveLastDims(2);
    TensorShape input_shape = shapeRaw;
    input_shape.AddDim(n);
    input_shape.AddDim(m);
    Tensor input_copy;
    // TODO(rmlarsen): Convert to std::make_unique when available.
    std::unique_ptr<CudaSolver> solver(new CudaSolver(context));
    OP_REQUIRES_OK_ASYNC(
        context,
        solver->allocate_scoped_tensor(M.dtype(), input_shape, &input_copy),
        done);
    auto device = context->eigen_device<GPUDevice>();
    OP_REQUIRES_OK_ASYNC(context, DoMatrixTranspose(device, M, &input_copy),
                         done);

    // Call the SVD: compute U S V* = M.
    RunSVD(context, done, m, n, p, input_copy, S, U, V, std::move(solver));
  }

  // The SVD if m < n
  void PerformSVD_MlessN(OpKernelContext* context, DoneCallback done, int64 m,
                         int64 n, int64 p, const Tensor& M, Tensor* S,
                         Tensor* U, Tensor* V) {
    // Perform the SVD on M'. cuSolver works column major so don't need to
    // transpose M.

    // Reuse the input buffer or make a copy for the SVD depending on whether
    // this op owns the input buffer exclusively. This is needed because the
    // SVD modifies the input
    // TODO(rmlarsen): Convert to std::make_unique when available.
    std::unique_ptr<CudaSolver> solver(new CudaSolver(context));
    Tensor input_copy;
    OP_REQUIRES_OK_ASYNC(
        context,
        solver->forward_input_or_allocate_scoped_tensor(
            {0}, DataTypeToEnum<Scalar>::value, M.shape(), &input_copy),
        done);

    if (!M.SharesBufferWith(input_copy)) {
      const GPUDevice& d = context->eigen_device<GPUDevice>();
      d.memcpy(input_copy.flat<Scalar>().data(), M.flat<Scalar>().data(),
               M.NumElements() * sizeof(Scalar));
    }

    // Call the SVD: compute V S U* = M*.
    // Note (m, n) and (U, V) are swapped accordingly.
    RunSVD(context, done, n, m, p, input_copy, S, V, U, std::move(solver));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) final {
    const Tensor& input = context->input(0);
    const int ndims = input.dims();

    // Validate inputs.
    OP_REQUIRES_ASYNC(
        context, ndims >= 2,
        errors::InvalidArgument("Input must have rank >= 2, got ", ndims),
        done);

    const int64 m = input.dim_size(ndims - 2);
    const int64 n = input.dim_size(ndims - 1);
    const int64 p = std::min(m, n);

    // output tensors.
    Tensor* outputU = NULL;
    Tensor* outputS = NULL;
    Tensor* outputV = NULL;

    // compute  shapes
    TensorShape shapeRaw = input.shape();
    shapeRaw.RemoveLastDims(2);
    TensorShape shapeS = shapeRaw;
    TensorShape shapeU = shapeRaw;
    TensorShape shapeV = shapeRaw;
    shapeS.AddDim(p);
    if (compute_uv_) {
      if (full_matrices_) {
        shapeU.AddDim(m);
        shapeU.AddDim(m);
        shapeV.AddDim(n);
        shapeV.AddDim(n);
      } else {
        shapeU.AddDim(m);
        shapeU.AddDim(p);
        shapeV.AddDim(n);
        shapeV.AddDim(p);
      }
    } else {
      shapeU = TensorShape({0});
      shapeV = TensorShape({0});
    }

    // allocate output
    OP_REQUIRES_OK_ASYNC(context, context->allocate_output(0, shapeS, &outputS),
                         done);
    OP_REQUIRES_OK_ASYNC(context, context->allocate_output(1, shapeU, &outputU),
                         done);
    OP_REQUIRES_OK_ASYNC(context, context->allocate_output(2, shapeV, &outputV),
                         done);

    if (n == 0 || m == 0) {
      if (n == m || !compute_uv_ || !full_matrices_) {
        // S, U, and V are all empty. Nothing to do.
        done();
        return;
      }
      auto device = context->eigen_device<GPUDevice>();
      functor::EyeFunctor<GPUDevice, Scalar> eye;
      if (m > 0) {
        // Return a full canonical basis for the column space.
        auto outputU_reshaped = outputU->flat_inner_dims<Scalar, 3>();
        eye(device, outputU_reshaped);
      } else if (n > 0) {
        // Return a full canonical basis for the row space.
        auto outputV_reshaped = outputV->flat_inner_dims<Scalar, 3>();
        eye(device, outputV_reshaped);
      }
      done();
      return;
    }

    // call implementations
    if (m >= n) {
      PerformSVD_MgeqN(context, done, m, n, p, input, outputS, outputU,
                       outputV);
    } else {
      PerformSVD_MlessN(context, done, m, n, p, input, outputS, outputU,
                        outputV);
    }
  }

 private:
  bool compute_uv_;
  bool full_matrices_;
};

// TODO: add support for complex types
REGISTER_LINALG_OP_GPU("Svd", (SvdOpGpu<float>), float);
REGISTER_LINALG_OP_GPU("Svd", (SvdOpGpu<double>), double);

// Deprecated kernels.
REGISTER_LINALG_OP_GPU("BatchSvd", (SvdOpGpu<float>), float);
REGISTER_LINALG_OP_GPU("BatchSvd", (SvdOpGpu<double>), double);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
