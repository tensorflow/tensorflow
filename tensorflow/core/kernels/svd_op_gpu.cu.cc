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
#include "tensorflow/core/kernels/cuda_solvers.h"
#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

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
__global__ void ComputeValueOfVKernel(Cuda2DLaunchConfig config, int64 m,
                                      int64 ldu, const Scalar* M,
                                      const Scalar* U, const Scalar* S,
                                      Scalar* V) {
  CUDA_AXIS_KERNEL_LOOP(batch, config.virtual_thread_count.x, X) {
    CUDA_AXIS_KERNEL_LOOP(i, config.virtual_thread_count.y, Y) {
      Scalar v = M[i + m * batch] * U[ldu * (i + m * batch)] * S[batch];
      CudaAtomicAdd(V + batch, v);
    }
  }
}

// Extracts the sign of V
// V[i] = V[i]>=0 ? 1 : 0
template <class Scalar>
__global__ void ExtractSignOfVKernel(CudaLaunchConfig config, Scalar* V) {
  CUDA_1D_KERNEL_LOOP(i, config.virtual_thread_count) {
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
              int64 p, int64 batch_size, Scalar* input_ptr,
              RealScalar* outputS_ptr, Scalar* outputU_ptr,
              Scalar* outputVT_ptr, int* dev_info_ptr, CudaSolver* solver) {
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
          outputVT = outputVT_ptr + batch * n * n;
          jobu = 'A';
          jobvt = 'A';
        } else {
          outputU = outputU_ptr + batch * m * p;
          outputVT = outputVT_ptr + batch * n * p;
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
      d.memset(outputVT_ptr, 0, batch_size * sizeof(Scalar));
      Cuda2DLaunchConfig cfg2D = GetCuda2DLaunchConfig(batch_size, m, d);
      ComputeValueOfVKernel<<<cfg2D.block_count, cfg2D.thread_per_block, 0,
                              d.stream()>>>(
          cfg2D, m, full_matrices_ ? m : p, input_copy.flat<Scalar>().data(),
          outputU_ptr, outputS_ptr, outputVT_ptr);
      // 2. clamp V to -1 or +1
      CudaLaunchConfig cfg1D = GetCudaLaunchConfig(batch_size, d);
      ExtractSignOfVKernel<<<cfg1D.block_count, cfg1D.thread_per_block, 0,
                             d.stream()>>>(cfg1D, outputVT_ptr);
    }
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
    TensorShape shapeRaw = M.shape();
    shapeRaw.RemoveLastDims(2);

    // Transpose M, because cuSolver expects it to be column-major
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

    // I need to transpose U at the end
    // Not V, because cuSolver work column-major
    Tensor u_copy;
    if (compute_uv_) {
      TensorShape u_shape;
      if (full_matrices_) {
        u_shape = U->shape();
      } else {
        u_shape = shapeRaw;
        u_shape.AddDim(p);
        u_shape.AddDim(m);
      }
      OP_REQUIRES_OK_ASYNC(
          context, solver->allocate_scoped_tensor(U->dtype(), u_shape, &u_copy),
          done);
    }

    // get the pointers to the data
    Scalar* input_ptr;
    RealScalar* outputS_ptr;
    Scalar* outputU_ptr = NULL;
    Scalar* outputV_ptr = NULL;
    auto input_reshaped = input_copy.template flat_inner_dims<Scalar, 3>();
    input_ptr = input_reshaped.data();
    outputS_ptr = S->template flat_inner_dims<RealScalar, 2>().data();
    if (compute_uv_) {
      outputU_ptr = u_copy.template flat_inner_dims<Scalar, 3>().data();
      outputV_ptr = V->template flat_inner_dims<Scalar, 3>().data();
    }

    // call the SVD
    const int64 batch_size = input_reshaped.dimension(0);
    std::vector<DeviceLapackInfo> dev_info;
    dev_info.push_back(solver->GetDeviceLapackInfo(batch_size, "gesvd"));
    RunSVD(context, done, m, n, p, batch_size, input_ptr, outputS_ptr,
           outputU_ptr, outputV_ptr, dev_info.back().mutable_data(),
           solver.get());

    // Transpose U
    if (compute_uv_) {
      OP_REQUIRES_OK_ASYNC(context, DoMatrixTranspose(device, u_copy, U), done);
    }

    // now check if the SVD operation succeeded or not
    CheckResult(context, std::move(done), dev_info, std::move(solver));
  }

  // The SVD if m < n
  void PerformSVD_MlessN(OpKernelContext* context, DoneCallback done, int64 m,
                         int64 n, int64 p, const Tensor& M, Tensor* S,
                         Tensor* U, Tensor* V) {
    // Perform the SVD on M'

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

    // I need to transpose V at the end
    Tensor v_copy;
    if (compute_uv_) {
      TensorShape v_shape;
      if (full_matrices_) {
        v_shape = V->shape();
      } else {
        TensorShape shapeRaw = M.shape();
        shapeRaw.RemoveLastDims(2);
        v_shape = shapeRaw;
        v_shape.AddDim(p);
        v_shape.AddDim(n);
      }
      OP_REQUIRES_OK_ASYNC(
          context, solver->allocate_scoped_tensor(V->dtype(), v_shape, &v_copy),
          done);
    }

    // get the pointers to the data
    Scalar* input_ptr;
    RealScalar* outputS_ptr;
    Scalar* outputU_ptr = NULL;
    Scalar* outputV_ptr = NULL;
    auto input_reshaped = input_copy.template flat_inner_dims<Scalar, 3>();
    input_ptr = input_reshaped.data();
    outputS_ptr = S->template flat_inner_dims<RealScalar, 2>().data();
    if (compute_uv_) {
      // Note that U and V are flipped
      outputU_ptr = v_copy.template flat_inner_dims<Scalar, 3>().data();
      outputV_ptr = U->template flat_inner_dims<Scalar, 3>().data();
    }

    // call the SVD
    const int64 batch_size = input_reshaped.dimension(0);
    std::vector<DeviceLapackInfo> dev_info;
    dev_info.push_back(solver->GetDeviceLapackInfo(batch_size, "gesvd"));
    // Note that m and n are flipped
    RunSVD(context, done, n, m, p, batch_size, input_ptr, outputS_ptr,
           outputU_ptr, outputV_ptr, dev_info.back().mutable_data(),
           solver.get());

    // Transpose V
    if (compute_uv_) {
      auto device = context->eigen_device<GPUDevice>();
      OP_REQUIRES_OK_ASYNC(context, DoMatrixTranspose(device, v_copy, V), done);
    }

    // now check if the SVD operation succeeded or not
    CheckResult(context, std::move(done), dev_info, std::move(solver));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) final {
    const Tensor& input = context->input(0);
    const int ndims = input.dims();
    const int64 m = input.dim_size(ndims - 2);
    const int64 n = input.dim_size(ndims - 1);
    const int64 p = std::min(m, n);

    // Validate inputs.
    OP_REQUIRES_ASYNC(
        context, ndims >= 2,
        errors::InvalidArgument("Input must have rank >= 2, got ", ndims),
        done);

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
      // If X is an empty matrix (0 rows, 0 col), X * X' == X.
      // Therefore, we return X.
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
