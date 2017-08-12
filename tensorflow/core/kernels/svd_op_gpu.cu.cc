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
// TODO(shamanDevel): Enable complex inputs. This will require additional tests
//                    and OP_REQUIRES.
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <algorithm>
#include <vector>

#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

#include "tensorflow/core/kernels/cuda_solvers.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

// I need to transpose V afterwards
#include "tensorflow/core/kernels/transpose_functor.h"

// Logging
#include <stdio.h>

namespace tensorflow {

static const char kErrMsg[] =
    "Singular Value Decomposition was not successful. The input might not be "
    "valid.";

typedef Eigen::GpuDevice GPUDevice;

// Scalar: The input scalar type (can be complex)
// SScalar: The output type for the singular value,
//   same as Scalar if real, or the real version if Scalar is complex
template <class Scalar, class SScalar>
class SvdOpGpu : public AsyncOpKernel {
 public:
  explicit SvdOpGpu(OpKernelConstruction* context) : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("compute_uv", &compute_uv_));
    OP_REQUIRES_OK(context, context->GetAttr("full_matrices", &full_matrices_));
  }

  void RunSVD(OpKernelContext* context, DoneCallback done, int64 m, int64 n,
              int64 p, int64 batch_size, Scalar* input_ptr,
              SScalar* outputS_ptr, Scalar* outputU_ptr, Scalar* outputVT_ptr,
              int* dev_info_ptr, CudaSolver& solver) {
    for (int64 i = 0; i < batch_size; ++i) {
      int lda = m;
      int ldu = m;
      int ldvt = n;
      Scalar* input = input_ptr + i * m * n;
      SScalar* outputS = outputS_ptr + i * p;
      Scalar* outputU = NULL;
      Scalar* outputVT = NULL;
      signed char jobu = 'N';
      signed char jobvt = 'N';

      if (compute_uv_) {
        if (full_matrices_) {
          outputU = outputU_ptr + i * m * m;
          outputVT = outputVT_ptr + i * n * n;
          jobu = 'A';
          jobvt = 'A';
        } else {
          outputU = outputU_ptr + i * m * p;
          outputVT = outputVT_ptr + i * n * p;
          jobu = 'S';
          jobvt = 'S';
        }
      }

      OP_REQUIRES_OK_ASYNC(
          context, solver.Gesvd(jobu, jobvt, m, n, input, lda, outputS, outputU,
                                ldu, outputVT, ldvt, dev_info_ptr + i),
          done);
    }
  }

  void CheckResult(OpKernelContext* context, DoneCallback done,
                   const std::vector<DeviceLapackInfo>& dev_info,
                   CudaSolver& solver, Tensor& catch1, Tensor& catch2) {
    auto info_checker = [context, dev_info, done, catch1, catch2](
        const Status& status, const std::vector<HostLapackInfo>& /* unused */) {
      Status full_status = status;
      if (!full_status.ok()) {
        full_status.Update(errors::InvalidArgument(kErrMsg));
      }
      OP_REQUIRES_OK_ASYNC(context, full_status, done);
      done();
    };

    OP_REQUIRES_OK_ASYNC(context, solver.CopyLapackInfoToHostAsync(
                                      dev_info, std::move(info_checker)),
                         done);
  }

  // The SVD if m >= n
  void PerformSVD_MgeqN(OpKernelContext* context, DoneCallback done, int64 m,
                        int64 n, int64 p, const gtl::ArraySlice<int32>& perm,
                        const Tensor& M, Tensor* S, Tensor* U, Tensor* V) {
    TensorShape shapeRaw = M.shape();
    shapeRaw.RemoveDim(shapeRaw.dims() - 1);
    shapeRaw.RemoveDim(shapeRaw.dims() - 1);

    // Transpose M, because cuSolver expects it to be column-major
    TensorShape input_shape = shapeRaw;
    input_shape.AddDim(n);
    input_shape.AddDim(m);
    Tensor input_copy;
    OP_REQUIRES_OK_ASYNC(
        context, context->allocate_temp(M.dtype(), input_shape, &input_copy),
        done);
    auto device = context->eigen_device<GPUDevice>();
    OP_REQUIRES_OK_ASYNC(
        context, 
        DoTranspose(device, M, perm, &input_copy),
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
          context, context->allocate_temp(U->dtype(), u_shape, &u_copy), done);
    }

    // get the pointers to the data
    Scalar* input_ptr;
    SScalar* outputS_ptr;
    Scalar* outputU_ptr = NULL;
    Scalar* outputV_ptr = NULL;
    auto input_reshaped = input_copy.template flat_inner_dims<Scalar, 3>();
    input_ptr = input_reshaped.data();
    outputS_ptr = S->template flat_inner_dims<SScalar, 2>().data();
    if (compute_uv_) {
      outputU_ptr = u_copy.template flat_inner_dims<Scalar, 3>().data();
      outputV_ptr = V->template flat_inner_dims<Scalar, 3>().data();
    }

    // call the SVD
    const int64 batch_size = input_reshaped.dimension(0);
    std::vector<DeviceLapackInfo> dev_info;
    dev_info.emplace_back(context, batch_size, "gesvd");
    CudaSolver solver(context);
    RunSVD(context, done, m, n, p, batch_size, input_ptr, outputS_ptr,
           outputU_ptr, outputV_ptr, dev_info.back().mutable_data(), solver);

    // Transpose U
    if (compute_uv_) {
      OP_REQUIRES_OK_ASYNC(
          context,
          DoTranspose(device, u_copy, perm, U),
          done);
    }

    // now check if the SVD operation succeeded or not
    CheckResult(context, done, dev_info, solver, input_copy, u_copy);
  }

  // The SVD if m < n
  void PerformSVD_MlessN(OpKernelContext* context, DoneCallback done, int64 m,
                         int64 n, int64 p, const gtl::ArraySlice<int32>& perm,
                         const Tensor& M, Tensor* S, Tensor* U, Tensor* V) {
    // Perform the SVD on M'

    // Reuse the input buffer or make a copy for the SVD depending on whether
    // this op owns the
    // input buffer exclusively. This is needed because the SVD modifies the
    // input
    Tensor input_copy;
    OP_REQUIRES_OK_ASYNC(context, context->forward_input_or_allocate_temp(
                                      {0}, DataTypeToEnum<Scalar>::value,
                                      M.shape(), &input_copy),
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
        shapeRaw.RemoveDim(shapeRaw.dims() - 1);
        shapeRaw.RemoveDim(shapeRaw.dims() - 1);
        v_shape = shapeRaw;
        v_shape.AddDim(p);
        v_shape.AddDim(n);
      }
      OP_REQUIRES_OK_ASYNC(
          context, context->allocate_temp(V->dtype(), v_shape, &v_copy), done);
    }

    // get the pointers to the data
    Scalar* input_ptr;
    SScalar* outputS_ptr;
    Scalar* outputU_ptr = NULL;
    Scalar* outputV_ptr = NULL;
    auto input_reshaped = input_copy.template flat_inner_dims<Scalar, 3>();
    input_ptr = input_reshaped.data();
    outputS_ptr = S->template flat_inner_dims<SScalar, 2>().data();
    if (compute_uv_) {
      // Note that U and V are flipped
      outputU_ptr = v_copy.template flat_inner_dims<Scalar, 3>().data();
      outputV_ptr = U->template flat_inner_dims<Scalar, 3>().data();
    }

    // call the SVD
    const int64 batch_size = input_reshaped.dimension(0);
    std::vector<DeviceLapackInfo> dev_info;
    dev_info.emplace_back(context, batch_size, "gesvd");
    CudaSolver solver(context);
    // Note that m and n are flipped
    RunSVD(context, done, n, m, p, batch_size, input_ptr, outputS_ptr,
           outputU_ptr, outputV_ptr, dev_info.back().mutable_data(), solver);

    // Transpose V
    if (compute_uv_) {
      auto device = context->eigen_device<GPUDevice>();
      OP_REQUIRES_OK_ASYNC(
          context,
          DoTranspose(device, v_copy, perm, V),
          done);
    }

    // now check if the SVD operation succeeded or not
    CheckResult(context, done, dev_info, solver, input_copy, v_copy);
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
    shapeRaw.RemoveDim(shapeRaw.dims() - 1);
    shapeRaw.RemoveDim(shapeRaw.dims() - 1);
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

    // Prepare permutation
    std::vector<int32> perm;
    for (size_t i = 0; i < ndims - 2; ++i) perm.push_back(i);
    perm.push_back(ndims - 1);  // transpose last two dimensions
    perm.push_back(ndims - 2);
    gtl::ArraySlice<int32> permAS(perm);

    // call implementations
    if (m >= n) {
      PerformSVD_MgeqN(context, done, m, n, p, permAS, input, outputS, outputU,
                       outputV);
    } else {
      PerformSVD_MlessN(context, done, m, n, p, permAS, input, outputS, outputU,
                        outputV);
    }
  }

 private:
  bool compute_uv_;
  bool full_matrices_;
};

// TODO: add support for complex types
REGISTER_LINALG_OP_GPU("Svd", (SvdOpGpu<float, float>), float);
REGISTER_LINALG_OP_GPU("Svd", (SvdOpGpu<double, double>), double);
REGISTER_LINALG_OP_GPU("BatchSvd", (SvdOpGpu<float, float>), float);
REGISTER_LINALG_OP_GPU("BatchSvd", (SvdOpGpu<double, double>), double);

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
