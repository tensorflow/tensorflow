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
// TODO(konstantinos): Enable complex inputs. This will require additional tests
//                     and OP_REQUIRES.
#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <algorithm>

#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/cuda_solvers.h"
#include "tensorflow/core/platform/stream_executor.h"

// I need to transpose V afterwards
#include "transpose_functor.h"
#include <vector>

// Logging
#include <stdio.h>

namespace tensorflow {

static const char kErrMsg[] =
    "Singular Value Decomposition was not successful. The input might not be valid.";


typedef Eigen::GpuDevice GPUDevice;

template <class Scalar>
class SvdOpGpu : public AsyncOpKernel {
 public:
  explicit SvdOpGpu(OpKernelConstruction* context)
      : AsyncOpKernel(context) 
  {
    OP_REQUIRES_OK(context, context->GetAttr("compute_uv", &compute_uv_));
    OP_REQUIRES_OK(context, context->GetAttr("full_matrices", &full_matrices_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) final {    
    const Tensor& input = context->input(0);
    const int ndims = input.dims();
    const int64 m = input.dim_size(ndims - 2);
    const int64 n = input.dim_size(ndims - 1);
    const int64 p = std::min(m, n);
    
    //This is a limitation of cuSolver
    OP_REQUIRES_ASYNC(context, m>=n,
                errors::InvalidArgument("GPU implementation of SVD only supports m>=n, but m=",m, ", n=",n), done);
    
    // Validate inputs.
    OP_REQUIRES_ASYNC(
        context, ndims >= 2,
        errors::InvalidArgument("Input must have rank >= 2, got ", ndims),
        done);

    // output tensors.
    Tensor* outputU = NULL;
    Tensor* outputS = NULL;
    Tensor outputVT;
    Tensor* outputV = NULL;
    
    //compute  shapes
    TensorShape shapeRaw = input.shape();
    shapeRaw.RemoveDim(shapeRaw.dims()-1);
    shapeRaw.RemoveDim(shapeRaw.dims()-1);
    TensorShape shapeS = shapeRaw;
    TensorShape shapeU = shapeRaw;
    TensorShape shapeVT = shapeRaw;
    TensorShape shapeV = shapeRaw;
    shapeS.AddDim(p);
    if (compute_uv_) {
        if (full_matrices_) {
            shapeU.AddDim(m);
            shapeU.AddDim(m);
            shapeVT.AddDim(n);
            shapeVT.AddDim(n);
            shapeV.AddDim(n);
            shapeV.AddDim(n);
        } else {
            shapeU.AddDim(m);
            shapeU.AddDim(p);
            shapeVT.AddDim(p);
            shapeVT.AddDim(n);
            shapeV.AddDim(n);
            shapeV.AddDim(p);
        }
    } else {
        shapeU = TensorShape({0});
        shapeV = TensorShape({0});
    }
    
    //allocate output
    OP_REQUIRES_OK_ASYNC(context,
                         context->allocate_output(
                             0, shapeS, &outputS),
                         done);
    OP_REQUIRES_OK_ASYNC(context,
                         context->allocate_output(
                             1, shapeU, &outputU),
                         done);
    OP_REQUIRES_OK_ASYNC(context,
                         context->allocate_output(
                             2, shapeV, &outputV),
                         done);
    if (compute_uv_) {
        OP_REQUIRES_OK_ASYNC(context,
                         context->allocate_temp(
                             input.dtype(), shapeVT, &outputVT),
                         done);
    }

    if (n == 0 || m == 0) {
      // If X is an empty matrix (0 rows, 0 col), X * X' == X.
      // Therefore, we return X.
      done();
      return;
    }

    // Copy and reshape input tensor
    // SVD modifies the input, so I need to copy it.
    Tensor inputCopy;
    OP_REQUIRES_OK_ASYNC(context,
                         context->allocate_temp(
                             input.dtype(), input.shape(), &inputCopy),
                         done);
                         
    cudaMemcpy(inputCopy.flat<Scalar>().data(), input.flat<Scalar>().data(), 
        input.NumElements() * sizeof(Scalar), cudaMemcpyDeviceToDevice);
        
    auto input_reshaped = inputCopy.template flat_inner_dims<Scalar, 3>();

    
    // Reshape output tensors
    auto outputS_reshaped = outputS->template flat_inner_dims<Scalar, 2>();
    Scalar* outputU_reshaped_ptr = NULL;
    Scalar* outputVT_reshaped_ptr = NULL;
    if (compute_uv_) {
        auto outputU_reshaped = outputU->template flat_inner_dims<Scalar, 3>();
        auto outputVT_reshaped = outputVT.template flat_inner_dims<Scalar, 3>();
        outputU_reshaped_ptr = outputU_reshaped.data();
        outputVT_reshaped_ptr = outputVT_reshaped.data();
    }

    // Launch a SVD kernel for each matrix in the batch.
    const int64 batch_size = input_reshaped.dimension(0);
    DeviceLapackInfo dev_info(context, batch_size, "gesvd");
    
    // TODO(rmlarsen): Parallelize over batches if it turns out to be
    // an important use case.
    CudaSolver solver(context);    
    for (int64 i = 0; i < batch_size; ++i) {
      Scalar* input_ptr = input_reshaped.data() + i * m * n;
      Scalar* outputS_ptr = outputS_reshaped.data() + i * p;
      int lda = m;
      
      Scalar* outputU_ptr = NULL;
      Scalar* outputVT_ptr = NULL;
      signed char jobu = 'N';
      signed char jobvt = 'N';
      int ldu = m;
      int ldvt = n;
      if (compute_uv_) {
        if (full_matrices_) {
            outputU_ptr = outputU_reshaped_ptr + i * m * m;
            outputVT_ptr = outputVT_reshaped_ptr + i * n * n;
            jobu = 'A';
            jobvt = 'A';
        } else {
            outputU_ptr = outputU_reshaped_ptr + i * m * p;
            outputVT_ptr = outputVT_reshaped_ptr + i * n * p;
            jobu = 'S';
            jobvt = 'S';
        }
      }

      int* dev_info_ptr = dev_info.mutable_data() + i;
      OP_REQUIRES_OK_ASYNC(
          context,
          solver.Gesvd(jobu, jobvt, m, n, 
            input_ptr, lda, outputS_ptr, outputU_ptr, ldu, outputVT_ptr, ldvt, dev_info_ptr),
          done);
    }

    // Test if it was successfull.
    // I'm not using solver.CopyLapackInfoToHostAsync
    // because it resulted in a memory corruption on the host
    // (because I have some operations afterwards, so I can't use done())
    HostLapackInfo host_info(context, batch_size, "gesvd");
    cudaMemcpy(host_info.mutable_data(), dev_info.data(), sizeof(int) * batch_size, cudaMemcpyDeviceToHost);
    Status status;
    for (int i = 0; i < batch_size && status.ok(); ++i) {
        const int info_value = host_info.data()[i];
        if (info_value != 0) {
          status = errors::InvalidArgument(
              "Got info = ", info_value, " for batch index ", i,
              ", expected info = 0. Debug_info =",
              host_info.debug_info());
        }
    }
    if (!status.ok()) {
        status.Update(errors::InvalidArgument(kErrMsg));
    }
    OP_REQUIRES_OK_ASYNC(context, status, done);
    // TODO: Maybe switch to solver.CopyLapackInfoToHostAsync again
    // but call this after the transposing below.
    
    if (compute_uv_) {
        // Transpose VT and copy to output tensor V
        std::vector<int32> perm;
        for (size_t i=0; i<ndims-2; ++i) perm.push_back(i);
        perm.push_back(ndims-1); //transpose last two dimensions
        perm.push_back(ndims-2);
        gtl::ArraySlice<int32> permAS(perm);
            outputV->shape().DebugString().c_str(), (int) outputV->shape().num_elements(), outputV);
        auto device = context->eigen_device<GPUDevice>();
        DoTranspose(device, outputVT, permAS, outputV);
    }

    done();
  }
  
private:
  bool compute_uv_;
  bool full_matrices_;
};

REGISTER_LINALG_OP_GPU("Svd", (SvdOpGpu<float>), float);
REGISTER_LINALG_OP_GPU("Svd", (SvdOpGpu<double>), double);
REGISTER_LINALG_OP_GPU("BatchSvd", (SvdOpGpu<float>), float);
REGISTER_LINALG_OP_GPU("BatchSvd", (SvdOpGpu<double>), double);

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
