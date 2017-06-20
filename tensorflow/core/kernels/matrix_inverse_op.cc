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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/Eigen/LU"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

#if GOOGLE_CUDA
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/cuda_solvers.h"
#endif

namespace tensorflow {

template <class Scalar>
class MatrixInverseOp : public LinearAlgebraOp<Scalar> {
 public:
  INHERIT_LINALG_TYPEDEFS(Scalar);

  explicit MatrixInverseOp(OpKernelConstruction* context) : Base(context) {
    OP_REQUIRES_OK(context, context->GetAttr("adjoint", &adjoint_));
  }

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
    const ConstMatrixMap& input = inputs[0];
    if (input.rows() == 0) {
      // By definition, an empty matrix's inverse is an empty matrix.
      return;
    }
    Eigen::PartialPivLU<Matrix> lu_decomposition;
    if (adjoint_) {
      // TODO(rmlarsen): For Eigen 3.2, this creates a temporary copy.
      // Make sure to backport: https://bitbucket.org/eigen/eigen/commits/
      // bd2219a74c96dfe3f6bc2c23588749e36d2d8173
      lu_decomposition.compute(input.adjoint());
    } else {
      lu_decomposition.compute(input);
    }
    // TODO(rmlarsen): Add check based on condition number estimation.
    // PartialPivLU cannot give strong guarantees on invertibility, but
    // we can at least guard against exact zero pivots. This can occur as
    // a result of basic user mistakes, such as providing integer valued
    // matrices that are exactly singular, or due to underflow if this
    // code is run with denormals being flushed to zero.
    using RealScalar = typename Base::RealScalar;
    const RealScalar min_abs_pivot =
        lu_decomposition.matrixLU().diagonal().cwiseAbs().minCoeff();
    OP_REQUIRES(context, min_abs_pivot > RealScalar(0),
                errors::InvalidArgument("Input is not invertible."));
    outputs->at(0).noalias() = lu_decomposition.inverse();
  }

 private:
  bool adjoint_;

  TF_DISALLOW_COPY_AND_ASSIGN(MatrixInverseOp);
};

#if GOOGLE_CUDA

typedef Eigen::GpuDevice GPUDevice;

template <class Scalar>
class MatrixInverseOpGpu : public AsyncOpKernel {
 public:
  explicit MatrixInverseOpGpu(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("adjoint", &adjoint_));
  }

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
        errors::InvalidArgument("Input matrices must be squares, got",
                                input.dim_size(ndims - 2), " != ", n),
        done);

    // Allocate output.
    Tensor* out;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->forward_input_or_allocate_output({0}, 0, input.shape(), &out),
        done);

    // By definition, an empty matrix's inverse is an empty matrix.
    if (input.NumElements() == 0) {
      done();
      return;
    }

    // Make a copy of the (possible adjointed) input that we will use for the
    // factorization step.
    Tensor input_copy;
    OP_REQUIRES_OK_ASYNC(context,
                         context->allocate_temp(DataTypeToEnum<Scalar>::value,
                                                input.shape(), &input_copy),
                         done);
    const GPUDevice& d = context->eigen_device<GPUDevice>();
    auto input_copy_reshaped = input_copy.template flat_inner_dims<Scalar, 3>();
    auto input_reshaped = input.template flat_inner_dims<Scalar, 3>();
    if (!adjoint_) {
      d.memcpy(input_copy_reshaped.data(), input_reshaped.data(),
               input.NumElements() * sizeof(Scalar));
    } else {
      functor::AdjointBatchFunctor<GPUDevice, Scalar> functor;
      functor(d, input_reshaped, input_copy_reshaped);
    }
    const int64 batch_size = input_copy_reshaped.dimension(0);

    // Allocate pivots on the device.
    ScratchSpace<int> pivots(context, n * batch_size, /* on_host */ false);

    // Prepare pointer arrays for cuBlas' batch interface.
    // TODO(rmlarsen): Find a way to encode pointer arrays in pinned host memory
    // without the ugly casting.
    ScratchSpace<uint8> input_copy_ptrs(context, sizeof(Scalar*) * batch_size,
                                        /* on_host */ true);
    ScratchSpace<uint8> output_ptrs(context, sizeof(Scalar*) * batch_size,
                                    /* on_host */ true);
    const Scalar** input_copy_ptrs_base =
        reinterpret_cast<const Scalar**>(input_copy_ptrs.mutable_data());
    const Scalar** output_ptrs_base =
        reinterpret_cast<const Scalar**>(output_ptrs.mutable_data());
    auto output_reshaped = out->template flat_inner_dims<Scalar, 3>();
    for (int64 i = 0; i < batch_size; ++i) {
      input_copy_ptrs_base[i] = input_copy_reshaped.data() + i * n * n;
      output_ptrs_base[i] = output_reshaped.data() + i * n * n;
    }

    // Launch the two solver kernels back to back without waiting.
    // 1. Compute the partially pivoted LU factorization(s) of the
    // matrix/matrices.
    CudaSolver solver(context);
    std::vector<DeviceLapackInfo> dev_info;
    dev_info.emplace_back(context, batch_size, "getrf");
    OP_REQUIRES_OK_ASYNC(
        context,
        solver.GetrfBatched(n, input_copy_ptrs_base, n, pivots.mutable_data(),
                            &dev_info.back(), batch_size),
        done);
    // 2. Compute the inverse(s).
    dev_info.emplace_back(context, batch_size, "getri");
    OP_REQUIRES_OK_ASYNC(
        context,
        solver.GetriBatched(n, input_copy_ptrs_base, n, pivots.data(),
                            output_ptrs_base, n, &dev_info.back(), batch_size),
        done);

    // Register callback to check info after kernels finish. Also capture the
    // temporary Tensors/ScratchSpace so they don't get deallocated before the
    // kernels run. TODO(rmlarsen): Use move capture once C++14 becomes
    // available.
    auto info_checker = [context, dev_info, input_copy, pivots, input_copy_ptrs,
                         output_ptrs,
                         done](const Status& status,
                               const std::vector<HostLapackInfo>& host_infos) {
      if (!status.ok() && errors::IsInvalidArgument(status) &&
          !host_infos.empty()) {
        for (int i = 0; i < host_infos[0].size(); ++i) {
          // Match the CPU error message for singular matrices. Otherwise
          // just print the original error message from the call itself
          // below.
          OP_REQUIRES_ASYNC(context, host_infos[0].data()[i] <= 0,
                            errors::InvalidArgument("Input is not invertible."),
                            done);
        }
      }
      OP_REQUIRES_OK_ASYNC(context, status, done);
      done();
    };

    OP_REQUIRES_OK_ASYNC(
        context,
        solver.CopyLapackInfoToHostAsync(dev_info, std::move(info_checker)),
        done);
  }

 private:
  bool adjoint_;
};

REGISTER_LINALG_OP_GPU("MatrixInverse", (MatrixInverseOpGpu<float>), float);
REGISTER_LINALG_OP_GPU("MatrixInverse", (MatrixInverseOpGpu<double>), double);
REGISTER_LINALG_OP_GPU("MatrixInverse", (MatrixInverseOpGpu<complex64>),
                       complex64);
REGISTER_LINALG_OP_GPU("MatrixInverse", (MatrixInverseOpGpu<complex128>),
                       complex128);

#endif  // GOOGLE_CUDA

REGISTER_LINALG_OP("MatrixInverse", (MatrixInverseOp<float>), float);
REGISTER_LINALG_OP("MatrixInverse", (MatrixInverseOp<double>), double);
REGISTER_LINALG_OP("MatrixInverse", (MatrixInverseOp<complex64>), complex64);
REGISTER_LINALG_OP("MatrixInverse", (MatrixInverseOp<complex128>), complex128);
REGISTER_LINALG_OP("BatchMatrixInverse", (MatrixInverseOp<float>), float);
REGISTER_LINALG_OP("BatchMatrixInverse", (MatrixInverseOp<double>), double);

}  // namespace tensorflow
