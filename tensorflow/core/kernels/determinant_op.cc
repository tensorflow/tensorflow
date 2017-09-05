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
#endif

#include "third_party/eigen3/Eigen/LU"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

#if GOOGLE_CUDA
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/cuda_solvers.h"
#include "tensorflow/core/kernels/fill_functor.h"
#endif

namespace tensorflow {

template <class Scalar>
class DeterminantOp : public LinearAlgebraOp<Scalar> {
 public:
  typedef LinearAlgebraOp<Scalar> Base;

  explicit DeterminantOp(OpKernelConstruction* context) : Base(context) {}

  using TensorShapes = typename Base::TensorShapes;
  using MatrixMaps = typename Base::MatrixMaps;
  using ConstMatrixMaps = typename Base::ConstMatrixMaps;

  TensorShapes GetOutputMatrixShapes(
      const TensorShapes& input_matrix_shape) const final {
    return TensorShapes({TensorShape({})});
  }

  void ComputeMatrix(OpKernelContext* context, const ConstMatrixMaps& inputs,
                     MatrixMaps* outputs) final {
    Scalar determinant;
    if (inputs[0].rows() == 0) {
      // An empty matrix' determinant is defined to be 1.  See wikipedia.
      determinant = 1;
    } else {
      determinant = inputs[0].determinant();
    }
    // TODO(rmlarsen): Don't fail on infinite determinants, since that could
    // be a valid result and the user should check for it instead.
    OP_REQUIRES(context, Eigen::numext::isfinite(determinant),
                errors::InvalidArgument("The determinant is not finite."));
    outputs->at(0)(0, 0) = determinant;
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

    // Reuse the input buffer or make a copy for the factorization step,
    // depending on whether this ops owns it exclusively.
    Tensor input_copy;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->forward_input_or_allocate_temp(
            {0}, DataTypeToEnum<Scalar>::value, input.shape(), &input_copy),
        done);
    if (!input.SharesBufferWith(input_copy)) {
      d.memcpy(input_copy.flat<Scalar>().data(), input.flat<Scalar>().data(),
               input.NumElements() * sizeof(Scalar));
    }
    auto input_copy_reshaped = input_copy.template flat_inner_dims<Scalar, 3>();
    const int64 batch_size = input_copy_reshaped.dimension(0);

    // Allocate pivots on the device.
    ScratchSpace<int> pivots(context, n * batch_size, /* on_host */ false);

    // Prepare pointer arrays for cuBlas' batch interface.
    // TODO(rmlarsen): Find a way to encode pointer arrays in pinned host memory
    // without the ugly casting.
    ScratchSpace<uint8> input_copy_ptrs(context, sizeof(Scalar*) * batch_size,
                                        /* on_host */ true);
    auto output_reshaped = out->template flat_inner_dims<Scalar, 1>();

    // Compute the partially pivoted LU factorization(s) of the matrix/matrices.
    CudaSolver solver(context);
    std::vector<DeviceLapackInfo> dev_info;
    if (n / batch_size <= 128) {
      // For small matrices or large batch sizes, we use the batched interface
      // from cuBlas.
      const Scalar** input_copy_ptrs_base =
          reinterpret_cast<const Scalar**>(input_copy_ptrs.mutable_data());
      for (int batch = 0; batch < batch_size; ++batch) {
        input_copy_ptrs_base[batch] =
            input_copy_reshaped.data() + batch * n * n;
      }
      dev_info.emplace_back(context, batch_size, "getrfBatched");
      OP_REQUIRES_OK_ASYNC(
          context,
          solver.GetrfBatched(n, input_copy_ptrs_base, n, pivots.mutable_data(),
                              &dev_info.back(), batch_size),
          done);
    } else {
      // For small batch sizes we use the non-batched interface from cuSolver,
      // which is much faster for large matrices.
      dev_info.emplace_back(context, batch_size, "getrf");
      int* dev_info_ptr = dev_info.back().mutable_data();
      Scalar* input_copy_ptr = input_copy.flat<Scalar>().data();
      int* pivots_ptr = pivots.mutable_data();
      for (int batch = 0; batch < batch_size; ++batch) {
        OP_REQUIRES_OK_ASYNC(
            context,
            solver.Getrf(n, n, input_copy_ptr, n, pivots_ptr, dev_info_ptr),
            done);
        input_copy_ptr += n * n;
        pivots_ptr += n;
        ++dev_info_ptr;
      }
    }

    // Compute the determinant for each batch as (-1)^s * prod(diag(U)),
    // where s is the order of the permutation encoded in pivots and U is the
    // upper triangular factor of the LU factorization, which is written to
    // input_copy by the Getrf{Batched} kernel.
    functor::DeterminantFromPivotedLUFunctor<GPUDevice, Scalar> functor;
    functor(d, input_copy_reshaped, pivots.data(), output_reshaped,
            dev_info.back().mutable_data());

    // Register callback to check info after kernels finish. Also capture the
    // temporary Tensors/ScratchSpace so they don't get deallocated before the
    // kernels run. TODO(rmlarsen): Use move capture once C++14 becomes
    // available.
    auto info_checker = [context, dev_info, input_copy, pivots, input_copy_ptrs,
                         done](const Status& status,
                               const std::vector<HostLapackInfo>& host_infos) {
      if (!status.ok() && errors::IsInvalidArgument(status) &&
          !host_infos.empty()) {
        for (int i = 0; i < host_infos[0].size(); ++i) {
          // It is OK for a matrix to be singular (signaled by info > 0),
          // corresponding to determinant of zero, but we do want to catch
          // invalid arguments to GetrfBatched.
          OP_REQUIRES_ASYNC(
              context,
              host_infos[0].data()[i] >= 0 ||
                  host_infos[0].data()[i] == kint32min,
              errors::InvalidArgument("Invalid input argument no. ",
                                      host_infos[0].data()[i],
                                      " for batch index ", i, "."),
              done);
          OP_REQUIRES_ASYNC(
              context, host_infos[0].data()[i] != kint32min,
              errors::InvalidArgument("The determinant is not finite."), done);
        }
      }
      done();
    };

    OP_REQUIRES_OK_ASYNC(
        context,
        solver.CopyLapackInfoToHostAsync(dev_info, std::move(info_checker)),
        done);
  }
};

REGISTER_LINALG_OP_GPU("MatrixDeterminant", (DeterminantOpGpu<float>), float);
REGISTER_LINALG_OP_GPU("MatrixDeterminant", (DeterminantOpGpu<double>), double);
REGISTER_LINALG_OP_GPU("MatrixDeterminant", (DeterminantOpGpu<complex64>),
                       complex64);
REGISTER_LINALG_OP_GPU("MatrixDeterminant", (DeterminantOpGpu<complex128>),
                       complex128);
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

}  // namespace tensorflow
