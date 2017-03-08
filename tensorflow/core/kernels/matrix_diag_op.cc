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

// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/kernels/matrix_diag_op.h"

#include <memory>
#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class MatrixDiagPartOp : public OpKernel {
 public:
  explicit MatrixDiagPartOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);

    const TensorShape& input_shape = input.shape();
    const int rank = input_shape.dims();

    // Preliminary validation of sizes.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input_shape),
                errors::InvalidArgument(
                    "input must be at least 2-dim, received shape: ",
                    input.shape().DebugString()));

    TensorShape output_shape;
    for (int i = 0; i < rank - 2; ++i) {
      output_shape.AddDim(input_shape.dim_size(i));
    }
    const int64 min_dim = std::min(input_shape.dim_size(rank - 2),
                                   input_shape.dim_size(rank - 1));
    output_shape.AddDim(min_dim);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    auto output_reshaped = output->flat_inner_dims<T, 2>();
    auto input_reshaped = input.flat_inner_dims<T, 3>();

    functor::MatrixDiagPart<Device, T>::Compute(
        context->eigen_device<Device>(), input_reshaped, output_reshaped);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(MatrixDiagPartOp);
};

template <typename Device, typename T>
class MatrixDiagOp : public OpKernel {
 public:
  explicit MatrixDiagOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);

    const TensorShape& input_shape = input.shape();
    const int rank = input_shape.dims();

    // Preliminary validation of sizes.
    OP_REQUIRES(context, TensorShapeUtils::IsVectorOrHigher(input_shape),
                errors::InvalidArgument(
                    "input must be at least 1-dim, received shape: ",
                    input.shape().DebugString()));

    const int64 k = input_shape.dim_size(rank - 1);
    auto input_reshaped = input.flat_inner_dims<T, 2>();

    TensorShape output_shape = input_shape;
    output_shape.AddDim(k);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    auto output_reshaped = output->flat_inner_dims<T, 3>();

    functor::MatrixDiag<Device, T>::Compute(context->eigen_device<Device>(),
                                            input_reshaped, output_reshaped);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(MatrixDiagOp);
};

#define REGISTER_MATRIX_DIAG(type)                                         \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("MatrixDiag").Device(DEVICE_CPU).TypeConstraint<type>("T"),     \
      MatrixDiagOp<CPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("MatrixDiagPart").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      MatrixDiagPartOp<CPUDevice, type>);
TF_CALL_NUMBER_TYPES(REGISTER_MATRIX_DIAG);
#undef REGISTER_MATRIX_DIAG

// Registration of the deprecated kernel.
// Delete after 10mar2017.
#define REGISTER_BATCH_MATRIX_DIAG(type)                                    \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("BatchMatrixDiag").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      MatrixDiagOp<CPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(Name("BatchMatrixDiagPart")                       \
                              .Device(DEVICE_CPU)                           \
                              .TypeConstraint<type>("T"),                   \
                          MatrixDiagPartOp<CPUDevice, type>);
TF_CALL_NUMBER_TYPES(REGISTER_BATCH_MATRIX_DIAG);
#undef REGISTER_BATCH_MATRIX_DIAG

// Implementation of the functor specialization for CPU.
namespace functor {
template <typename T>
struct MatrixDiag<CPUDevice, T> {
  static void Compute(const CPUDevice& d,
                      typename TTypes<T, 2>::ConstTensor input,
                      typename TTypes<T, 3>::Tensor output) {
    output.device(d) = output.constant(T());
    for (int64 r = 0; r < output.dimension(0); ++r) {
      for (int64 d = 0; d < output.dimension(1); ++d) {
        output(r, d, d) = input(r, d);
      }
    }
  }
};

template <typename T>
struct MatrixDiagPart<CPUDevice, T> {
  static void Compute(const CPUDevice& d,
                      typename TTypes<T, 3>::ConstTensor input,
                      typename TTypes<T, 2>::Tensor output) {
    for (int64 r = 0; r < output.dimension(0); ++r) {
      for (int64 d = 0; d < output.dimension(1); ++d) {
        output(r, d) = input(r, d, d);
      }
    }
  }
};

}  // namespace functor

#if GOOGLE_CUDA

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                         \
  template <>                                                       \
  void MatrixDiag<GPUDevice, T>::Compute(                           \
      const GPUDevice& d, typename TTypes<T, 2>::ConstTensor input, \
      typename TTypes<T, 3>::Tensor output);                        \
  extern template struct MatrixDiag<GPUDevice, T>;                  \
  template <>                                                       \
  void MatrixDiagPart<GPUDevice, T>::Compute(                       \
      const GPUDevice& d, typename TTypes<T, 3>::ConstTensor input, \
      typename TTypes<T, 2>::Tensor output);                        \
  extern template struct MatrixDiagPart<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);

}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_MATRIX_DIAG_GPU(type)                                     \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("MatrixDiag").Device(DEVICE_GPU).TypeConstraint<type>("T"),     \
      MatrixDiagOp<GPUDevice, type>);                                      \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("MatrixDiagPart").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      MatrixDiagPartOp<GPUDevice, type>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_MATRIX_DIAG_GPU);
#undef REGISTER_MATRIX_DIAG_GPU

// Registration of the deprecated kernel.
// Delete after 10mar2017.
#define REGISTER_BATCH_MATRIX_DIAG_GPU(type)                                \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("BatchMatrixDiag").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      MatrixDiagOp<GPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(Name("BatchMatrixDiagPart")                       \
                              .Device(DEVICE_GPU)                           \
                              .TypeConstraint<type>("T"),                   \
                          MatrixDiagPartOp<GPUDevice, type>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_BATCH_MATRIX_DIAG_GPU);
#undef REGISTER_BATCH_MATRIX_DIAG_GPU

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
