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

#include "tensorflow/core/kernels/batch_matrix_set_diag_op.h"

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
class BatchMatrixSetDiagOp : public OpKernel {
 public:
  explicit BatchMatrixSetDiagOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& diag = context->input(1);

    const TensorShape& input_shape = input.shape();
    const TensorShape& diag_shape = diag.shape();
    const int rank = input_shape.dims();

    // Preliminary validation of sizes.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input_shape),
                errors::InvalidArgument(
                    "input must be at least 2-dim, received shape: ",
                    input.shape().DebugString()));

    // Check to make sure the last two dimensions have the same value
    const int64 k = input_shape.dim_size(rank - 1);
    OP_REQUIRES(
        context, k == input_shape.dim_size(rank - 2),
        errors::InvalidArgument(
            "input's last two dimensions must be equal, received shape: ",
            input.shape().DebugString()));

    TensorShape input_shape_but_one = input_shape;
    input_shape_but_one.RemoveDim(rank - 1);

    OP_REQUIRES(context, input_shape_but_one == diag_shape,
                errors::InvalidArgument(
                    "must have diagonal.shape == input.shape[:-1], but "
                    "received input shape: ",
                    input_shape.DebugString(), " and diagonal shape: ",
                    diag_shape.DebugString()));

    auto input_reshaped = input.flat_inner_dims<T, 3>();
    auto diag_reshaped = diag.flat_inner_dims<T, 2>();

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output));

    auto output_reshaped = output->flat_inner_dims<T, 3>();
    Tensor scratch_tensor;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DataTypeToEnum<T>::value,
                                          TensorShape({}), &scratch_tensor));
    auto scratch = scratch_tensor.scalar<T>();

    functor::BatchMatrixSetDiag<Device, T>::Compute(
        context->eigen_device<Device>(), input_reshaped, diag_reshaped, scratch,
        output_reshaped);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(BatchMatrixSetDiagOp);
};

#define REGISTER_BATCH_MATRIX_SET_DIAG(type)                                   \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("BatchMatrixSetDiag").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BatchMatrixSetDiagOp<CPUDevice, type>);

TF_CALL_NUMBER_TYPES(REGISTER_BATCH_MATRIX_SET_DIAG);

namespace functor {

// Implementation of the functor specialization for CPU.
template <typename T>
struct BatchMatrixSetDiag<CPUDevice, T> {
  static void Compute(const CPUDevice& d,
                      typename TTypes<T, 3>::ConstTensor input,
                      typename TTypes<T, 2>::ConstTensor diag,
                      typename TTypes<T>::Scalar scratch,
                      typename TTypes<T, 3>::Tensor output) {
    output.device(d) = input;
    for (int64 r = 0; r < output.dimension(0); ++r) {
      for (int64 d = 0; d < output.dimension(1); ++d) {
        output(r, d, d) = diag(r, d);
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
  void BatchMatrixSetDiag<GPUDevice, T>::Compute(                   \
      const GPUDevice& d, typename TTypes<T, 3>::ConstTensor input, \
      typename TTypes<T, 2>::ConstTensor diag,                      \
      typename TTypes<T>::Scalar scratch,                           \
      typename TTypes<T, 3>::Tensor output);                        \
  extern template struct BatchMatrixSetDiag<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);

}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_BATCH_MATRIX_SET_DIAG_GPU(type)                               \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("BatchMatrixSetDiag").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      BatchMatrixSetDiagOp<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_BATCH_MATRIX_SET_DIAG_GPU);

#undef REGISTER_BATCH_MATRIX_SET_DIAG_GPU

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
