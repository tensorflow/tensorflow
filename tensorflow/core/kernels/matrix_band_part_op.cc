/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/matrix_band_part_op.h"

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
class MatrixBandPartOp : public OpKernel {
 public:
  explicit MatrixBandPartOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& num_lower_in = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(num_lower_in.shape()),
                errors::InvalidArgument("num_lower must be scalar, got shape ",
                                        num_lower_in.shape().DebugString()));
    const int64 num_lower = num_lower_in.scalar<int64>()();

    const Tensor& num_upper_in = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(num_upper_in.shape()),
                errors::InvalidArgument("num_upper must be scalar, got shape ",
                                        num_upper_in.shape().DebugString()));
    const int64 num_upper = num_upper_in.scalar<int64>()();

    const TensorShape& input_shape = input.shape();
    // Preliminary validation of sizes.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input_shape),
                errors::InvalidArgument(
                    "input must be at least 2-dim, received shape: ",
                    input.shape().DebugString()));
    auto input_reshaped = input.flat_inner_dims<T, 3>();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &output));
    auto output_reshaped = output->flat_inner_dims<T, 3>();
    functor::MatrixBandPart<Device, T>::Compute(
        context->eigen_device<Device>(), num_lower, num_upper, input_reshaped,
        output_reshaped);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(MatrixBandPartOp);
};

#define REGISTER_MATRIX_BAND_PART(type)                                    \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("MatrixBandPart").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      MatrixBandPartOp<CPUDevice, type>);
TF_CALL_POD_TYPES(REGISTER_MATRIX_BAND_PART);
#undef REGISTER_MATRIX_BAND_PART

// Registration of the deprecated kernel.
// Delete after 10mar2017.
#define REGISTER_BATCH_MATRIX_BAND_PART(type)             \
  REGISTER_KERNEL_BUILDER(Name("BatchMatrixBandPart")     \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<type>("T"), \
                          MatrixBandPartOp<CPUDevice, type>);
TF_CALL_NUMBER_TYPES(REGISTER_BATCH_MATRIX_BAND_PART);
#undef REGISTER_BATCH_MATRIX_BAND_PART

// Implementation of the functor specialization for CPU.
namespace functor {
template <typename T>
struct MatrixBandPart<CPUDevice, T> {
  static void Compute(const CPUDevice& d, int64 num_lower, int64 num_upper,
                      typename TTypes<T, 3>::ConstTensor input,
                      typename TTypes<T, 3>::Tensor output) {
    if ((num_lower < 0 || num_lower >= input.dimension(1)) &&
        (num_upper < 0 || num_upper >= input.dimension(2))) {
      output.device(d) = input;
    } else {
      output.device(d) = output.constant(T());
      for (int64 r = 0; r < output.dimension(0); ++r) {
        for (int64 i = 0; i < output.dimension(1); ++i) {
          const int64 band_start =
              num_lower < 0 ? 0 : std::max(0ll, i - num_lower);
          const int64 band_end =
              num_upper < 0 ? output.dimension(2)
                            : std::min(static_cast<int64>(output.dimension(2)),
                                       i + num_upper + 1);
          if (band_start < band_end) {
            const Eigen::DSizes<Eigen::DenseIndex, 3> indices(r, i, band_start);
            const Eigen::DSizes<Eigen::DenseIndex, 3> sizes(
                1, 1, band_end - band_start);
            output.slice(indices, sizes) = input.slice(indices, sizes);
          }
        }
      }
    }
  }
};

}  // namespace functor

#if GOOGLE_CUDA

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                  \
  template <>                                                                \
  void MatrixBandPart<GPUDevice, T>::Compute(                                \
      const GPUDevice& d, Eigen::DenseIndex num_lower,                       \
      Eigen::DenseIndex num_upper, typename TTypes<T, 3>::ConstTensor input, \
      typename TTypes<T, 3>::Tensor output);                                 \
  extern template struct MatrixBandPart<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
TF_CALL_bool(DECLARE_GPU_SPEC);
TF_CALL_complex64(DECLARE_GPU_SPEC);
TF_CALL_complex128(DECLARE_GPU_SPEC);
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_MATRIX_BAND_PART_GPU(type)              \
  REGISTER_KERNEL_BUILDER(Name("MatrixBandPart")         \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("num_lower")   \
                              .HostMemory("num_upper"),  \
                          MatrixBandPartOp<GPUDevice, type>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_MATRIX_BAND_PART_GPU);
TF_CALL_bool(REGISTER_MATRIX_BAND_PART_GPU);
TF_CALL_complex64(REGISTER_MATRIX_BAND_PART_GPU);
TF_CALL_complex128(REGISTER_MATRIX_BAND_PART_GPU);
#undef REGISTER_MATRIX_BAND_PART_GPU

// Registration of the deprecated kernel.
// Delete after 10mar2017.
#define REGISTER_BATCH_MATRIX_BAND_PART_GPU(type)        \
  REGISTER_KERNEL_BUILDER(Name("BatchMatrixBandPart")    \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("num_lower")   \
                              .HostMemory("num_upper"),  \
                          MatrixBandPartOp<GPUDevice, type>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_BATCH_MATRIX_BAND_PART_GPU);
#undef REGISTER_BATCH_MATRIX_BAND_PART_GPU

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
