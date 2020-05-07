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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/matrix_band_part_op.h"

#include <algorithm>
#include <memory>
#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/threadpool.h"
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
    const TensorShape& input_shape = input.shape();
    // Preliminary validation of sizes.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input_shape),
                errors::InvalidArgument(
                    "input must be at least 2-dim, received shape: ",
                    input.shape().DebugString()));
    auto input_reshaped = input.flat_inner_dims<T, 3>();

    const Tensor& num_lower_in = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(num_lower_in.shape()),
                errors::InvalidArgument("num_lower must be scalar, got shape ",
                                        num_lower_in.shape().DebugString()));

    auto as_int64_scalar = [](const Tensor& tensor) -> int64 {
      if (tensor.dtype() == DT_INT32) {
        return tensor.scalar<int32>()();
      } else {
        return tensor.scalar<int64>()();
      }
    };
    const int64 num_lower = as_int64_scalar(num_lower_in);
    OP_REQUIRES(
        context, num_lower <= input_reshaped.dimension(1),
        errors::InvalidArgument(
            "num_lower must be negative or less or equal to number of rows (",
            input_reshaped.dimension(1), ") got: ", num_lower));

    const Tensor& num_upper_in = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(num_upper_in.shape()),
                errors::InvalidArgument("num_upper must be scalar, got shape ",
                                        num_upper_in.shape().DebugString()));
    const int64 num_upper = as_int64_scalar(num_upper_in);
    OP_REQUIRES(context, num_upper <= input_reshaped.dimension(2),
                errors::InvalidArgument("num_upper must be negative or less or "
                                        "equal to number of columns (",
                                        input_reshaped.dimension(2),
                                        ") got: ", num_upper));

    if (input.NumElements() == 0 ||
        ((num_lower < 0 || num_lower == input_reshaped.dimension(1)) &&
         (num_upper < 0 || num_upper == input_reshaped.dimension(2)))) {
      // This is a no-op.
      context->set_output(0, input);
      return;
    }

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input_shape, &output));
    auto output_reshaped = output->flat_inner_dims<T, 3>();
    functor::MatrixBandPartFunctor<Device, T> fn;
    fn(context, context->eigen_device<Device>(), num_lower, num_upper,
       input_reshaped, output_reshaped);
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

// CPU implementation of BandPartFunctor.
typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Scalar>
struct MatrixBandPartFunctor<CPUDevice, Scalar> {
  void operator()(OpKernelContext* context, const CPUDevice& device,
                  int num_lower_diags, int num_upper_diags,
                  typename TTypes<Scalar, 3>::ConstTensor input,
                  typename TTypes<Scalar, 3>::Tensor output) {
    const int64 b = input.dimension(0);
    const int64 m = input.dimension(1);
    const int64 n = input.dimension(2);
    auto thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    const int64 total_rows = b * m;
    const int64 row_cost = 10 * n;
    const bool in_place = input.data() == output.data();
    auto compute_shard = [=, &input, &output](int64 begin, int64 end) {
      if (!in_place) {
        std::fill(output.data() + begin * n, output.data() + end * n,
                  Scalar(0));
      }
      const int64 batch_begin = begin / m;
      const int64 batch_end = (end + m - 1) / m;
      for (int64 batch = batch_begin; batch < batch_end; ++batch) {
        const int64 row_begin = begin > batch * m ? begin % m : 0;
        const int64 row_end = end < (batch + 1) * m ? end % m : m;
        for (int64 row = row_begin; row < row_end; ++row) {
          const int64 band_start =
              num_lower_diags < 0
                  ? 0
                  : std::min(n, std::max(int64{0}, row - num_lower_diags));
          const int64 band_end =
              num_upper_diags < 0
                  ? n
                  : std::min(static_cast<int64>(n), row + num_upper_diags + 1);
          if (in_place) {
            if (band_start > 0) {
              std::fill(&output(batch, row, 0), &output(batch, row, band_start),
                        Scalar(0));
            }
            if (band_end < n) {
              std::fill(&output(batch, row, band_end), &output(batch, row, n),
                        Scalar(0));
            }
          } else {
            if (band_start < band_end) {
              const Eigen::DSizes<Eigen::DenseIndex, 3> indices(batch, row,
                                                                band_start);
              const Eigen::DSizes<Eigen::DenseIndex, 3> sizes(
                  1, 1, band_end - band_start);
              output.slice(indices, sizes) = input.slice(indices, sizes);
            }
          }
        }
      }
    };
    thread_pool->ParallelFor(total_rows, row_cost, std::move(compute_shard));
  }
};

#define DEFINE_CPU_SPEC(T) template struct MatrixBandPartFunctor<CPUDevice, T>;
TF_CALL_POD_TYPES(DEFINE_CPU_SPEC);
#undef DEFINE_CPU_SPEC

}  // namespace functor

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                            \
  template <>                                                          \
  struct MatrixBandPartFunctor<GPUDevice, T> {                         \
    void operator()(OpKernelContext* context, const GPUDevice& device, \
                    int num_upper_diags, int num_lower_diags,          \
                    typename TTypes<T, 3>::ConstTensor input,          \
                    typename TTypes<T, 3>::Tensor output);             \
  };                                                                   \
  extern template struct MatrixBandPartFunctor<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
TF_CALL_bool(DECLARE_GPU_SPEC);
TF_CALL_complex64(DECLARE_GPU_SPEC);
TF_CALL_complex128(DECLARE_GPU_SPEC);
#undef DECLARE_GPU_SPEC
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

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
