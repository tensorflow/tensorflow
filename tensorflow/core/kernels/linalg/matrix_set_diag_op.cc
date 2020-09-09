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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/linalg/matrix_set_diag_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/linalg/matrix_diag_op.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class MatrixSetDiagOp : public OpKernel {
 public:
  explicit MatrixSetDiagOp(OpKernelConstruction* context) : OpKernel(context) {
    // MatrixSetDiagV3-specific.
    if (context->HasAttr("align")) {
      functor::ReadAlignment(context, &left_align_superdiagonal_,
                             &left_align_subdiagonal_);
    }
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& diag = context->input(1);

    // MatrixSetDiag and MatrixSetDiagV2 both use this OpKernel. MatrixSetDiag
    // only has two inputs, so we have to check the number of inputs before
    // reading additional parameters in MatrixSetDiagV2.
    int32 lower_diag_index = 0;
    int32 upper_diag_index = 0;

    // MatrixSetDiagV2-specific.
    if (context->num_inputs() > kNumV1Inputs) {
      auto& diag_index = context->input(2);
      OP_REQUIRES(context,
                  TensorShapeUtils::IsScalar(diag_index.shape()) ||
                      TensorShapeUtils::IsVector(diag_index.shape()),
                  errors::InvalidArgument(
                      "diag_index must be a scalar or vector, received shape: ",
                      diag_index.shape().DebugString()));
      lower_diag_index = diag_index.flat<int32>()(0);
      upper_diag_index = lower_diag_index;
      if (TensorShapeUtils::IsVector(diag_index.shape())) {
        auto diag_index_size = diag_index.dim_size(0);
        OP_REQUIRES(
            context, 0 < diag_index_size && diag_index_size <= 2,
            errors::InvalidArgument(
                "diag_index must have only one or two elements, received ",
                diag_index_size, " elements."));
        if (diag_index_size > 1) {
          upper_diag_index = diag_index.flat<int32>()(1);
        }
      }
    }

    const TensorShape& input_shape = input.shape();
    const TensorShape& diag_shape = diag.shape();
    const int input_rank = input_shape.dims();

    // Preliminary validation of sizes.
    OP_REQUIRES(context, TensorShapeUtils::IsMatrixOrHigher(input_shape),
                errors::InvalidArgument(
                    "input must be at least 2-dim, received shape: ",
                    input.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVectorOrHigher(diag_shape),
                errors::InvalidArgument(
                    "diagonal must be at least 1-dim, received shape: ",
                    diag_shape.DebugString()));

    // Make sure lower_diag_index and upper_diag_index is valid.
    const Eigen::Index num_rows = input_shape.dim_size(input_rank - 2);
    const Eigen::Index num_cols = input_shape.dim_size(input_rank - 1);
    OP_REQUIRES(  // Checks lower_diag_index == 0 for when matrix shape = 0.
        context,
        (-num_rows < lower_diag_index && lower_diag_index < num_cols) ||
            lower_diag_index == 0,
        errors::InvalidArgument(
            "lower_diag_index is out of bound: ", lower_diag_index,
            " It must be between ", -num_rows, " and ", num_cols));
    OP_REQUIRES(context,
                (-num_rows < upper_diag_index && upper_diag_index < num_cols) ||
                    upper_diag_index == 0,
                errors::InvalidArgument(
                    "upper_diag_index is out of bound: ", upper_diag_index,
                    " It must be between ", -num_rows, " and ", num_cols));
    OP_REQUIRES(
        context, lower_diag_index <= upper_diag_index,
        errors::InvalidArgument(
            "lower_diag_index must not be larger than upper_diag_index: ",
            lower_diag_index, " > ", upper_diag_index));

    // Check if diag size is consistent with input.
    const Eigen::Index num_diags = upper_diag_index - lower_diag_index + 1;
    OP_REQUIRES(
        context,
        lower_diag_index == upper_diag_index ||
            (diag_shape.dim_size(input_rank - 2) == num_diags),
        errors::InvalidArgument("The number of diagonals provided in `diag` "
                                "is not consistent with `lower_diag_index` and "
                                "`upper_diag_index`"));

    TensorShape expected_diag_shape = input_shape;
    expected_diag_shape.RemoveLastDims(2);
    if (num_diags > 1) expected_diag_shape.AddDim(num_diags);
    const int32 max_diag_len =
        std::min(num_rows + std::min(upper_diag_index, 0),
                 num_cols - std::max(lower_diag_index, 0));
    expected_diag_shape.AddDim(max_diag_len);
    OP_REQUIRES(
        context, expected_diag_shape == diag_shape,
        errors::InvalidArgument(
            "Either first dimensions of diagonal don't match input.shape[:-2], "
            "or diagonal.shape[:-1] is not equal to the longests diagonal in "
            "range [lower_diag_index:upper_diag_index].\nInput shape: ",
            input_shape.DebugString(),
            "\nDiagonal shape: ", diag_shape.DebugString(),
            "\nExpected diagonal shape: ", expected_diag_shape.DebugString()));

    if (input.NumElements() == 0) {
      // This is a no-op.
      context->set_output(0, input);
      return;
    }

    auto input_reshaped = input.flat_inner_dims<T, 3>();
    auto diag_reshaped = diag.flat<T>();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input_shape, &output));
    auto output_reshaped = output->flat_inner_dims<T, 3>();
    functor::MatrixSetDiag<Device, T>::Compute(
        context, context->eigen_device<Device>(), input_reshaped, diag_reshaped,
        output_reshaped, lower_diag_index, upper_diag_index, max_diag_len,
        left_align_superdiagonal_, left_align_subdiagonal_);
  }

 private:
  bool left_align_superdiagonal_ = true;
  bool left_align_subdiagonal_ = true;
  static constexpr int kNumV1Inputs = 2;
  TF_DISALLOW_COPY_AND_ASSIGN(MatrixSetDiagOp);
};

#define REGISTER_MATRIX_SET_DIAG(type)                                      \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("MatrixSetDiag").Device(DEVICE_CPU).TypeConstraint<type>("T"),   \
      MatrixSetDiagOp<CPUDevice, type>);                                    \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("MatrixSetDiagV2").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      MatrixSetDiagOp<CPUDevice, type>);                                    \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("MatrixSetDiagV3").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      MatrixSetDiagOp<CPUDevice, type>);

TF_CALL_POD_TYPES(REGISTER_MATRIX_SET_DIAG);
#undef REGISTER_MATRIX_SET_DIAG

// Registration of the deprecated kernel.
// Delete after 10mar2017.
#define REGISTER_BATCH_MATRIX_SET_DIAG(type)                                   \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("BatchMatrixSetDiag").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      MatrixSetDiagOp<CPUDevice, type>);
TF_CALL_POD_TYPES(REGISTER_BATCH_MATRIX_SET_DIAG);
#undef REGISTER_BATCH_MATRIX_SET_DIAG

namespace functor {

// Implementation of the functor specialization for CPU.
template <typename T>
struct MatrixSetDiag<CPUDevice, T> {
  static void Compute(OpKernelContext* context, const CPUDevice& device,
                      typename TTypes<T, 3>::ConstTensor& input,
                      typename TTypes<T>::ConstTensor& diag,
                      typename TTypes<T, 3>::Tensor& output,
                      const Eigen::Index lower_diag_index,
                      const Eigen::Index upper_diag_index,
                      const Eigen::Index max_diag_len,
                      const bool left_align_superdiagonal,
                      const bool left_align_subdiagonal) {
    if (input.data() != output.data()) {
      output.device(device) = input;
    }
    const Eigen::Index num_diags = upper_diag_index - lower_diag_index + 1;
    auto compute_shard = [&output, &diag, &upper_diag_index, &max_diag_len,
                          &num_diags, &left_align_superdiagonal,
                          &left_align_subdiagonal](Eigen::Index begin,
                                                   Eigen::Index end) {
      const Eigen::Index num_rows = output.dimension(1);
      const Eigen::Index num_cols = output.dimension(2);
      Eigen::Index diag_base_index = begin * num_diags * max_diag_len;
      for (Eigen::Index batch = begin; batch < end; ++batch) {
        for (Eigen::Index m = 0; m < num_diags; ++m) {
          const Eigen::Index diag_index = upper_diag_index - m;
          int diag_len, content_offset;
          std::tie(diag_len, content_offset) = ComputeDiagLenAndContentOffset(
              diag_index, max_diag_len, num_rows, num_cols,
              left_align_superdiagonal, left_align_subdiagonal);

          // Make two separate cases to save some index calculations.
          if (diag_index >= 0) {
            for (Eigen::Index n = 0; n < diag_len; ++n) {
              output(batch, n, n + diag_index) =
                  diag(diag_base_index + n + content_offset);
            }
          } else {
            for (Eigen::Index n = 0; n < diag_len; ++n) {
              output(batch, n - diag_index, n) =
                  diag(diag_base_index + n + content_offset);
            }
          }
          diag_base_index += max_diag_len;
        }
      }
    };
    auto thread_pool =
        context->device()->tensorflow_cpu_worker_threads()->workers;
    // TODO(penporn): Tune for the best constant in cost_per_batch.
    const Eigen::Index cost_per_batch = 10 * num_diags * max_diag_len;
    thread_pool->ParallelFor(output.dimension(0), cost_per_batch,
                             std::move(compute_shard));
  }
};

}  // namespace functor

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                    \
  template <>                                                                  \
  void MatrixSetDiag<GPUDevice, T>::Compute(                                   \
      OpKernelContext* context, const GPUDevice& device,                       \
      typename TTypes<T, 3>::ConstTensor& input,                               \
      typename TTypes<T>::ConstTensor& diag,                                   \
      typename TTypes<T, 3>::Tensor& output,                                   \
      const Eigen::Index lower_diag_index,                                     \
      const Eigen::Index upper_diag_index, const Eigen::Index max_diag_len,    \
      const bool left_align_superdiagonal, const bool left_align_subdiagonal); \
  extern template struct MatrixSetDiag<GPUDevice, T>;

TF_CALL_GPU_ALL_TYPES(DECLARE_GPU_SPEC);

}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_MATRIX_SET_DIAG_GPU(type)                                \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("MatrixSetDiag").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      MatrixSetDiagOp<GPUDevice, type>);                                  \
  REGISTER_KERNEL_BUILDER(Name("MatrixSetDiagV2")                         \
                              .Device(DEVICE_GPU)                         \
                              .TypeConstraint<type>("T")                  \
                              .HostMemory("k"),                           \
                          MatrixSetDiagOp<GPUDevice, type>);              \
  REGISTER_KERNEL_BUILDER(Name("MatrixSetDiagV3")                         \
                              .Device(DEVICE_GPU)                         \
                              .TypeConstraint<type>("T")                  \
                              .HostMemory("k"),                           \
                          MatrixSetDiagOp<GPUDevice, type>);

TF_CALL_GPU_ALL_TYPES(REGISTER_MATRIX_SET_DIAG_GPU);
#undef REGISTER_MATRIX_SET_DIAG_GPU

// Registration of the deprecated kernel.
// Delete after 10mar2017.
#define REGISTER_BATCH_MATRIX_SET_DIAG_GPU(type)                               \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("BatchMatrixSetDiag").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      MatrixSetDiagOp<GPUDevice, type>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_BATCH_MATRIX_SET_DIAG_GPU);
#undef REGISTER_BATCH_MATRIX_SET_DIAG_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
