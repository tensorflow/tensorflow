/* Copyright 2015 Google Inc. All Rights Reserved.

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

// See docs in ../ops/nn_ops.cc.

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include <vector>
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
struct LaunchGeneric {
  static void launch(OpKernelContext* ctx, const Tensor& input,
                     const Tensor& filter, int row_stride, int col_stride,
                     const Eigen::PaddingType& padding, Tensor* output,
                     TensorFormat data_format) {
    CHECK(data_format == FORMAT_NHWC) << "Generic conv implementation only "
                                         "supports NHWC tensor format for now.";
    if (filter.dim_size(1) == filter.dim_size(0) && filter.dim_size(0) == 1 &&
        row_stride == 1 && col_stride == 1) {
      // For 1x1 kernel, the 2D convolution is reduced to matrix
      // multiplication.
      //
      // TODO(vrv): We should be able to call SpatialConvolution
      // and it will produce the same result, but doing so
      // led to NaNs during training.  Using matmul instead for now.
      int conv_width = 1;  // Width for the convolution step.
      for (int i = 0; i < 3; ++i) {
        conv_width *= output->dim_size(i);
      }

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
      functor::MatMulConvFunctor<Device, T>()(
          ctx->eigen_device<Device>(),
          output->shaped<T, 2>({conv_width, filter.dim_size(3)}),
          input.shaped<T, 2>({conv_width, filter.dim_size(2)}),
          filter.shaped<T, 2>({filter.dim_size(2), filter.dim_size(3)}),
          dim_pair);
    } else {
      functor::SpatialConvolution<Device, T>()(
          ctx->eigen_device<Device>(), output->tensor<T, 4>(),
          input.tensor<T, 4>(), filter.tensor<T, 4>(), row_stride, col_stride,
          padding);
    }
  }
};

template <typename Device, typename T>
struct LaunchConvOp;

template <typename T>
struct LaunchConvOp<CPUDevice, T> {
  static void launch(OpKernelContext* ctx, bool use_cudnn, const Tensor& input,
                     const Tensor& filter, int row_stride, int col_stride,
                     const Eigen::PaddingType& padding, Tensor* output,
                     TensorFormat data_format) {
    LaunchGeneric<CPUDevice, T>::launch(ctx, input, filter, row_stride,
                                        col_stride, padding, output,
                                        data_format);
  }
};

template <typename Device, typename T>
class Conv2DOp : public BinaryOp<T> {
 public:
  explicit Conv2DOp(OpKernelConstruction* context) : BinaryOp<T>(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("use_cudnn_on_gpu", &use_cudnn_));
    use_cudnn_ &= CanUseCudnn();
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    const int64 stride_n = GetTensorDim(strides_, data_format_, 'N');
    const int64 stride_c = GetTensorDim(strides_, data_format_, 'C');
    OP_REQUIRES(
        context, stride_n == 1 && stride_c == 1,
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]

    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth]
    const Tensor& filter = context->input(1);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, filter.dims() == 4,
                errors::InvalidArgument("filter must be 4-dimensional: ",
                                        filter.shape().DebugString()));

    for (int i = 0; i < 3; i++) {
      OP_REQUIRES(context, FastBoundsCheck(filter.dim_size(i),
                                           std::numeric_limits<int>::max()),
                  errors::InvalidArgument("filter too large"));
    }

    // The last dimension for input is in_depth. It must be the same as the
    // filter's in_depth.
    const int64 in_depth = GetTensorDim(input, data_format_, 'C');
    OP_REQUIRES(
        context, in_depth == filter.dim_size(2),
        errors::InvalidArgument("input and filter must have the same depth: ",
                                in_depth, " vs ", filter.dim_size(2)));

    // The last dimension for filter is out_depth.
    const int out_depth = static_cast<int>(filter.dim_size(3));

    // The second dimension for input is rows/height.
    // The first dimension for filter is rows/height.
    const int64 input_rows_raw = GetTensorDim(input, data_format_, 'H');
    OP_REQUIRES(context, FastBoundsCheck(input_rows_raw,
                                         std::numeric_limits<int>::max()),
                errors::InvalidArgument("Input rows too large"));
    const int input_rows = static_cast<int>(input_rows_raw);
    const int filter_rows = static_cast<int>(filter.dim_size(0));

    // The third dimension for input is columns/width.
    // The second dimension for filter is columns/width.
    const int64 input_cols_raw = GetTensorDim(input, data_format_, 'W');
    OP_REQUIRES(context, FastBoundsCheck(input_cols_raw,
                                         std::numeric_limits<int>::max()),
                errors::InvalidArgument("Input cols too large"));
    const int input_cols = static_cast<int>(input_cols_raw);
    const int filter_cols = static_cast<int>(filter.dim_size(1));

    // The first dimension for input is batch.
    const int64 batch_raw = GetTensorDim(input, data_format_, 'N');
    OP_REQUIRES(context,
                FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
                errors::InvalidArgument("batch is too large"));
    const int batch = static_cast<int>(batch_raw);

    // For now we take the stride from the second and third dimensions only (we
    // do not support striding on the batch or depth dimension).
    const int stride_rows = GetTensorDim(strides_, data_format_, 'H');
    const int stride_cols = GetTensorDim(strides_, data_format_, 'W');

    int out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    if (filter_cols == filter_rows && filter_rows == 1 && stride_rows == 1 &&
        stride_cols == 1) {
      // For 1x1 kernel, the 2D convolution is reduced to matrix
      // multiplication.
      out_rows = input_rows;
      out_cols = input_cols;
    } else {
      OP_REQUIRES_OK(
          context,
          Get2dOutputSize(input_rows, input_cols, filter_rows, filter_cols,
                          stride_rows, stride_cols, padding_, &out_rows,
                          &out_cols, &pad_rows, &pad_cols));
    }
    TensorShape out_shape =
        ShapeFromFormat(data_format_, batch, out_rows, out_cols, out_depth);

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    VLOG(2) << "Conv2D: in_depth = " << in_depth
            << ", input_cols = " << input_cols
            << ", filter_cols = " << filter_cols
            << ", input_rows = " << input_rows
            << ", filter_rows = " << filter_rows
            << ", stride_rows = " << stride_rows
            << ", stride_cols = " << stride_cols
            << ", out_depth = " << out_depth;

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }
    LaunchConvOp<Device, T>::launch(
        context, use_cudnn_, input, filter, stride_rows, stride_cols,
        BrainPadding2EigenPadding(padding_), output, data_format_);
  }

 private:
  std::vector<int32> strides_;
  bool use_cudnn_;
  Padding padding_;
  TensorFormat data_format_;

  TF_DISALLOW_COPY_AND_ASSIGN(Conv2DOp);
};

REGISTER_KERNEL_BUILDER(
    Name("Conv2D").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    Conv2DOp<CPUDevice, float>);

#if GOOGLE_CUDA

int64 GetCudnnWorkspaceLimit(const string& envvar_in_mb,
                             int64 default_value_in_bytes) {
  const char* workspace_limit_in_mb_str = getenv(envvar_in_mb.c_str());
  if (workspace_limit_in_mb_str != nullptr &&
      strcmp(workspace_limit_in_mb_str, "") != 0) {
    int64 scratch_limit_in_mb = -1;
    if (strings::safe_strto64(workspace_limit_in_mb_str,
                              &scratch_limit_in_mb)) {
      return scratch_limit_in_mb * (1 << 20);
    } else {
      LOG(WARNING) << "Invalid value for env-var " << envvar_in_mb << ": "
                   << workspace_limit_in_mb_str;
    }
  }
  return default_value_in_bytes;
}

template <typename T>
struct LaunchConvOp<GPUDevice, T> {
  static void launch(OpKernelContext* ctx, bool use_cudnn,
                     const Tensor& input_param, const Tensor& filter,
                     int row_stride, int col_stride,
                     const Eigen::PaddingType& padding, Tensor* output,
                     TensorFormat data_format) {
    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

    if (use_cudnn) {
      Tensor input = input_param;
      if (filter.dim_size(0) == 1 && filter.dim_size(1) == 1 &&
          row_stride == 1 && col_stride == 1 &&
          data_format == FORMAT_NHWC) {
        // 1x1 filter, so call cublas directly.
        const uint64 m =
            input.dim_size(0) * input.dim_size(1) * input.dim_size(2);
        const uint64 k = filter.dim_size(2);
        const uint64 n = filter.dim_size(3);

        auto a_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                    input.template flat<T>().size());
        auto b_ptr = AsDeviceMemory(filter.template flat<T>().data(),
                                    filter.template flat<T>().size());
        auto c_ptr = AsDeviceMemory(output->template flat<T>().data(),
                                    output->template flat<T>().size());

        auto no_transpose = perftools::gputools::blas::Transpose::kNoTranspose;
        bool blas_launch_status =
            stream
                ->ThenBlasGemm(no_transpose, no_transpose, n, m, k, 1.0f, b_ptr,
                               n, a_ptr, k, 0.0f, &c_ptr, n)
                .ok();
        if (!blas_launch_status) {
          ctx->SetStatus(errors::Internal("Blas SGEMM launch failed : m=", m,
                                          ", n=", n, ", k=", k));
        }
        return;
      }
      int padding_rows = 0;
      int padding_cols = 0;
      const int64 in_batch = GetTensorDim(input, data_format, 'N');
      int64 in_rows = GetTensorDim(input, data_format, 'H');
      int64 in_cols = GetTensorDim(input, data_format, 'W');
      const int64 in_depths = GetTensorDim(input, data_format, 'C');
      const int64 out_batch = GetTensorDim(*output, data_format, 'N');
      const int64 out_rows = GetTensorDim(*output, data_format, 'H');
      const int64 out_cols = GetTensorDim(*output, data_format, 'W');
      const int64 out_depths = GetTensorDim(*output, data_format, 'C');
      const int64 patch_rows = filter.dim_size(0);
      const int64 patch_cols = filter.dim_size(1);
      if (padding == Eigen::PADDING_SAME) {
        // Total padding on rows and cols is
        // Pr = (R' - 1) * S + Kr - R
        // Pc = (C' - 1) * S + Kc - C
        // where (R', C') are output dimensions, (R, C) are input dimensions, S
        // is stride, (Kr, Kc) are filter dimensions.
        // We pad Pr/2 on the left and Pr - Pr/2 on the right, Pc/2 on the top
        // and Pc - Pc/2 on the bottom.  When Pr or Pc is odd, this means
        // we pad more on the right and bottom than on the top and left.
        padding_rows = (out_rows - 1) * row_stride + patch_rows - in_rows;
        padding_cols = (out_cols - 1) * col_stride + patch_cols - in_cols;
        const bool rows_odd = (padding_rows % 2 != 0);
        const bool cols_odd = (padding_cols % 2 != 0);
        if (rows_odd || cols_odd) {
          Tensor transformed_input;
          int64 new_in_rows = in_rows + rows_odd;
          int64 new_in_cols = in_cols + cols_odd;
          OP_REQUIRES_OK(ctx,
                         ctx->allocate_temp(
                             DataTypeToEnum<T>::value,
                             ShapeFromFormat(data_format, in_batch, new_in_rows,
                                             new_in_cols, in_depths),
                             &transformed_input));

          functor::PadInput<GPUDevice, T, int>()(
              ctx->eigen_device<GPUDevice>(),
              To32Bit(input_param.tensor<T, 4>()), 0, rows_odd, 0, cols_odd,
              To32Bit(transformed_input.tensor<T, 4>()), data_format);
          input = transformed_input;
          in_rows = new_in_rows;
          in_cols = new_in_cols;
        }
      }

      if (data_format == FORMAT_NHWC) {
        // Convert the input tensor from NHWC to NCHW.
        Tensor transformed_input;
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                                DataTypeToEnum<T>::value,
                                ShapeFromFormat(FORMAT_NCHW, in_batch, in_rows,
                                                in_cols, in_depths),
                                &transformed_input));
        functor::NHWCToNCHW<GPUDevice, T>()(
            ctx->eigen_device<GPUDevice>(),
            const_cast<const Tensor&>(input).tensor<T, 4>(),
            transformed_input.tensor<T, 4>());
        input = transformed_input;
      }

      perftools::gputools::dnn::BatchDescriptor input_desc;
      input_desc.set_count(in_batch)
          .set_feature_map_count(in_depths)
          .set_height(in_rows)
          .set_width(in_cols)
          .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);
      perftools::gputools::dnn::BatchDescriptor output_desc;
      output_desc.set_count(out_batch)
          .set_height(out_rows)
          .set_width(out_cols)
          .set_feature_map_count(out_depths)
          .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);
      perftools::gputools::dnn::FilterDescriptor filter_desc;
      filter_desc.set_input_filter_height(filter.dim_size(0))
          .set_input_filter_width(filter.dim_size(1))
          .set_input_feature_map_count(filter.dim_size(2))
          .set_output_feature_map_count(filter.dim_size(3));
      perftools::gputools::dnn::ConvolutionDescriptor conv_desc;
      conv_desc.set_vertical_filter_stride(row_stride)
          .set_horizontal_filter_stride(col_stride)
          .set_zero_padding_height(padding_rows / 2)
          .set_zero_padding_width(padding_cols / 2);

      Tensor transformed_filter;
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_temp(
                         DataTypeToEnum<T>::value,
                         TensorShape({filter.dim_size(3), filter.dim_size(2),
                                      filter.dim_size(0), filter.dim_size(1)}),
                         &transformed_filter));

      functor::TransformFilter<GPUDevice, T, int>()(
          ctx->eigen_device<GPUDevice>(), To32Bit(filter.tensor<T, 4>()),
          To32Bit(transformed_filter.tensor<T, 4>()));

      Tensor transformed_output;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                              DataTypeToEnum<T>::value,
                              ShapeFromFormat(FORMAT_NCHW, out_batch, out_rows,
                                              out_cols, out_depths),
                              &transformed_output));

      auto input_ptr = AsDeviceMemory(input.template flat<T>().data(),
                                      input.template flat<T>().size());
      auto filter_ptr =
          AsDeviceMemory(transformed_filter.template flat<T>().data(),
                         transformed_filter.template flat<T>().size());
      auto output_ptr =
          AsDeviceMemory(transformed_output.template flat<T>().data(),
                         transformed_output.template flat<T>().size());

      static int64 ConvolveScratchSize = GetCudnnWorkspaceLimit(
          "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB by default
          );
      CudnnScratchAllocator scratch_allocator(ConvolveScratchSize, ctx);
      bool cudnn_launch_status =
          stream
              ->ThenConvolveWithScratch(input_desc, input_ptr, filter_desc,
                                        filter_ptr, conv_desc, output_desc,
                                        &output_ptr, &scratch_allocator)
              .ok();

      if (!cudnn_launch_status) {
        ctx->SetStatus(errors::Internal(
            "cuDNN launch failure : input shape(", input.shape().DebugString(),
            ") filter shape(", filter.shape().DebugString(), ")"));
      }

      // Convert the output tensor back from NHWC to NCHW.
      if (data_format == FORMAT_NHWC) {
        functor::NCHWToNHWC<GPUDevice, T>()(
            ctx->eigen_device<GPUDevice>(),
            const_cast<const Tensor&>(transformed_output).tensor<T, 4>(),
            output->tensor<T, 4>());
      } else {
        *output = transformed_output;
      }
    } else {
      LaunchGeneric<GPUDevice, T>::launch(ctx, input_param, filter, row_stride,
                                          col_stride, padding, output,
                                          data_format);
    }
  }
};

#endif  // GOOGLE_CUDA

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                  \
  template <>                                                                \
  void SpatialConvolution<GPUDevice, T>::operator()(                         \
      const GPUDevice& d, typename TTypes<T, 4>::Tensor output,              \
      typename TTypes<T, 4>::ConstTensor input,                              \
      typename TTypes<T, 4>::ConstTensor filter, int row_stride,             \
      int col_stride, const Eigen::PaddingType& padding);                    \
  extern template struct SpatialConvolution<GPUDevice, T>;                   \
  template <>                                                                \
  void MatMulConvFunctor<GPUDevice, T>::operator()(                          \
      const GPUDevice& d, typename TTypes<T, 2>::Tensor out,                 \
      typename TTypes<T, 2>::ConstTensor in0,                                \
      typename TTypes<T, 2>::ConstTensor in1,                                \
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair); \
  extern template struct MatMulConvFunctor<GPUDevice, T>;                    \
  template <>                                                                \
  void TransformFilter<GPUDevice, T, int>::operator()(                       \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor in,        \
      typename TTypes<T, 4, int>::Tensor out);                               \
  extern template struct TransformFilter<GPUDevice, T, int>;                 \
  template <>                                                                \
  void PadInput<GPUDevice, T, int>::operator()(                              \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor in,        \
      int padding_rows_left, int padding_rows_right, int padding_cols_left,  \
      int padding_cols_right, typename TTypes<T, 4, int>::Tensor out,        \
      TensorFormat data_format);                                             \
  extern template struct PadInput<GPUDevice, T, int>

DECLARE_GPU_SPEC(float);
#undef DECLARE_GPU_SPEC
}  // namespace functor

// Registration of the GPU implementations.
REGISTER_KERNEL_BUILDER(
    Name("Conv2D").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    Conv2DOp<GPUDevice, float>);

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
