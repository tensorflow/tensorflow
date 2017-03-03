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

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/conv_3d.h"

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
using perftools::gputools::dnn::DimIndex;
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
struct LaunchConvOp;

template <typename T>
struct LaunchConvOp<CPUDevice, T> {
  static void launch(OpKernelContext* context, const Tensor& input,
                     const Tensor& filter, const std::array<int64, 3>& strides,
                     const Padding padding, Tensor* output) {
    functor::CuboidConvolution<CPUDevice, T>()(
        context->eigen_device<CPUDevice>(), output->tensor<T, 5>(),
        input.tensor<T, 5>(), filter.tensor<T, 5>(), strides[2], strides[1],
        strides[0], BrainPadding2EigenPadding(padding));
  }
};

template <typename Device, typename T>
class Conv3DOp : public BinaryOp<T> {
 public:
  explicit Conv3DOp(OpKernelConstruction* context) : BinaryOp<T>(context) {
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(
        context, (stride_[0] == 1 && stride_[4] == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }

  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_z, in_y, in_x, in_channels ]
    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_z, filter_y, filter_x, in_channels, out_channels]
    const Tensor& filter = context->input(1);

    // NOTE: The ordering of the spatial dimensions is arbitrary, but has to be
    // kept consistent between input/filter/output.
    OP_REQUIRES(context, input.dims() == 5,
                errors::InvalidArgument("input must be 5-dimensional"));
    OP_REQUIRES(context, filter.dims() == 5,
                errors::InvalidArgument("filter must be 5-dimensional"));

    const int64 in_depth = input.dim_size(4);
    const int64 in_batch = input.dim_size(0);

    const int64 out_depth = filter.dim_size(4);
    OP_REQUIRES(
        context, in_depth == filter.dim_size(3),
        errors::InvalidArgument("input and filter must have the same depth"));

    // Dimension order for these arrays is: z, y, x.
    std::array<int64, 3> input_size = {
        {input.dim_size(1), input.dim_size(2), input.dim_size(3)}};
    std::array<int64, 3> filter_size = {
        {filter.dim_size(0), filter.dim_size(1), filter.dim_size(2)}};
    std::array<int64, 3> strides = {{stride_[1], stride_[2], stride_[3]}};
    std::array<int64, 3> out, padding;

    OP_REQUIRES_OK(context, Get3dOutputSize(input_size, filter_size, strides,
                                            padding_, &out, &padding));

    TensorShape out_shape = {in_batch, out[0], out[1], out[2], out_depth};
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    // Return early if nothing to do.
    if (out_shape.num_elements() == 0) return;

    LaunchConvOp<Device, T>::launch(context, input, filter, strides, padding_,
                                    output);
  }

 private:
  std::vector<int32> stride_;
  Padding padding_;
};

#define REGISTER_CPU_KERNEL(T)                                  \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("Conv3D").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      Conv3DOp<CPUDevice, T>);
TF_CALL_float(REGISTER_CPU_KERNEL);
TF_CALL_double(REGISTER_CPU_KERNEL);
#undef REGISTER_CPU_KERNEL

#if GOOGLE_CUDA

// TODO(mjanusz): Share logic with 2d implementation as much as possible.
template <typename T>
struct LaunchConvOp<GPUDevice, T> {
  static void launch(OpKernelContext* ctx, const Tensor& input_param,
                     const Tensor& filter, const std::array<int64, 3>& strides,
                     const Padding padding, Tensor* output) {
    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));

    Tensor input = input_param;

    const int64 in_batch = input.dim_size(0);
    int64 in_planes = input.dim_size(1);
    int64 in_rows = input.dim_size(2);
    int64 in_cols = input.dim_size(3);
    const int64 in_depth = input.dim_size(4);

    const int64 filter_planes = filter.dim_size(0);
    const int64 filter_rows = filter.dim_size(1);
    const int64 filter_cols = filter.dim_size(2);
    const int64 out_depth = filter.dim_size(4);

    int64 pad_planes = 0, pad_rows = 0, pad_cols = 0;
    int64 out_planes = output->dim_size(1);
    int64 out_rows = output->dim_size(2);
    int64 out_cols = output->dim_size(3);

    if (padding == Padding::SAME) {
      pad_planes = std::max<int64>(
          0, (out_planes - 1) * strides[0] + filter_planes - in_planes);
      pad_rows = std::max<int64>(
          0, (out_rows - 1) * strides[1] + filter_rows - in_rows);
      pad_cols = std::max<int64>(
          0, (out_cols - 1) * strides[2] + filter_cols - in_cols);
    }

    // NOTE: This only works in NHWC.
    if (filter_planes == 1 && filter_rows == 1 && filter_cols == 1 &&
        strides[0] == 1 && strides[1] == 1 && strides[2] == 1) {
      // 1x1 filter, so call cublas directly.
      const uint64 m = in_batch * in_planes * in_rows * in_cols;
      const uint64 k = in_depth;
      const uint64 n = out_depth;

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
    } else if (filter_planes == in_planes && filter_rows == in_rows &&
               filter_cols == in_cols && padding == Padding::VALID) {
      // The input data and filter have the same planes/height/width, so call
      // cublas directly.
      const uint64 m = in_batch;
      const uint64 k = in_planes * in_rows * in_cols * in_depth;
      const uint64 n = out_depth;

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

    if (padding == Padding::SAME) {
      const bool rows_odd = (pad_rows % 2 != 0);
      const bool cols_odd = (pad_cols % 2 != 0);
      const bool planes_odd = (pad_planes % 2 != 0);

      // Necessary because cuDNN only supports symmetric padding.
      // TODO(mjanusz): Consider making this optional? This would save some
      // overhead and would work as long as an op trained this way is only
      // used on GPU.
      if (rows_odd || cols_odd || planes_odd) {
        Tensor transformed_input;
        int64 new_in_rows = in_rows + rows_odd;
        int64 new_in_cols = in_cols + cols_odd;
        int64 new_in_planes = in_planes + planes_odd;

        TensorShape transformed_shape(
            {in_batch, new_in_planes, new_in_rows, new_in_cols, in_depth});
        OP_REQUIRES_OK(
            ctx, ctx->allocate_temp(DataTypeToEnum<T>::value, transformed_shape,
                                    &transformed_input));

        functor::PadInput<GPUDevice, T, int, 5>()(
            ctx->eigen_device<GPUDevice>(), To32Bit(input_param.tensor<T, 5>()),
            {{0, 0, 0}}, {{planes_odd, rows_odd, cols_odd}},
            To32Bit(transformed_input.tensor<T, 5>()), FORMAT_NHWC);
        input = transformed_input;
        in_rows = new_in_rows;
        in_cols = new_in_cols;
        in_planes = new_in_planes;
      }
    }

    Tensor transformed_input;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(
                 DataTypeToEnum<T>::value,
                 TensorShape({in_batch, in_depth, in_planes, in_rows, in_cols}),
                 &transformed_input));
    // input: [b, x, y, z, d]
    // t_input: [b, d, x, y, z]
    // NCDHW is the only format universally supported by cuDNN.
    functor::NHWCToNCHW<GPUDevice, T, 5>()(
        ctx->eigen_device<GPUDevice>(),
        const_cast<const Tensor&>(input).tensor<T, 5>(),
        transformed_input.tensor<T, 5>());
    input = transformed_input;

    CHECK(pad_rows >= 0 && pad_cols >= 0 && pad_planes >= 0)
        << "Negative paddings: (" << pad_rows << ", " << pad_cols << ", "
        << pad_planes << ")";
    perftools::gputools::dnn::BatchDescriptor input_desc(3);
    input_desc.set_count(in_batch)
        .set_feature_map_count(in_depth)
        .set_spatial_dim(DimIndex::X, in_cols)
        .set_spatial_dim(DimIndex::Y, in_rows)
        .set_spatial_dim(DimIndex::Z, in_planes)
        .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);
    perftools::gputools::dnn::BatchDescriptor output_desc(3);
    output_desc.set_count(in_batch)
        .set_spatial_dim(DimIndex::X, out_cols)
        .set_spatial_dim(DimIndex::Y, out_rows)
        .set_spatial_dim(DimIndex::Z, out_planes)
        .set_feature_map_count(out_depth)
        .set_layout(perftools::gputools::dnn::DataLayout::kBatchDepthYX);
    perftools::gputools::dnn::FilterDescriptor filter_desc(3);
    filter_desc.set_spatial_dim(DimIndex::X, filter_cols)
        .set_spatial_dim(DimIndex::Y, filter_rows)
        .set_spatial_dim(DimIndex::Z, filter_planes)
        .set_input_feature_map_count(in_depth)
        .set_output_feature_map_count(out_depth);
    perftools::gputools::dnn::ConvolutionDescriptor conv_desc(3);
    conv_desc.set_filter_stride(DimIndex::X, strides[2])
        .set_filter_stride(DimIndex::Y, strides[1])
        .set_filter_stride(DimIndex::Z, strides[0])
        .set_zero_padding(DimIndex::X, pad_cols / 2)
        .set_zero_padding(DimIndex::Y, pad_rows / 2)
        .set_zero_padding(DimIndex::Z, pad_planes / 2);

    Tensor transformed_filter;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                TensorShape({out_depth, in_depth, filter_planes,
                                             filter_rows, filter_cols}),
                                &transformed_filter));
    // filter: [x, y, z, in, out]
    // t_filter: [out, in, x, y, z]
    functor::TransformFilter<GPUDevice, T, int, 5>()(
        ctx->eigen_device<GPUDevice>(), To32Bit(filter.tensor<T, 5>()),
        To32Bit(transformed_filter.tensor<T, 5>()));

    Tensor transformed_output;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                TensorShape({in_batch, out_depth, out_planes,
                                             out_rows, out_cols}),
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
        "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32);  // 4GB by default
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

    // t_output: [b, out, x, y, z]
    // output: [b, x, y, z, out]
    functor::NCHWToNHWC<GPUDevice, T, 5>()(
        ctx->eigen_device<GPUDevice>(),
        const_cast<const Tensor&>(transformed_output).tensor<T, 5>(),
        output->tensor<T, 5>());
  }
};

// Forward declarations of the functor specializations for GPU.
// This ensures that the custom implementation is used instead of the default
// Eigen one (which is used for CPU).
namespace functor {
#define DECLARE_GPU_SPEC(T)                                           \
  template <>                                                         \
  void TransformFilter<GPUDevice, T, int, 5>::operator()(             \
      const GPUDevice& d, typename TTypes<T, 5, int>::ConstTensor in, \
      typename TTypes<T, 5, int>::Tensor out);                        \
  template <>                                                         \
  void ReverseTransformFilter<GPUDevice, T, 5>::operator()(           \
      const GPUDevice& d, typename TTypes<T, 5>::ConstTensor in,      \
      typename TTypes<T, 5>::Tensor out);                             \
  template <>                                                         \
  void PadInput<GPUDevice, T, int, 5>::operator()(                    \
      const GPUDevice& d, typename TTypes<T, 5, int>::ConstTensor in, \
      const std::array<int, 3>& padding_left,                         \
      const std::array<int, 3>& padding_right,                        \
      typename TTypes<T, 5, int>::Tensor out, TensorFormat format);

DECLARE_GPU_SPEC(float);
#undef DECLARE_GPU_SPEC

}  // namespace functor

// Registration of the GPU implementations.
REGISTER_KERNEL_BUILDER(
    Name("Conv3D").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    Conv3DOp<GPUDevice, float>);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
