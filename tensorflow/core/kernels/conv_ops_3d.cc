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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <vector>

#include "tensorflow/core/framework/kernel_shape_util.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/kernels/conv_2d.h"
#include "tensorflow/core/kernels/conv_3d.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/kernels/conv_ops_impl.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/profiler/lib/scoped_annotation.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/use_cudnn.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/cast_op.h"
#include "tensorflow/core/kernels/numeric_options_utils.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/protobuf/autotuning.pb.h"
#include "tensorflow/core/util/autotune_maps/conv_parameters.h"
#include "tensorflow/core/util/proto/proto_utils.h"
using stream_executor::dnn::DimIndex;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#if GOOGLE_CUDA
#include "third_party/gpus/cudnn/cudnn.h"
#include "xla/stream_executor/gpu/asm_compiler.h"
#include "xla/stream_executor/gpu/redzone_allocator.h"
#include "xla/stream_executor/integrations/tf_allocator_adapter.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class Conv3DOp : public BinaryOp<T> {
 public:
  explicit Conv3DOp(OpKernelConstruction* context) : BinaryOp<T>(context) {
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
    OP_REQUIRES(context, stride_.size() == 5,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(
        context,
        (GetTensorDim(stride_, data_format_, 'N') == 1 &&
         GetTensorDim(stride_, data_format_, 'C') == 1),
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES(
        context,
        (GetTensorDim(stride_, data_format_, '0') > 0 &&
         GetTensorDim(stride_, data_format_, '1') > 0 &&
         GetTensorDim(stride_, data_format_, '2') > 0),
        errors::InvalidArgument("Spatial strides should be larger than 0."));
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilation_));
    OP_REQUIRES(context, dilation_.size() == 5,
                errors::InvalidArgument("Dilation rates field must "
                                        "specify 5 dimensions"));
    OP_REQUIRES(context,
                (GetTensorDim(dilation_, data_format_, 'N') == 1 &&
                 GetTensorDim(dilation_, data_format_, 'C') == 1),
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilation rates in the batch and depth dimensions."));
    OP_REQUIRES(
        context,
        (GetTensorDim(dilation_, data_format_, '0') > 0 &&
         GetTensorDim(dilation_, data_format_, '1') > 0 &&
         GetTensorDim(dilation_, data_format_, '2') > 0),
        errors::InvalidArgument("Dilated rates should be larger than 0."));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    cudnn_use_autotune_ = CudnnUseAutotune();
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

    const int64_t in_depth = GetTensorDim(input, data_format_, 'C');
    const int64_t in_batch = GetTensorDim(input, data_format_, 'N');

    const int64_t filter_depth = filter.dim_size(3);
    const int64_t out_depth = filter.dim_size(4);

    OP_REQUIRES(context, filter_depth != 0,
                errors::InvalidArgument("filter_depth must be non-zero"));
    OP_REQUIRES(context, in_depth % filter_depth == 0,
                errors::InvalidArgument(
                    "Input depth must be evenly divisible by filter depth: ",
                    in_depth, " vs ", filter_depth));
    OP_REQUIRES(
        context, filter.NumElements() > 0,
        errors::InvalidArgument("filter must not have zero elements "
                                "(i.e. all dimensions must be non-zero)"));

    // Dimension order for these arrays is: z, y, x.
    std::array<int64_t, 3> input_size = {
        {GetTensorDim(input, data_format_, '0'),
         GetTensorDim(input, data_format_, '1'),
         GetTensorDim(input, data_format_, '2')}};
    std::array<int64_t, 3> filter_size = {
        {filter.dim_size(0), filter.dim_size(1), filter.dim_size(2)}};
    std::array<int64_t, 3> dilations = {
        {GetTensorDim(dilation_, data_format_, '0'),
         GetTensorDim(dilation_, data_format_, '1'),
         GetTensorDim(dilation_, data_format_, '2')}};
    std::array<int64_t, 3> strides = {
        {GetTensorDim(stride_, data_format_, '0'),
         GetTensorDim(stride_, data_format_, '1'),
         GetTensorDim(stride_, data_format_, '2')}};
    std::array<int64_t, 3> out, padding;

    OP_REQUIRES_OK(
        context, Get3dOutputSizeV2(input_size, filter_size, dilations, strides,
                                   padding_, &out, &padding));
    TensorShape out_shape;
    OP_REQUIRES_OK(context,
                   ShapeFromFormatWithStatus(data_format_, in_batch,
                                             {{out[0], out[1], out[2]}},
                                             out_depth, &out_shape));
    Tensor* output;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    // Return early if nothing to do.
    if (out_shape.num_elements() == 0) return;

    LaunchConv3DOp<Device, T>::launch(context, cudnn_use_autotune_, input,
                                      filter, dilations, strides, padding_,
                                      data_format_, output);
  }

 private:
  std::vector<int32> dilation_;
  std::vector<int32> stride_;
  Padding padding_;
  TensorFormat data_format_;
  bool cudnn_use_autotune_;
};

#define REGISTER_CPU_KERNEL(T)                                  \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("Conv3D").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      Conv3DOp<CPUDevice, T>);
TF_CALL_half(REGISTER_CPU_KERNEL);
TF_CALL_float(REGISTER_CPU_KERNEL);
TF_CALL_double(REGISTER_CPU_KERNEL);
TF_CALL_bfloat16(REGISTER_CPU_KERNEL);
#undef REGISTER_CPU_KERNEL

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <typename T>
struct LaunchConv3DOp<GPUDevice, T> {
  static void launch(OpKernelContext* ctx, bool cudnn_use_autotune,
                     const Tensor& input_param, const Tensor& filter,
                     const std::array<int64, 3>& dilations,
                     const std::array<int64, 3>& strides, const Padding padding,
                     TensorFormat data_format, Tensor* output) {
    // Empty explicit paddings.
    std::vector<int64_t> explicit_paddings;
    // Cast strides and dilations.
    gtl::InlinedVector<int64_t, 3> casted_strides(strides.begin(),
                                                  strides.end());
    gtl::InlinedVector<int64_t, 3> casted_dilations(dilations.begin(),
                                                    dilations.end());
    LaunchConvOpImpl<T>(ctx, cudnn_use_autotune, input_param, filter,
                        casted_dilations, casted_strides, padding,
                        explicit_paddings, data_format, output);
  }
};

template <>
struct LaunchConv3DOp<GPUDevice, Eigen::bfloat16> {
  static void launch(OpKernelContext* ctx, bool cudnn_use_autotune,
                     const Tensor& input_param, const Tensor& filter,
                     const std::array<int64, 3>& dilations,
                     const std::array<int64, 3>& strides, const Padding padding,
                     TensorFormat data_format, Tensor* output) {
    // Empty explicit paddings.
    std::vector<int64_t> explicit_paddings;
    // Cast strides and dilations.
    gtl::InlinedVector<int64_t, 3> casted_strides(strides.begin(),
                                                  strides.end());
    gtl::InlinedVector<int64_t, 3> casted_dilations(dilations.begin(),
                                                    dilations.end());

    auto* stream = ctx->op_device_context()->stream();
    const bool cast_to_float = !IsBF16SupportedInOps(stream);

    if (cast_to_float) {
      Tensor casted_input = input_param;
      Tensor casted_filter = filter;
      Tensor casted_out = *output;

      const GPUDevice& device = ctx->eigen_device<GPUDevice>();
      functor::CastFunctor<GPUDevice, float, Eigen::bfloat16> cast;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, input_param.shape(),
                                             &casted_input));
      cast(device, casted_input.template flat<float>(),
           input_param.template flat<Eigen::bfloat16>());

      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DT_FLOAT, filter.shape(), &casted_filter));
      cast(device, casted_filter.template flat<float>(),
           filter.template flat<Eigen::bfloat16>());

      OP_REQUIRES_OK(
          ctx, ctx->allocate_temp(DT_FLOAT, output->shape(), &casted_out));

      LaunchConvOpImpl<float>(ctx, cudnn_use_autotune, casted_input,
                              casted_filter, casted_dilations, casted_strides,
                              padding, explicit_paddings, data_format,
                              &casted_out);

      functor::CastFunctor<GPUDevice, Eigen::bfloat16, float> cast_back;
      const Tensor& casted_out_const = casted_out;
      cast_back(device, output->template flat<Eigen::bfloat16>(),
                casted_out_const.template flat<float>());
      return;
    }

    LaunchConvOpImpl<Eigen::bfloat16>(
        ctx, cudnn_use_autotune, input_param, filter, casted_dilations,
        casted_strides, padding, explicit_paddings, data_format, output);
  }
};

// Forward declarations of the functor specializations for GPU.
// This ensures that the custom implementation is used instead of the default
// Eigen one (which is used for CPU).
namespace functor {
#define DECLARE_GPU_SPEC(T)                                             \
  template <>                                                           \
  void TransformFilter<GPUDevice, T, int, 4>::operator()(               \
      const GPUDevice& d, FilterTensorFormat dst_filter_format,         \
      typename TTypes<T, 4, int>::ConstTensor in,                       \
      typename TTypes<T, 4, int>::Tensor out);                          \
  template <>                                                           \
  void PadInput<GPUDevice, T, int, 4>::operator()(                      \
      const GPUDevice& d, typename TTypes<T, 4, int>::ConstTensor in,   \
      const std::array<int, 2>& padding_left,                           \
      const std::array<int, 2>& padding_right,                          \
      typename TTypes<T, 4, int>::Tensor out, TensorFormat data_format, \
      const T& padding_value);                                          \
  template <>                                                           \
  void TransformFilter<GPUDevice, T, int, 5>::operator()(               \
      const GPUDevice& d, FilterTensorFormat dst_filter_format,         \
      typename TTypes<T, 5, int>::ConstTensor in,                       \
      typename TTypes<T, 5, int>::Tensor out);                          \
  template <>                                                           \
  void ReverseTransformFilter<GPUDevice, T, 5>::operator()(             \
      const GPUDevice& d, FilterTensorFormat src_filter_format,         \
      typename TTypes<T, 5>::ConstTensor in,                            \
      typename TTypes<T, 5>::Tensor out);                               \
  template <>                                                           \
  void PadInput<GPUDevice, T, int, 5>::operator()(                      \
      const GPUDevice& d, typename TTypes<T, 5, int>::ConstTensor in,   \
      const std::array<int, 3>& padding_left,                           \
      const std::array<int, 3>& padding_right,                          \
      typename TTypes<T, 5, int>::Tensor out, TensorFormat format,      \
      const T& padding_value);                                          \
  template <>                                                           \
  void NHWCToNCHW<GPUDevice, T, 5>::operator()(                         \
      const GPUDevice& d, typename TTypes<T, 5>::ConstTensor in,        \
      typename TTypes<T, 5>::Tensor out);                               \
  template <>                                                           \
  void NCHWToNHWC<GPUDevice, T, 5>::operator()(                         \
      const GPUDevice& d, typename TTypes<T, 5>::ConstTensor in,        \
      typename TTypes<T, 5>::Tensor out);

DECLARE_GPU_SPEC(Eigen::half);
DECLARE_GPU_SPEC(Eigen::bfloat16);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC

}  // namespace functor

// Registration of the GPU implementations.
REGISTER_KERNEL_BUILDER(
    Name("Conv3D").Device(DEVICE_GPU).TypeConstraint<Eigen::half>("T"),
    Conv3DOp<GPUDevice, Eigen::half>);
REGISTER_KERNEL_BUILDER(
    Name("Conv3D").Device(DEVICE_GPU).TypeConstraint<Eigen::bfloat16>("T"),
    Conv3DOp<GPUDevice, Eigen::bfloat16>);
REGISTER_KERNEL_BUILDER(
    Name("Conv3D").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    Conv3DOp<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(
    Name("Conv3D").Device(DEVICE_GPU).TypeConstraint<double>("T"),
    Conv3DOp<GPUDevice, double>);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
