/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <type_traits>

// This include can't be in the conv_ops_fused_impl.h headers. See b/62899350.
#if GOOGLE_CUDA
#include "tensorflow/core/kernels/numeric_options_utils.h"
#include "tensorflow/core/protobuf/autotuning.pb.h"
#endif  // GOOGLE_CUDA
#include "tensorflow/core/kernels/autotune_conv_impl.h"
#include "tensorflow/core/kernels/conv_ops_fused_impl.h"
#include "tensorflow/core/kernels/cwise_ops.h"
#include "tensorflow/core/util/activation_mode.h"

namespace tensorflow {

// If we're using the alternative GEMM-based implementation of Conv2D for the
// CPU implementation, don't register this EigenTensor-based version.
#if !defined(USE_GEMM_FOR_CONV)
TF_CALL_int8(REGISTER_FUSED_CPU_CONV2D);
TF_CALL_qint8(REGISTER_FUSED_CPU_CONV2D);
#endif  // !USE_GEMM_FOR_CONV

#if GOOGLE_CUDA

namespace functor {
DECLARE_FUNCTOR_GPU_SPEC(int32);
}  // namespace functor

TF_CALL_int8(REGISTER_FUSED_GPU_CONV2D);
TF_CALL_qint8(REGISTER_FUSED_GPU_CONV2D);

#endif  // GOOGLE_CUDA

template <typename T>
struct LaunchFusedConv2DOpCpuInt8Helper {
  using BiasType = float;
  using ScaleType = float;
  using ComputeT = float;  // convert inputs to fp32 for tensor contraction
  using TempT = float;     // temporary accumulator type for tensor contraction

  void operator()(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
                  const Tensor& conv_input, const Tensor& filter,
                  const FusedComputationType fusion,
                  const FusedComputationArgs& fusion_args,
                  const Conv2DParameters& params,
                  const Conv2DDimensions& dimensions, Tensor* output) {
    OP_REQUIRES(ctx, dimensions.in_depth == filter.dim_size(2),
                errors::Unimplemented("Fused conv implementation does not "
                                      "support grouped convolutions for now."));
    OP_REQUIRES(
        ctx, params.data_format == FORMAT_NHWC,
        errors::Unimplemented(
            "Fused conv implementation for int8/qint8 on CPU only supports "
            "NHWC tensor format for now."));
    OP_REQUIRES(ctx,
                DataTypeToEnum<T>::value == DT_INT8 ||
                    DataTypeToEnum<T>::value == DT_QINT8,
                errors::Unimplemented("Specialized fused conv implemented for "
                                      "only int8 and qint8 on CPU."));
    OP_REQUIRES(
        ctx, dimensions.dilation_rows == 1 && dimensions.dilation_cols == 1,
        errors::Unimplemented(
            "Fused conv implementation for int8/qint8 on CPU only supports "
            "dilation of 1 for rows and cols."));
    OP_REQUIRES(
        ctx,
        fusion == FusedComputationType::kBiasAdd ||
            fusion == FusedComputationType::kBiasAddWithRelu,
        errors::Unimplemented(
            "Fused conv implementation for int8/qint8 on CPU only supports "
            "BiasAdd + None or BiasAdd + Relu."));

    constexpr int kBias = 2;
    constexpr int kSideInput = 3;
    constexpr int kConvInputScale = 4;
    constexpr int kSideInputScale = 5;

    const Tensor& bias = ctx->input(kBias);
    const Tensor& side_input = ctx->input(kSideInput);
    const Tensor& conv_input_scale = ctx->input(kConvInputScale);
    const Tensor& side_input_scale_param = ctx->input(kSideInputScale);

    Eigen::PaddingType padding = BrainPadding2EigenPadding(params.padding);
    int32_t row_stride = dimensions.stride_rows;
    int32_t col_stride = dimensions.stride_cols;

    // Output tensor has type T (QInt8/int8), but we can only evaluate
    // Tensor contraction using 32-bit accumulation (fp32).
    Tensor temp_output(DataTypeToEnum<TempT>::value, output->shape());

    const int32_t row_dilation = dimensions.dilation_rows;
    const int32_t col_dilation = dimensions.dilation_cols;

    auto& device = ctx->eigen_device<CPUDevice>();

    // CPU convolution works with input in NHWC and filter in HWIO data formats.
    // NOTE: This code is mostly shared with 'Conv2D' and 'FusedConv2D'.

    const ScaleType side_input_scale =
        side_input_scale_param.scalar<ScaleType>()();
    BiasActivationOutputKernel output_kernel(
        conv_input_scale, side_input, side_input_scale, bias, fusion, output);

    if (filter.dim_size(0) == 1 && filter.dim_size(1) == 1 && row_stride == 1 &&
        col_stride == 1) {
      int conv_width =  // Width for the convolution step.
          output->dim_size(0) * output->dim_size(1) * output->dim_size(2);

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);

      auto out = temp_output.shaped<TempT, 2>({conv_width, filter.dim_size(3)});
      auto in0 = conv_input.shaped<T, 2>({conv_width, filter.dim_size(2)});
      auto in1 = filter.shaped<T, 2>({filter.dim_size(2), filter.dim_size(3)});

      out.device(device) = in0.template cast<ComputeT>().contract(
          in1.template cast<ComputeT>(), dim_pair, output_kernel);
    } else if (filter.dim_size(0) == conv_input.dim_size(1) &&
               filter.dim_size(1) == conv_input.dim_size(2) &&
               row_dilation == 1 && col_dilation == 1 &&
               padding == Eigen::PaddingType::PADDING_VALID) {
      // If the input data and filter have the same height/width,
      // reduce the 2D convolution to matrix multiplication.
      const auto k =  // Length of reduction dimension.
          filter.dim_size(0) * filter.dim_size(1) * filter.dim_size(2);

      Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
      dim_pair[0] = Eigen::IndexPair<Eigen::DenseIndex>(1, 0);

      auto out = temp_output.shaped<TempT, 2>(
          {conv_input.dim_size(0), filter.dim_size(3)});
      auto in0 = conv_input.shaped<T, 2>({conv_input.dim_size(0), k});
      auto in1 = filter.shaped<T, 2>({k, filter.dim_size(3)});

      out.device(device) = in0.template cast<ComputeT>().contract(
          in1.template cast<ComputeT>(), dim_pair, output_kernel);
    } else {
      auto out = temp_output.tensor<TempT, 4>();
      auto in0 = conv_input.tensor<T, 4>();
      auto in1 = filter.tensor<T, 4>();

      // Need to swap row/col when calling Eigen.
      out.device(device) = Eigen::SpatialConvolution(
          in0.template cast<ComputeT>(), in1.template cast<ComputeT>(),
          col_stride, row_stride, padding, col_dilation, row_dilation,
          output_kernel);
    }
  }

 private:
  // Contraction output mapper for temporary QInt32 tensor.
  using ContractionOutputMapper =
      Eigen::internal::blas_data_mapper<TempT, Eigen::Index, Eigen::ColMajor>;

  // This output kernel computes an expressions corresponding to cuDNN
  // implementation of INT8 cudnnConvolutionBiasActivationForward:
  // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#scaling-parameters__fig-conv-bias-activation-forward
  struct BiasActivationOutputKernel {
    explicit BiasActivationOutputKernel(const Tensor& conv_input_scale,
                                        const Tensor& side_input,
                                        ScaleType side_input_scale,
                                        const Tensor& bias,
                                        const FusedComputationType fusion,
                                        Tensor* output)
        : fusion(fusion),
          conv_input_scale_data(conv_input_scale.flat<ScaleType>().data()),
          bias_data(bias.flat<BiasType>().data()),
          side_input_data(side_input.flat<T>().data()),
          side_input_scale(side_input_scale),
          output_data(const_cast<T*>(output->flat<T>().data())),
          conv_input_scale_tensor_size(conv_input_scale.NumElements()) {}

    EIGEN_ALWAYS_INLINE void operator()(
        const ContractionOutputMapper& conv_output_mapper,
        const Eigen::TensorContractionParams& params, Eigen::Index i,
        Eigen::Index j, Eigen::Index num_rows, Eigen::Index num_cols) const {
      DCHECK(params.swapped_arguments);

      const auto stride = conv_output_mapper.stride();

      const BiasType* bias_base = bias_data + i;
      const ScaleType* conv_input_scale_base = conv_input_scale_data;
      if (conv_input_scale_tensor_size > 1) {
        conv_input_scale_base += i;
      }

      const T* side_input_base;
      if (side_input_data == nullptr) {
        // side_input_data can be null when the tf::Tensor for the side input is
        // empty.
        side_input_base = nullptr;
      } else {
        side_input_base = side_input_data + i + j * stride;
      }
      T* output_base = output_data + i + j * stride;

      for (int col = 0; col < num_cols; ++col) {
        // A column of an output tensor after QInt8xQInt8 -> QInt32 contraction.
        // This is a temporary tensor, that we will scale, add bias with
        // side_input, and quantize before writing to final output tensor.
        typename TTypes<TempT>::UnalignedTensor conv_output(
            &conv_output_mapper(0, col), num_rows);

        // A column of output quantized tensor corresponding to conv output row.
        typename TTypes<T>::UnalignedTensor output(output_base + col * stride,
                                                   num_rows);

        const BiasType* bias_ptr = bias_base;

        static_assert(
            std::is_same<TempT, ScaleType>::value,
            "Temporary contraction result type must match with scale type.");

        // CuDNN 8 introduced many new kernels for sm75+ GPUs. These kernels use
        // different numerics than those in CuDNN 7-.
        //
        // In cudnn 7-:
        //
        // conv_output = Fma(conv_output, conv_input_scale, bias)
        // conv_output = Fma(conv_output, side_input_scale, side_input),
        //
        // In cudnn 8:
        //
        // conv_output = conv_output * conv_input_scale
        // conv_output = Fma(conv_output, side_input_scale, side_input)
        // conv_output = conv_output + bias
        //
        // One caveat is that the numerics of
        // cudnnConvolutionBiasActivationForward depend on not only the cudnn
        // version but also the GPU's compute capability, which is not visible
        // to the CPU implementation of FusedConv2dBiasActivationOp. So we
        // expect this implementation to be bit exact for cudnn7-/sm70- and
        // cudnn8+/sm75+ but not for cudnn8+/sm70-.
        //
        // NOTE(ezhulenev): We do not use packet FMA for this loop,
        // because it seems that it produces slightly different results,
        // and we are targeting close equality with Nvidia implementation.
        typename TTypes<BiasType>::UnalignedConstTensor bias_slice(bias_ptr,
                                                                   num_rows);

        // (1) Scale.
        if (conv_input_scale_tensor_size > 1) {
          typename TTypes<ScaleType>::UnalignedConstTensor
              conv_input_scale_slice(conv_input_scale_base, num_rows);
          conv_output = conv_output * conv_input_scale_slice;
        } else {
          conv_output = conv_output * (*conv_input_scale_base);
        }

        // (2) Side input.
        if (side_input_scale != 0.0f && side_input_base != nullptr) {
          const T* side_input_ptr = side_input_base + col * stride;
          TempT* conv_output_ptr = conv_output.data();
          for (int idx = 0; idx < num_rows; ++idx) {
            conv_output_ptr[idx] = std::fmaf(
                side_input_ptr[idx], side_input_scale, conv_output_ptr[idx]);
          }
        }

        // (3) Bias.
        conv_output += bias_slice;

        // Round-up, clip and apply activation function.
        static constexpr ScaleType kMaxRange = static_cast<ScaleType>(127.f);
        static constexpr ScaleType kMinRange = static_cast<ScaleType>(-128.f);

        ScaleType lower_bound =
            (fusion == FusedComputationType::kBiasAdd ? kMinRange : 0);
        output = conv_output
                     .unaryExpr(
                         Eigen::internal::scalar_round_half_to_even_op<float>())
                     .clip(lower_bound, kMaxRange)
                     .cast<T>();
      }
    }

   private:
    const FusedComputationType fusion;
    const ScaleType* conv_input_scale_data;
    const BiasType* bias_data;
    const T* side_input_data;
    ScaleType side_input_scale;
    T* output_data;
    const int conv_input_scale_tensor_size;
  };
};

template <>
struct LaunchFusedConv2DOp<CPUDevice, int8>
    : LaunchFusedConv2DOpCpuInt8Helper<int8> {
};

template <>
struct LaunchFusedConv2DOp<CPUDevice, qint8>
    : LaunchFusedConv2DOpCpuInt8Helper<qint8> {};

#if GOOGLE_CUDA

template <typename T>
struct LaunchFusedConv2DOpGpuInt8Helper {
void operator()(
    OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
    const Tensor& input_param, const Tensor& filter_param,
    FusedComputationType fusion, const FusedComputationArgs& fusion_args,
    const Conv2DParameters& params, const Conv2DDimensions& dimensions,
    Tensor* output_param) {
    OP_REQUIRES(ctx, dimensions.in_depth == filter_param.dim_size(1),
                errors::Unimplemented("Fused conv implementation does not "
                                      "support grouped convolutions for now."));
    OP_REQUIRES(ctx, params.data_format == TensorFormat::FORMAT_NCHW_VECT_C,
                errors::Unimplemented(
                    "Fused convolution for int8 is only supported on GPU "
                    "for NCHW_VECT_C format"));
    OP_REQUIRES(ctx,
                DataTypeToEnum<T>::value == DT_INT8 ||
                    DataTypeToEnum<T>::value == DT_QINT8,
                errors::Unimplemented("Specialized fused conv implemented for "
                                      "only int8 and qint8 on GPU."));
    OP_REQUIRES(
        ctx, dimensions.dilation_rows == 1 && dimensions.dilation_cols == 1,
        errors::Unimplemented(
            "Fused conv implementation for int8/qint8 on GPU only supports "
            "dilation of 1 for rows and cols."));
    OP_REQUIRES(
        ctx,
        fusion == FusedComputationType::kBiasAdd ||
            fusion == FusedComputationType::kBiasAddWithRelu,
        errors::Unimplemented(
            "Fused conv implementation for int8/qint8 on GPU only supports "
            "BiasAdd + None or BiasAdd + Relu."));

    constexpr int kBias = 2;
    constexpr int kSideInput = 3;
    constexpr int kConvInputScale = 4;
    constexpr int kSideInputScale = 5;

    const Tensor& bias = ctx->input(kBias);
    const Tensor& side_input_param = ctx->input(kSideInput);
    const Tensor& conv_input_scale_param = ctx->input(kConvInputScale);
    const Tensor& side_input_scale_param = ctx->input(kSideInputScale);

    // Assuming int8 <--> NCHW_VECT_C, OIHW_VECT_I (int8x4) here.
    constexpr TensorFormat data_format = TensorFormat::FORMAT_NCHW_VECT_C;
    constexpr FilterTensorFormat filter_format =
        FilterTensorFormat::FORMAT_OIHW_VECT_I;
    const Padding padding = params.padding;

    int32_t row_stride = dimensions.stride_rows;
    int32_t col_stride = dimensions.stride_cols;

    auto* stream = ctx->op_device_context()->stream();
    OP_REQUIRES(ctx, stream, errors::Internal("No GPU stream available."));
    OP_REQUIRES(ctx, stream->GetCudaComputeCapability().IsAtLeast(6, 1),
                errors::Unimplemented(
                    "Fused convolution for int8 is only supported on GPUs with "
                    "compute capability 6.1 or later."));

    se::TfAllocatorAdapter tf_allocator_adapter(ctx->device()->GetAllocator({}),
                                                stream);
    se::RedzoneAllocator rz_allocator(stream, &tf_allocator_adapter,
                                      se::GpuAsmOpts());

    const int batch_size = GetTensorDim(input_param, data_format, 'N');
    int conv_input_rows = GetTensorDim(input_param, data_format, 'H');
    int conv_input_cols = GetTensorDim(input_param, data_format, 'W');
    const int conv_input_depth =
        GetTensorDim(input_param, data_format, 'C') * 4;

    const int output_rows = GetTensorDim(*output_param, data_format, 'H');
    const int output_cols = GetTensorDim(*output_param, data_format, 'W');
    const int output_depth = GetFilterDim(filter_param, filter_format, 'O');
    const int filter_rows = GetFilterDim(filter_param, filter_format, 'H');
    const int filter_cols = GetFilterDim(filter_param, filter_format, 'W');
    int padding_rows = 0;
    int padding_cols = 0;
    const Tensor* conv_input = &input_param;

    Tensor maybe_padded_conv_input;
    if (padding == Padding::SAME) {
      // Adjusts padding so cudnn supports it. Sets `adjusted_padding` to be the
      // adjusted padding, and `extra_padding_before` and `extra_padding_after`
      // to be the extra padding that FusedConv needs to apply before calling
      // cudnn.
      auto AdjustPaddingForCudnn =
          [](int padding, int filter_size, int* adjusted_padding,
             int* extra_padding_before, int* extra_padding_after) {
#if CUDNN_VERSION < 7000
            if (filter_size >= 6) {
              // TODO(b/70795525): Remove after NVIDIA fixes this bug with int8
              // fused convolution. I don't know cuDNN7 still has the bug, so
              // enable this workaround for cuDNN6 or older.
              *adjusted_padding = 0;
              *extra_padding_before = padding / 2;
              *extra_padding_after = padding - *extra_padding_before;
              return;
            }
#endif
            *adjusted_padding = padding / 2 * 2;
            *extra_padding_before = 0;
            *extra_padding_after = padding % 2;
          };

      // Total padding on rows and cols is
      // Pr = (R' - 1) * S + Kr - R
      // Pc = (C' - 1) * S + Kc - C
      // where (R', C') are output dimensions, (R, C) are input dimensions, S
      // is stride, (Kr, Kc) are filter dimensions.
      // We pad Pr/2 on the left and Pr - Pr/2 on the right, Pc/2 on the top
      // and Pc - Pc/2 on the bottom.  When Pr or Pc is odd, this means
      // we pad more on the right and bottom than on the top and left.
      padding_rows = std::max<int>(
          0, (output_rows - 1) * row_stride + filter_rows - conv_input_rows);
      padding_cols = std::max<int>(
          0, (output_cols - 1) * col_stride + filter_cols - conv_input_cols);
      int extra_top_padding = 0;
      int extra_bottom_padding = 0;
      int extra_left_padding = 0;
      int extra_right_padding = 0;
      AdjustPaddingForCudnn(padding_rows, filter_rows, &padding_rows,
                            &extra_top_padding, &extra_bottom_padding);
      AdjustPaddingForCudnn(padding_cols, filter_cols, &padding_cols,
                            &extra_left_padding, &extra_right_padding);
      if (extra_top_padding != 0 || extra_bottom_padding != 0 ||
          extra_left_padding != 0 || extra_right_padding != 0) {
        const int new_conv_input_rows =
            conv_input_rows + extra_top_padding + extra_bottom_padding;
        const int new_conv_input_cols =
            conv_input_cols + extra_left_padding + extra_right_padding;

        using VectT = int32;
        auto pad_data_format = FORMAT_NCHW;

        TensorShape maybe_padded_conv_input_shape;
        OP_REQUIRES_OK(ctx, ShapeFromFormatWithStatus(
                                data_format, batch_size, new_conv_input_rows,
                                new_conv_input_cols, conv_input_depth,
                                &maybe_padded_conv_input_shape));
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(DataTypeToEnum<T>::value,
                                               maybe_padded_conv_input_shape,
                                               &maybe_padded_conv_input));

        auto conv_input_eigen_tensor =
            To32Bit(input_param.reinterpret_last_dimension<VectT, 4>());
        auto padded_conv_input_eigen_tensor = To32Bit(
            maybe_padded_conv_input.reinterpret_last_dimension<VectT, 4>());

        functor::PadInput<GPUDevice, VectT, int, 4>()(
            ctx->eigen_device<GPUDevice>(), conv_input_eigen_tensor,
            {{extra_top_padding, extra_left_padding}},
            {{extra_bottom_padding, extra_right_padding}},
            padded_conv_input_eigen_tensor, pad_data_format, T{});

        conv_input = &maybe_padded_conv_input;
        conv_input_rows = new_conv_input_rows;
        conv_input_cols = new_conv_input_cols;
      }
  }

  constexpr auto data_layout = se::dnn::DataLayout::kBatchDepthYX4;
  constexpr auto filter_layout = se::dnn::FilterLayout::kOutputInputYX4;

  se::dnn::BatchDescriptor conv_input_desc;
  conv_input_desc.set_count(batch_size)
      .set_feature_map_count(conv_input_depth)
      .set_height(conv_input_rows)
      .set_width(conv_input_cols)
      .set_layout(data_layout);
  se::dnn::FilterDescriptor filter_desc;
  filter_desc.set_input_filter_height(filter_rows)
      .set_input_filter_width(filter_cols)
      .set_input_feature_map_count(conv_input_depth)
      .set_output_feature_map_count(output_depth)
      .set_layout(filter_layout);
  se::dnn::BatchDescriptor side_input_desc;
  side_input_desc.set_count(batch_size)
      .set_height(output_rows)
      .set_width(output_cols)
      .set_feature_map_count(output_depth)
      .set_layout(data_layout);
  se::dnn::BatchDescriptor bias_desc;
  bias_desc.set_count(1)
      .set_height(1)
      .set_width(1)
      .set_feature_map_count(output_depth)
      .set_layout(se::dnn::DataLayout::kBatchDepthYX);
  se::dnn::BatchDescriptor output_desc;
  output_desc.set_count(batch_size)
      .set_height(output_rows)
      .set_width(output_cols)
      .set_feature_map_count(output_depth)
      .set_layout(data_layout);
  se::dnn::ConvolutionDescriptor conv_desc;
  CHECK_EQ(0, padding_rows % 2);  // Crash OK
  CHECK_EQ(0, padding_cols % 2);  // Crash OK
  conv_desc.set_vertical_filter_stride(row_stride)
      .set_horizontal_filter_stride(col_stride)
      .set_zero_padding_height(padding_rows / 2)
      .set_zero_padding_width(padding_cols / 2);

  auto conv_input_ptr = AsDeviceMemory(
      reinterpret_cast<const int8*>(conv_input->template flat<T>().data()),
      conv_input->template flat<T>().size());
  auto filter_ptr = AsDeviceMemory(
      reinterpret_cast<const int8*>(filter_param.template flat<T>().data()),
      filter_param.template flat<T>().size());
  auto side_input_ptr =
      AsDeviceMemory(reinterpret_cast<const int8*>(
                         side_input_param.template flat<T>().data()),
                     side_input_param.template flat<T>().size());
  auto output_ptr = AsDeviceMemory(
      reinterpret_cast<const int8*>(output_param->template flat<T>().data()),
      output_param->template flat<T>().size());
  using BiasType = float;
  auto bias_ptr = AsDeviceMemory(bias.template flat<BiasType>().data(),
                                 bias.template flat<BiasType>().size());

  static int64_t ConvolveScratchSize = GetDnnWorkspaceLimit(
      // default value is in bytes despite the name of the environment variable
      "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB
  );

  se::dnn::ActivationMode dnn_activation_mode;
  switch (fusion) {
    case FusedComputationType::kBiasAdd:
      dnn_activation_mode = se::dnn::ActivationMode::kNone;
      break;
    case FusedComputationType::kBiasAddWithRelu:
      dnn_activation_mode = se::dnn::ActivationMode::kRelu;
      break;
    default:
      LOG(FATAL) << "Unsupported activation type " << (int)fusion;  // Crash OK
  }

  const float conv_scale = conv_input_scale_param.scalar<float>()();
  const float side_input_scale = side_input_scale_param.scalar<float>()();

  constexpr double leakyrelu_alpha = 0;  // This op doesn't support leaky relu
  ConvParameters fused_conv_parameters = {
      stream->parent(),
      batch_size,
      conv_input_depth,
      {{conv_input_rows, conv_input_cols}},
      data_format,
      output_depth,
      {{filter_rows, filter_cols}},
      // TODO(yangzihao): Add support for arbitrary dilations for fused conv.
      {{1, 1}},  // dilation_rows, dilation_cols
      {{row_stride, col_stride}},
      {{padding_rows, padding_cols}},
      conv_input->dtype(),
      /*group_count=*/1,  // This op doesn't support grouped convolutions.
      ConvParameters::FusionInfo{conv_scale, side_input_scale, leakyrelu_alpha,
                                 dnn_activation_mode,
                                 /*is_contrib=*/false},
  };

  constexpr auto type = se::dnn::ToDataType<int8>::value;
  constexpr auto bias_type = se::dnn::ToDataType<BiasType>::value;

  const bool use_cudnn_frontend = CudnnUseFrontend();
  AutotuneEntry<se::dnn::FusedConvOp> autotune_entry;
  if (!FusedConvAutotuneMap::GetInstance()->Find(fused_conv_parameters,
                                                 &autotune_entry)) {
    VLOG(2) << "Autotuning fused convolution (use_frontend="
            << use_cudnn_frontend << "): " << fused_conv_parameters.ToString();
    profiler::ScopedAnnotation trace("cudnn_autotuning");

    std::vector<std::unique_ptr<const se::dnn::FusedConvRunner>> runners;
    TF_CHECK_OK(stream->parent()->GetFusedConvolveRunners(
        use_cudnn_frontend, se::dnn::ConvolutionKind::FORWARD, type, bias_type,
        type, conv_scale, side_input_scale, /*leakyrelu_alpha=*/0.0, stream,
        conv_input_desc, filter_desc, bias_desc, output_desc, conv_desc,
        /*use_fallback=*/false, dnn_activation_mode, GetNumericOptions(),
        &runners));

    auto launch_func =
        [&](se::ScratchAllocator* allocator_used,
            const std::unique_ptr<const se::dnn::FusedConvRunner>& runner,
            se::dnn::ProfileResult* profile_result) -> Status {
      TF_ASSIGN_OR_RETURN(auto scratch, allocator_used->AllocateBytes(
                                            runner->GetWorkspaceSize()));
      return (*runner)(stream, profile_result, scratch, conv_input_ptr,
                       filter_ptr, side_input_ptr, bias_ptr, output_ptr);
    };

    auto results_or = internal::AutotuneConvImpl(
        ctx, runners, cudnn_use_autotune, launch_func, ConvolveScratchSize,
        rz_allocator);
    OP_REQUIRES_OK(ctx, results_or.status());
    auto results = std::move(results_or).value();

    LogFusedConvForwardAutotuneResults(
        type, conv_input_ptr, filter_ptr, output_ptr, bias_ptr, side_input_ptr,
        conv_input_desc, filter_desc, output_desc, conv_desc, conv_scale,
        side_input_scale, dnn_activation_mode, stream->parent(), results);

    // Two-level autotuning: Cudnn frontend supports two engine lists:
    // heuristics and fallback. Heuristics engines are normally faster.
    // To reduce autotuning time, we evaluate the fallback engines only when
    // none of the heuristics engines work.
    bool found_working_engine = false;
    for (auto& result : results) {
      if (!result.has_failure()) {
        found_working_engine = true;
        break;
      }
    }

    if (!CudnnUseFrontend() || found_working_engine) {
      auto runners_or = BestCudnnConvAlgorithm<se::dnn::FusedConvOp>(
          results, std::move(runners));
      OP_REQUIRES_OK(ctx, runners_or.status());
      autotune_entry = {std::move(runners_or).value()};
    } else {
      std::vector<std::unique_ptr<const se::dnn::FusedConvRunner>>
          fallback_runners;
      TF_CHECK_OK(stream->parent()->GetFusedConvolveRunners(
          use_cudnn_frontend, se::dnn::ConvolutionKind::FORWARD, type,
          bias_type, type, conv_scale, side_input_scale, leakyrelu_alpha,
          stream, conv_input_desc, filter_desc, bias_desc, output_desc,
          conv_desc,
          /*use_fallback=*/true, dnn_activation_mode, GetNumericOptions(),
          &fallback_runners));

      auto fallback_results_or = internal::AutotuneConvImpl(
          ctx, fallback_runners, cudnn_use_autotune, launch_func,
          ConvolveScratchSize, rz_allocator);
      OP_REQUIRES_OK(ctx, fallback_results_or.status());
      auto fallback_results = std::move(fallback_results_or).value();

      LogFusedConvForwardAutotuneResults(
          type, conv_input_ptr, filter_ptr, output_ptr, bias_ptr,
          side_input_ptr, conv_input_desc, filter_desc, output_desc, conv_desc,
          conv_scale, side_input_scale, dnn_activation_mode, stream->parent(),
          fallback_results);

      auto fallback_runners_or = BestCudnnConvAlgorithm<se::dnn::FusedConvOp>(
          fallback_results, std::move(fallback_runners));
      OP_REQUIRES_OK(ctx, fallback_runners_or.status());
      autotune_entry = {std::move(fallback_runners_or).value()};
    }

    FusedConvAutotuneMap::GetInstance()->Insert(fused_conv_parameters,
                                                autotune_entry);
  }

  DnnScratchAllocator scratch_allocator(ConvolveScratchSize, ctx);
  Status cudnn_launch_status;
  if (!autotune_entry.is_algorithm_config()) {
    auto& runners = autotune_entry.GetOpRunners();
    typename se::dnn::FusedConvOp::Config config{
        se::dnn::ConvolutionKind::FORWARD,
        type,
        bias_type,
        type,
        conv_scale,
        side_input_scale,
        leakyrelu_alpha,
        conv_input_desc,
        filter_desc,
        bias_desc,
        output_desc,
        conv_desc,
        dnn_activation_mode};

    auto primary_or = runners.primary->GetOrCreateRunner(config, stream);
    OP_REQUIRES_OK(ctx, primary_or.status());
    auto primary = primary_or.value();

    const se::dnn::FusedConvRunner* no_scratch_fallback = nullptr;
    if (runners.no_scratch_fallback) {
      auto no_scratch_fallback_or =
          runners.no_scratch_fallback->GetOrCreateRunner(config, stream);
      OP_REQUIRES_OK(ctx, no_scratch_fallback_or.status());
      no_scratch_fallback = no_scratch_fallback_or.value();
    }

    auto runner_and_scratch_or =
        AllocateScratchOrFallback<se::dnn::FusedConvOp::Signature>(
            &scratch_allocator, primary, no_scratch_fallback);
    OP_REQUIRES_OK(ctx, runner_and_scratch_or.status());
    auto runner_and_scratch = std::move(runner_and_scratch_or).value();
    auto& runner =
        *std::get<const se::dnn::FusedConvRunner*>(runner_and_scratch);
    cudnn_launch_status = runner(
        stream, /*output_profile_result=*/nullptr,
        std::get<se::DeviceMemoryBase>(runner_and_scratch), conv_input_ptr,
        filter_ptr, side_input_ptr, bias_ptr, output_ptr);
  } else {
    cudnn_launch_status = stream->FusedConvolveWithAlgorithm(
        conv_input_desc, conv_input_ptr, conv_scale, filter_desc, filter_ptr,
        conv_desc, side_input_ptr, side_input_scale, bias_desc, bias_ptr,
        dnn_activation_mode, output_desc, &output_ptr, &scratch_allocator,
        autotune_entry.GetAlgorithmConfig(),
        /*output_profile_result=*/nullptr);
  }

  if (!cudnn_launch_status.ok()) {
    ctx->SetStatus(cudnn_launch_status);
  }
}
};

template <>
struct LaunchFusedConv2DOp<GPUDevice, int8>
    : LaunchFusedConv2DOpGpuInt8Helper<int8> {};

template <>
struct LaunchFusedConv2DOp<GPUDevice, qint8>
    : LaunchFusedConv2DOpGpuInt8Helper<qint8> {};

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
