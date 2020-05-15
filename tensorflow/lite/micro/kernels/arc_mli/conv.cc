/* Copyright 2019-2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/reference/conv.h"

#include "mli_api.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/arc_mli/mli_slicers.h"
#include "tensorflow/lite/micro/kernels/arc_mli/mli_tf_utils.h"
#include "tensorflow/lite/micro/kernels/arc_mli/scratch_buf_mgr.h"
#include "tensorflow/lite/micro/kernels/arc_mli/scratch_buffers.h"

namespace tflite {
namespace ops {
namespace micro {
namespace conv {

constexpr int kInputTensor = 0;
constexpr int kFilterTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;
constexpr int kMaxChannels = 256;

// Conv is quantized along dimension 0:
// https://www.tensorflow.org/lite/performance/quantization_spec
constexpr int kConvQuantizedDimension = 0;

struct OpData {
  TfLitePaddingValues padding;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;

  // Per channel output multiplier and shift.
  int32_t per_channel_output_multiplier[kMaxChannels];
  int32_t per_channel_output_shift[kMaxChannels];

  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
};

inline PaddingType RuntimePaddingType(TfLitePadding padding) {
  switch (padding) {
    case TfLitePadding::kTfLitePaddingSame:
      return PaddingType::kSame;
    case TfLitePadding::kTfLitePaddingValid:
      return PaddingType::kValid;
    case TfLitePadding::kTfLitePaddingUnknown:
    default:
      return PaddingType::kNone;
  }
}


bool IsMliApplicable(TfLiteContext* context, const TfLiteTensor* input,
                     const TfLiteTensor* filter, const TfLiteTensor* bias,
                     const TfLiteConvParams* params) {
  const auto* affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);
  // MLI optimized version only supports int8 dataype, dilation factor of 1 and
  // per-axis quantization of weights (no broadcasting/per-tensor)
  bool ret_val = (filter->type == kTfLiteInt8) && 
                 (input->type == kTfLiteInt8) &&
                 (bias->type == kTfLiteInt32) &&
                 (params->dilation_width_factor == 1) &&
                 (params->dilation_height_factor == 1) &&
                 (affine_quantization->scale->size ==
                  filter->dims->data[kConvQuantizedDimension]) &&
                 affine_quantization->scale->size <= (kMaxChannels * 2);
  return ret_val;
}


TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node,
                             TfLiteConvParams* params, int width, int height,
                             int filter_width, int filter_height, int out_width,
                             int out_height, const TfLiteType data_type,
                             bool mli_is_applicable, OpData* data) {
  bool has_bias = node->inputs->size == 3;
  // Check number of inputs/outputs
  TF_LITE_ENSURE(context, has_bias || node->inputs->size == 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params->padding;
  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width,
      params->dilation_height_factor, params->dilation_width_factor, height,
      width, filter_height, filter_width, padding, &out_height, &out_width);

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
#if !defined(TF_LITE_STRIP_REFERENCE_IMPL)
  if (data_type != kTfLiteFloat32 && !mli_is_applicable) {
    const TfLiteTensor* input = GetInput(context, node, kInputTensor);
    const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
    const TfLiteTensor* bias =
        GetOptionalInputTensor(context, node, kBiasTensor);
    TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
    int output_channels = filter->dims->data[kConvQuantizedDimension];

    TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params->activation,
        &data->output_multiplier, &data->output_shift,
        &data->output_activation_min, &data->output_activation_max,
        data->per_channel_output_multiplier,
        reinterpret_cast<int*>(data->per_channel_output_shift),
        output_channels));
  }
#endif
  return kTfLiteOk;
}

TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                           TfLiteConvParams* params, OpData* data,
                           const TfLiteTensor* input, const TfLiteTensor* filter,
                           const TfLiteTensor* bias, TfLiteTensor* im2col,
                           TfLiteTensor* hwcn_weights, TfLiteTensor* output) {
#if !defined(TF_LITE_STRIP_REFERENCE_IMPL)
  const int32_t input_offset = -input->params.zero_point;
  const int32_t filter_offset = -filter->params.zero_point;
  const int32_t output_offset = output->params.zero_point;

  ConvParams op_params;
  op_params.padding_type = RuntimePaddingType(params->padding);
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = -data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  reference_ops::Conv(op_params, GetTensorShape(input),
                      GetTensorData<uint8_t>(input), GetTensorShape(filter),
                      GetTensorData<uint8_t>(filter), GetTensorShape(bias),
                      GetTensorData<int32_t>(bias), GetTensorShape(output),
                      GetTensorData<uint8_t>(output), GetTensorShape(im2col),
                      GetTensorData<uint8_t>(im2col), nullptr);
  return kTfLiteOk;
#else
  TF_LITE_KERNEL_LOG(context, "Type %s (%d) is not supported by ARC MLI Library.",
                     TfLiteTypeGetName(input->type), input->type);
  return kTfLiteError;
#endif
}

TfLiteStatus EvalMliQuantizedPerChannel(
    TfLiteContext* context, TfLiteNode* node, TfLiteConvParams* params,
    OpData* data, const TfLiteTensor* input, const TfLiteTensor* filter,
    const TfLiteTensor* bias, TfLiteTensor* output) {
  // Run Conv MLI kernel
  // MLI optimized version only supports int8 dataype and dilation factor of 1
  if ((input->type == kTfLiteInt8) && (params->dilation_width_factor == 1) &&
      (params->dilation_height_factor == 1)) {
    mli_tensor mli_in = {0};
    mli_tensor mli_weights = {0};
    mli_tensor mli_bias = {0};
    mli_tensor mli_out = {0};
    mli_conv2d_cfg cfg = {};

    // reuse space allocated for OpData parameters
    mli_weights.el_params.asym.scale.pi16 =
        (int16_t*)data->per_channel_output_multiplier;
    mli_bias.el_params.asym.scale.pi16 =
        (int16_t*)data->per_channel_output_shift;

    int16_t filter_zero_point = 0;
    int16_t bias_zero_point = 0;
    mli_weights.el_params.asym.zero_point.pi16 = &filter_zero_point;
    mli_bias.el_params.asym.zero_point.pi16 = &bias_zero_point;

    ConvertToMliTensor<int8_t>(input, &mli_in);
    ConvertToMliTensorPerChannel<int8_t>(filter, &mli_weights);
    ConvertToMliTensorPerChannel<int32_t>(bias, &mli_bias);
    ConvertToMliTensor<int8_t>(output, &mli_out);

    if (params->activation == kTfLiteActRelu) {
      cfg.relu.type = MLI_RELU_GEN;
    } else if (params->activation == kTfLiteActRelu6) {
      cfg.relu.type = MLI_RELU_6;
    } else if (params->activation == kTfLiteActRelu1) {
      cfg.relu.type = MLI_RELU_1;
    } else {
      cfg.relu.type = MLI_RELU_NONE;
    }

    cfg.stride_width = params->stride_width;
    cfg.stride_height = params->stride_height;
    if (params->padding == kTfLitePaddingValid) {
      cfg.padding_left = 0;
      cfg.padding_right = 0;
      cfg.padding_top = 0;
      cfg.padding_bottom = 0;
    } else {
      cfg.padding_left = data->padding.width;
      cfg.padding_right = data->padding.width + data->padding.width_offset;
      cfg.padding_top = data->padding.height;
      cfg.padding_bottom = data->padding.height + data->padding.height_offset;
    }

    // for height slicing
    const int height_dimension = 1;
    int in_slice_height = 0;
    int out_slice_height = 0;
    const int kernel_height =
        static_cast<int>(mli_weights.shape[KRNL_H_DIM_HWC]);
    const int overlap = kernel_height - cfg.stride_height;

    // for weight slicing (on output channels)
    // NHWC layout for weigths, output channel dimension is the first dimension.
    const int weight_out_ch_dimension = 0;        
    int slice_channels =
        static_cast<int>(mli_weights.shape[weight_out_ch_dimension]);
    // Batch-Height-Width-Channel layout means last dimension is output channels.
    const int out_tensor_ch_dimension = 3;
            

    // Tensors for data in fast (local) memory and config to copy data from
    // external to local memory
    mli_tensor weights_local = mli_weights;
    mli_tensor bias_local = mli_bias;
    mli_tensor in_local = mli_in;
    mli_tensor out_local = mli_out;
    mli_mov_cfg_t copy_config;
    mli_mov_cfg_for_copy(&copy_config);
    TF_LITE_ENSURE_STATUS(get_arc_scratch_buffer_for_conv_tensors(
        context, &in_local, &weights_local, &bias_local, &out_local));
    TF_LITE_ENSURE_STATUS(arc_scratch_buffer_calc_slice_size_io(
        &in_local, &out_local, kernel_height, cfg.stride_height,
        cfg.padding_top, cfg.padding_bottom, &in_slice_height,
        &out_slice_height));
    TF_LITE_ENSURE_STATUS(arc_scratch_buffer_calc_slice_size_weights(
        &weights_local, &bias_local, weight_out_ch_dimension, &slice_channels));

    /* is_local indicates that the tensor is already in local memory,
       so in that case the original tensor can be used,
       and there is no need to copy it to the local tensor*/
    const bool in_is_local = in_local.data == mli_in.data;
    const bool out_is_local = out_local.data == mli_out.data;
    const bool w_is_local = weights_local.data == mli_weights.data;
    const bool b_is_local = bias_local.data == mli_bias.data;

    TensorSlicer w_slice(&mli_weights, weight_out_ch_dimension, slice_channels);
    TensorSlicer b_slice(&mli_bias, weight_out_ch_dimension, slice_channels);
    TensorSlicer out_ch_slice(&mli_out, out_tensor_ch_dimension, slice_channels,
                              0, 0, 0, true);

    mli_tensor* w_ptr = w_is_local ? w_slice.Sub() : &weights_local;
    mli_tensor* b_ptr = b_is_local ? b_slice.Sub() : &bias_local;

    void* input_buffer_ptr = NULL;
    int input_buffer_size = 0;

    while (!w_slice.Done()) {
      mli_mov_tensor_sync(w_slice.Sub(), &copy_config, w_ptr);
      mli_mov_tensor_sync(b_slice.Sub(), &copy_config, b_ptr);

      /* mli_in tensor contains batches of HWC tensors. so it is a 4 dimensional
      tensor. because the mli kernel will process one HWC tensor at a time, the
      4 dimensional tensor needs to be sliced into nBatch 3 dimensional tensors.
      on top of that there could be a need to also slice in the Height
      dimension. for that the sliceHeight has been calculated. The tensor slicer
      is configured that it will completely slice the nBatch dimension (0) and
      slice the height dimension (1) in chunks of 'sliceHeight' */
      TensorSlicer in_slice(&mli_in, height_dimension, in_slice_height,
                            cfg.padding_top, cfg.padding_bottom, overlap);

      /* output tensor is alreade sliced in the output channel dimension.
      out_ch_slice.Sub() is the tensor for the amount of output channels of this
      itteration of the weight slice loop. This tensor needs to be further
      sliced over the batch and height dimension. */
      TensorSlicer out_slice(out_ch_slice.Sub(), height_dimension,
                             out_slice_height);

      /* setup the pointers to the local or remote tensor to make the code
       * inside the loop easier. */
      mli_tensor* in_ptr = in_is_local ? in_slice.Sub() : &in_local;
      mli_tensor* out_ptr = out_is_local ? out_slice.Sub() : &out_local;

      while (!out_slice.Done()) {
        TF_LITE_ENSURE(context, !in_slice.Done());
        cfg.padding_top = in_slice.GetPaddingPre();
        cfg.padding_bottom = in_slice.GetPaddingPost();

        // if same input copy as previous iteration, skip the copy of input
        if ((in_slice.Sub()->data != input_buffer_ptr) ||
            (mli_hlp_count_elem_num(in_slice.Sub(), 0) != input_buffer_size)) {
          mli_mov_tensor_sync(in_slice.Sub(), &copy_config, in_ptr);
          input_buffer_ptr = in_slice.Sub()->data;
          input_buffer_size = mli_hlp_count_elem_num(in_slice.Sub(), 0);
        }
        mli_krn_conv2d_nhwc_sa8_sa8_sa32(in_ptr, w_ptr, b_ptr, &cfg, out_ptr);
        mli_mov_tensor_sync(out_ptr, &copy_config, out_slice.Sub());

        in_slice.Next();
        out_slice.Next();
      }
      w_slice.Next();
      b_slice.Next();
      out_ch_slice.Next();
      TF_LITE_ENSURE(context, in_slice.Done());
    }
  }
  return kTfLiteOk;
}

TfLiteStatus EvalQuantizedPerChannel(TfLiteContext* context, TfLiteNode* node,
                                     TfLiteConvParams* params, OpData* data,
                                     const TfLiteTensor* input,
                                     const TfLiteTensor* filter,
                                     const TfLiteTensor* bias,
                                     TfLiteTensor* output) {
#if !defined(TF_LITE_STRIP_REFERENCE_IMPL)
  ConvParams op_params;
  op_params.input_offset = -input->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;

  reference_integer_ops::ConvPerChannel(
      op_params, data->per_channel_output_multiplier,
      data->per_channel_output_shift, GetTensorShape(input),
      GetTensorData<int8>(input), GetTensorShape(filter),
      GetTensorData<int8>(filter), GetTensorShape(bias),
      GetTensorData<int32>(bias), GetTensorShape(output),
      GetTensorData<int8>(output));
  return kTfLiteOk;
#else
  TF_LITE_KERNEL_LOG(context,
                     "Node configuration is not supported by ARC MLI Library.");
  return kTfLiteError;
#endif
}

TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node,
                       TfLiteConvParams* params, OpData* data,
                       const TfLiteTensor* input, const TfLiteTensor* filter,
                       const TfLiteTensor* bias, TfLiteTensor* im2col,
                       TfLiteTensor* hwcn_weights, TfLiteTensor* output) {
#if !defined(TF_LITE_STRIP_REFERENCE_IMPL)
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

  ConvParams op_params;
  op_params.padding_type = RuntimePaddingType(params->padding);
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.dilation_width_factor = params->dilation_width_factor;
  op_params.dilation_height_factor = params->dilation_height_factor;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  reference_ops::Conv(op_params, GetTensorShape(input),
                      GetTensorData<float>(input), GetTensorShape(filter),
                      GetTensorData<float>(filter), GetTensorShape(bias),
                      GetTensorData<float>(bias), GetTensorShape(output),
                      GetTensorData<float>(output), GetTensorShape(im2col),
                      GetTensorData<float>(im2col));
  return kTfLiteOk;
#else
  TF_LITE_KERNEL_LOG(context, "Type %s (%d) is not supported by ARC MLI Library.",
                     TfLiteTypeGetName(input->type), input->type);
  return kTfLiteError;
#endif
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);

  int input_width = input->dims->data[2];
  int input_height = input->dims->data[1];
  int filter_width = filter->dims->data[2];
  int filter_height = filter->dims->data[1];
  int output_width = output->dims->data[2];
  int output_height = output->dims->data[1];

  OpData data;

  // All per-channel quantized tensors need valid zero point and scale arrays.
  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);

    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);
    TF_LITE_ENSURE(context, affine_quantization->zero_point);

    TF_LITE_ENSURE(context,
                   affine_quantization->scale->size == 1 ||
                       affine_quantization->scale->size ==
                           filter->dims->data[kConvQuantizedDimension]);
    TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size,
                      affine_quantization->zero_point->size);
  }
  bool mli_is_applicable = IsMliApplicable(context, input, filter, bias, params);
  TF_LITE_ENSURE_STATUS(
      CalculateOpData(context, node, params, input_width, input_height,
                      filter_width, filter_height, output_width, output_height,
                      input->type, mli_is_applicable, &data));

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      return EvalFloat(context, node, params, &data, input, filter, bias, nullptr,
                nullptr, output);
      break;
    case kTfLiteInt8:
      if (mli_is_applicable) {
        return EvalMliQuantizedPerChannel(context, node, params, &data, input,
                                          filter, bias, output);

      } else {
        return EvalQuantizedPerChannel(context, node, params, &data, input,
                                       filter, bias, output);
      }
      break;
    case kTfLiteUInt8:
      return EvalQuantized(context, node, params, &data, input, filter, bias, nullptr,
                    nullptr, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace conv

TfLiteRegistration* Register_CONV_2D() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr,
                                 /*prepare=*/nullptr,
                                 /*invoke=*/conv::Eval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
