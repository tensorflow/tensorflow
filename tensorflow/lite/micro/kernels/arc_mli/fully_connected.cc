/* Copyright 2017-2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"

#include "mli_api.h"  // NOLINT
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/arc_mli/mli_slicers.h"
#include "tensorflow/lite/micro/kernels/arc_mli/mli_tf_utils.h"
#include "tensorflow/lite/micro/kernels/arc_mli/scratch_buf_mgr.h"
#include "tensorflow/lite/micro/kernels/arc_mli/scratch_buffers.h"

namespace tflite {
namespace ops {
namespace micro {
namespace fully_connected {
namespace {

struct OpData {
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
  // The index of the temporary tensor where the quantized inputs are cached.
  int input_quantized_index;
};

constexpr int kInputTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;

bool IsMliApplicable(TfLiteContext* context, const TfLiteTensor* input,
                     const TfLiteTensor* filter, const TfLiteTensor* bias,
                     const TfLiteFullyConnectedParams* params) {
  // MLI optimized version only supports int8 dataype and no fused Relu and
  // symmetric per-tensor quantization of weights (not per-axis)
  bool ret_val = (filter->type == kTfLiteInt8) &&
                 (input->type == kTfLiteInt8) && (bias->type == kTfLiteInt32) &&
                 (params->activation == kTfLiteActNone) &&
                 (filter->params.zero_point == 0);
  return ret_val;
}

TfLiteStatus CalculateOpData(TfLiteContext* context,
                             TfLiteFullyConnectedParams* params,
                             TfLiteType data_type, const TfLiteTensor* input,
                             const TfLiteTensor* filter,
                             const TfLiteTensor* bias, TfLiteTensor* output,
                             OpData* data) {
  TfLiteStatus status = kTfLiteOk;
#if !defined(TF_LITE_STRIP_REFERENCE_IMPL)
  if (data_type != kTfLiteFloat32 &&
      !IsMliApplicable(context, input, filter, bias, params)) {
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, filter, bias, output, &real_multiplier));
    int exponent;
    QuantizeMultiplier(real_multiplier, &data->output_multiplier, &exponent);
    data->output_shift = -exponent;
    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, params->activation, output, &data->output_activation_min,
        &data->output_activation_max));
  }
#endif
  return status;
}

}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  OpData* data = nullptr;
  TfLiteStatus status = context->AllocatePersistentBuffer(
      context, sizeof(OpData), reinterpret_cast<void**>(&data));
  if (status != kTfLiteOk || data == nullptr) {
    return nullptr;
  }
  return data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  auto* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE(context, data != nullptr);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_MSG(context, input->type == filter->type,
                     "Hybrid models are not supported on TFLite Micro.");

  TfLiteType data_type = input->type;
  TF_LITE_ENSURE_STATUS(CalculateOpData(context, params, data_type, input,
                                        filter, bias, output, data));

  return kTfLiteOk;
}

TfLiteStatus EvalMliQuantizedInt8(TfLiteContext* context, TfLiteNode* node,
                                  TfLiteFullyConnectedParams* params,
                                  OpData* data, const TfLiteTensor* input,
                                  const TfLiteTensor* filter,
                                  const TfLiteTensor* bias,
                                  TfLiteTensor* output) {
  mli_tensor mli_in = {0};
  mli_tensor mli_weights = {0};
  mli_tensor mli_bias = {0};
  mli_tensor mli_out = {0};

  ConvertToMliTensor<int8_t>(input, &mli_in);
  ConvertToMliTensor<int8_t>(filter, &mli_weights);
  ConvertToMliTensor<int32_t>(bias, &mli_bias);
  ConvertToMliTensor<int8_t>(output, &mli_out);

  /* The input tensor can have more than 2 dimensions. for the compute this
     doesn't make any difference because all the inputs or a batch entry will
     be used anyway. because the MLI kernel doesn't recognize the multiple
     dimensions, the tensor shape is casted to a {batchnum, inputsize} shape. */
  mli_in.shape[0] = mli_out.shape[0];
  mli_in.shape[1] = mli_weights.shape[1];
  mli_in.shape[2] = 0;
  mli_in.shape[3] = 0;
  mli_in.rank = 2;

  // Tensors for data in fast (local) memory and config to copy data from
  // external to local memory
  mli_tensor weights_local = mli_weights;
  mli_tensor bias_local = mli_bias;
  mli_tensor in_local = mli_in;
  mli_tensor out_local = mli_out;
  mli_mov_cfg_t copy_config;
  mli_mov_cfg_for_copy(&copy_config);
  const int weight_out_dimension = 0;
  const int out_tensor_dimension = 1;
  const int input_size_dimension = 1;
  int slice_size = mli_weights.shape[weight_out_dimension];

  /* allocate the local buffers, and compute the slice size */
  TF_LITE_ENSURE_STATUS(get_arc_scratch_buffer_for_fully_connect_tensors(
      context, &in_local, &weights_local, &bias_local, &out_local));
  TF_LITE_ENSURE_STATUS(arc_scratch_buffer_calc_slice_size_weights(
      &weights_local, &bias_local, weight_out_dimension, &slice_size));
  int max_out_slice_size =
      out_local.capacity / mli_hlp_tensor_element_size(&out_local);
  if (slice_size > max_out_slice_size) slice_size = max_out_slice_size;

  /* is_local indicates that the tensor is already in local memory,
     so in that case the original tensor can be used,
     and there is no need to copy it to the local tensor*/
  const bool in_is_local = in_local.data == mli_in.data;
  const bool out_is_local = out_local.data == mli_out.data;
  const bool w_is_local = weights_local.data == mli_weights.data;
  const bool b_is_local = bias_local.data == mli_bias.data;

  TensorSlicer w_slice(&mli_weights, weight_out_dimension, slice_size);
  TensorSlicer b_slice(&mli_bias, weight_out_dimension, slice_size);
  TensorSlicer out_ch_slice(&mli_out, out_tensor_dimension, slice_size, 0, 0, 0,
                            true);

  mli_tensor* w_ptr = w_is_local ? w_slice.Sub() : &weights_local;
  mli_tensor* b_ptr = b_is_local ? b_slice.Sub() : &bias_local;

  void* input_buffer_ptr = NULL;

  while (!w_slice.Done()) {
    mli_mov_tensor_sync(w_slice.Sub(), &copy_config, w_ptr);
    mli_mov_tensor_sync(b_slice.Sub(), &copy_config, b_ptr);

    // Slice the input over the batches (one at a time with the size of a
    // complete input)
    TensorSlicer in_slice(&mli_in, input_size_dimension,
                          mli_in.shape[input_size_dimension]);

    /* output tensor is alreade sliced in the output size dimension.
    out_ch_slice.Sub() is the tensor for the amount of output size of this
    itteration of the weight slice loop. This tensor needs to be further
    sliced over the batch */
    TensorSlicer out_slice(out_ch_slice.Sub(), out_tensor_dimension,
                           slice_size);

    /* setup the pointers to the local or remote tensor to make the code
     * inside the loop easier. */
    mli_tensor* in_ptr = in_is_local ? in_slice.Sub() : &in_local;
    mli_tensor* out_ptr = out_is_local ? out_slice.Sub() : &out_local;

    while (!out_slice.Done()) {
      // if same input copy as previous iteration, skip the copy of input
      if (in_slice.Sub()->data != input_buffer_ptr) {
        mli_mov_tensor_sync(in_slice.Sub(), &copy_config, in_ptr);
        input_buffer_ptr = in_slice.Sub()->data;
      }
      mli_krn_fully_connected_sa8_sa8_sa32(in_ptr, w_ptr, b_ptr, out_ptr);
      mli_mov_tensor_sync(out_ptr, &copy_config, out_slice.Sub());

      in_slice.Next();
      out_slice.Next();
    }
    w_slice.Next();
    b_slice.Next();
    out_ch_slice.Next();
  }
  return kTfLiteOk;
}

TfLiteStatus EvalQuantizedInt8(TfLiteContext* context, TfLiteNode* node,
                               TfLiteFullyConnectedParams* params, OpData* data,
                               const TfLiteTensor* input,
                               const TfLiteTensor* filter,
                               const TfLiteTensor* bias, TfLiteTensor* output) {
#if !defined(TF_LITE_STRIP_REFERENCE_IMPL)
  FullyConnectedParams op_params;
  op_params.input_offset = -input->params.zero_point;
  op_params.weights_offset = -filter->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = -data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;

  reference_integer_ops::FullyConnected(
      op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
      GetTensorShape(filter), GetTensorData<int8_t>(filter),
      GetTensorShape(bias), GetTensorData<int32_t>(bias),
      GetTensorShape(output), GetTensorData<int8_t>(output));
  return kTfLiteOk;
#else
  TF_LITE_KERNEL_LOG(context,
                     "Node configuration is not supported by ARC MLI Library.");
  return kTfLiteError;
#endif
}

TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                           TfLiteFullyConnectedParams* params, OpData* data,
                           const TfLiteTensor* input,
                           const TfLiteTensor* filter, const TfLiteTensor* bias,
                           TfLiteTensor* output) {
#if !defined(TF_LITE_STRIP_REFERENCE_IMPL)
  const int32_t input_offset = -input->params.zero_point;
  const int32_t filter_offset = -filter->params.zero_point;
  const int32_t output_offset = output->params.zero_point;

  tflite::FullyConnectedParams op_params;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = data->output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = -data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;

#define TF_LITE_FULLY_CONNECTED(output_data_type)                      \
  reference_ops::FullyConnected(                                       \
      op_params, GetTensorShape(input), GetTensorData<uint8_t>(input), \
      GetTensorShape(filter), GetTensorData<uint8_t>(filter),          \
      GetTensorShape(bias), GetTensorData<int32_t>(bias),              \
      GetTensorShape(output), GetTensorData<output_data_type>(output))
  switch (output->type) {
    case kTfLiteUInt8:
      TF_LITE_FULLY_CONNECTED(uint8_t);
      break;
    case kTfLiteInt16:
      TF_LITE_FULLY_CONNECTED(int16_t);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(output->type), output->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
#else
  TF_LITE_KERNEL_LOG(context,
                     "Type %s (%d) is not supported by ARC MLI Library.",
                     TfLiteTypeGetName(input->type), input->type);
  return kTfLiteError;
#endif
}

TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node,
                       TfLiteFullyConnectedParams* params, OpData* data,
                       const TfLiteTensor* input, const TfLiteTensor* filter,
                       const TfLiteTensor* bias, TfLiteTensor* output) {
#if !defined(TF_LITE_STRIP_REFERENCE_IMPL)
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);
  tflite::FullyConnectedParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  tflite::reference_ops::FullyConnected(
      op_params, GetTensorShape(input), GetTensorData<float>(input),
      GetTensorShape(filter), GetTensorData<float>(filter),
      GetTensorShape(bias), GetTensorData<float>(bias), GetTensorShape(output),
      GetTensorData<float>(output));
  return kTfLiteOk;
#else
  TF_LITE_KERNEL_LOG(context,
                     "Type %s (%d) is not supported by ARC MLI Library.",
                     TfLiteTypeGetName(input->type), input->type);
  return kTfLiteError;
#endif
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE(context, data != nullptr);

  // Checks in Prepare ensure input, output and filter types are all the same.
  switch (input->type) {
    case kTfLiteFloat32:
      return EvalFloat(context, node, params, data, input, filter, bias,
                       output);
    case kTfLiteInt8:
      if (IsMliApplicable(context, input, filter, bias, params)) {
        return EvalMliQuantizedInt8(context, node, params, data, input, filter,
                                    bias, output);
      } else {
        return EvalQuantizedInt8(context, node, params, data, input, filter,
                                 bias, output);
      }

    case kTfLiteUInt8:
      return EvalQuantized(context, node, params, data, input, filter, bias,
                           output);

    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(filter->type), filter->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace fully_connected

TfLiteRegistration* Register_FULLY_CONNECTED() {
  static TfLiteRegistration r = {/*init=*/fully_connected::Init,
                                 /*free=*/nullptr,
                                 /*prepare=*/fully_connected::Prepare,
                                 /*invoke=*/fully_connected::Eval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
