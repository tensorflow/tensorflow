/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <xtensa/tie/xt_hifi3.h>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace xtensa {
namespace hifi3 {

// Helper XTENSA wrappers:
#define XT_PACK_INT8_16X4_REG(v, reg)                     \
  {                                                       \
    ae_int32x2 r;                                         \
    ae_int16x4 packed;                                    \
    const ae_p16x2s* vec_8x2 = (const ae_p16x2s*)(v - 4); \
    AE_L16X2M_IU(r, vec_8x2, 4);                          \
    ((int16_t*)&packed)[0] = ((int16_t*)&r)[1];           \
    ((int16_t*)&packed)[1] = ((int16_t*)&r)[0];           \
    ((int16_t*)&packed)[2] = ((int16_t*)&r)[3];           \
    ((int16_t*)&packed)[3] = ((int16_t*)&r)[2];           \
    reg = AE_SRAI16(packed, 8);                           \
    reg = AE_SEL16_7520(packed, reg);                     \
  }

void FullyConnected(const FullyConnectedParams& params,
                    const RuntimeShape& input_shape, const int8_t* input_data,
                    const RuntimeShape& filter_shape, const int8_t* filter_data,
                    const RuntimeShape& bias_shape, const int32_t* bias_data,
                    const RuntimeShape& output_shape, int8_t* output_data) {
  const int32 input_offset = params.input_offset;
  const int32 filter_offset = params.weights_offset;
  const int32 output_offset = params.output_offset;
  const int32 output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_GE(filter_shape.DimensionsCount(), 2);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 2);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = filter_shape.DimensionsCount();
  const int batches = output_shape.Dims(0);
  const int output_depth = output_shape.Dims(1);
  TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
  const int accum_depth = filter_shape.Dims(filter_dim_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      // same...
      int num_iters = (accum_depth + 3) / 4;

      //int32 i1 = input_data[b * accum_depth];
      //int32 i2 = *(input_data + (b * accum_depth));
      //int32 f1 = filter_data[out_c * accum_depth];
      //int32 f2 = *(filter_data + (out_c * accum_depth));

      const int8_t* in_ptr = input_data + (b * accum_depth);
      for (int d = 0; d < num_iters; d++) {
        for (int i = 0; i < 4; i++) {

          printf("(%d) input: %d - %d\n",
                 b * accum_depth * d + i,
                 *in_ptr, input_data[b * accum_depth * d + i]);
          in_ptr++;
        }
      }
      for (int d = 0; d < accum_depth; ++d) {
        printf("(%d)  ", b * accum_depth + d);
      }
      printf("\n");

      //int32 acc = 0;
      //for (int d = 0; d < accum_depth; ++d) {
      //  int32 input_val = input_data[b * accum_depth + d];
      //  int32 filter_val = filter_data[out_c * accum_depth + d];
      //  printf("input: %d - %d\n", input_val,
      //         *(input_data + (b * accum_depth + d)));
      //  printf("filter: %d - %d\n", filter_val,
      //         *(filter_data + (out_c * accum_depth + d)));
      //  acc += (filter_val + filter_offset) * (input_val + input_offset);
      //}

      /* printf("accum_depth: %d\n", accum_depth); */
      /* printf("num_iters: %d\n", num_iters); */

      /* const int8_t* input_ptr = input_data + b * accum_depth; */
      /* const int8_t* filter_ptr = filter_data + out_c * accum_depth; */

      /* printf("input_ptr: %d - %d\n", *input_ptr, input_data[b * accum_depth]); */
      /* printf("filter_ptr: %d - %d\n", *filter_ptr, filter_ptr[out_c * accum_depth]); */

      /* // TODO - account for offset! */
      /* while (num_iters--) { */
      /*   // */
      /*   // TODO(kreeger): Left off right here. Need to do this properly. */
      /*   // Intrinsics are working... */
      /*   // */
      /*   int8_t v1[] = {1, 2, 3, 4}; */
      /*   ae_int16x4 reg1; */
      /*   XT_PACK_INT8_16X4_REG(v1, reg1); */

      /*   int8_t v2[] = {4, 3, 2, 1}; */
      /*   ae_int16x4 reg2; */
      /*   XT_PACK_INT8_16X4_REG(v2, reg2); */

      /*   ae_int32x2 sum1 = AE_ZERO32(); */
      /*   ae_int32x2 sum2 = AE_ZERO32(); */

      /*   AE_MULA16X4(sum1, sum2, reg1, reg2); */

      /*   ae_int32x2 sum12 = AE_ADD32(sum1, sum2); */
      /*   printf("SUM: %d\n", AE_MOVAD32_L(sum12) + AE_MOVAD32_H(sum12)); */
      /* } */


      // if (bias_data) {
      //   acc += bias_data[out_c];
      // }
      // acc = MultiplyByQuantizedMultiplier(acc, output_multiplier,
      // output_shift); acc += output_offset; acc = std::max(acc,
      // output_activation_min); acc = std::min(acc, output_activation_max);
      // output_data[out_c + output_depth * b] = static_cast<int8_t>(acc);
    }
  }
}

/*
  ae_int32x2 sum1 = AE_ZERO32();
  ae_int32x2 sum2 = AE_ZERO32();

  // Load the vectors.
  const ae_int16x4* vec_1x4 = (const ae_int16x4*) (vec_1);
  const ae_int16x4* vec_2x4 = (const ae_int16x4*) (vec_2);

  int num_iterations = (n + 3) / 4;
  while (num_iterations--) {
    ae_int16x4 reg_1, reg_2;
    // Load 4 16-bit elements.
    AE_L16X4_IP(reg_1, vec_1x4, 8);
    AE_L16X4_IP(reg_2, vec_2x4, 8);
    // Multiply 4 16-bit numbers, and accumulate them in 4 32-bit accumulators.
    AE_MULA16X4(sum1, sum2, reg_1, reg_2);
  }
  // Reduce the 4 32-bit accumulators into 2 32-bit accumulators.
  ae_int32x2 sum12 = AE_ADD32(sum1, sum2);
  // Return the sum of the two accumulators.
  int sum12_L = AE_MOVAD32_L(sum12);
  int sum12_H = AE_MOVAD32_H(sum12);
  return sum12_L + sum12_H;
*/

}  // namespace hifi3
}  // namespace xtensa
}  // namespace tflite

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

TfLiteStatus CalculateOpData(TfLiteContext* context,
                             TfLiteFullyConnectedParams* params,
                             TfLiteType data_type, const TfLiteTensor* input,
                             const TfLiteTensor* filter,
                             const TfLiteTensor* bias, TfLiteTensor* output,
                             OpData* data) {
  TfLiteStatus status = kTfLiteOk;
  if (data_type != kTfLiteFloat32) {
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
  return status;
}

}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return nullptr;
}

void Free(TfLiteContext* context, void* buffer) {}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus EvalQuantizedInt8(TfLiteContext* context, TfLiteNode* node,
                               TfLiteFullyConnectedParams* params, OpData* data,
                               const TfLiteTensor* input,
                               const TfLiteTensor* filter,
                               const TfLiteTensor* bias, TfLiteTensor* output) {
  FullyConnectedParams op_params;
  op_params.input_offset = -input->params.zero_point;
  op_params.weights_offset = -filter->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.output_multiplier = data->output_multiplier;
  // TODO(b/138810107): Figure out whether output shift should be inverted
  op_params.output_shift = -data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;

  xtensa::hifi3::FullyConnected(
      op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
      GetTensorShape(filter), GetTensorData<int8_t>(filter),
      GetTensorShape(bias), GetTensorData<int32_t>(bias),
      GetTensorShape(output), GetTensorData<int8_t>(output));
  return kTfLiteOk;
}

TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                           TfLiteFullyConnectedParams* params, OpData* data,
                           const TfLiteTensor* input,
                           const TfLiteTensor* filter, const TfLiteTensor* bias,
                           TfLiteTensor* output) {
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
      context->ReportError(
          context,
          "Quantized FullyConnected expects output data type uint8 or int16");
      return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node,
                       TfLiteFullyConnectedParams* params, OpData* data,
                       const TfLiteTensor* input, const TfLiteTensor* filter,
                       const TfLiteTensor* bias, TfLiteTensor* output) {
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
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TfLiteType data_type = input->type;
  OpData local_data_object;
  OpData* data = &local_data_object;
  TF_LITE_ENSURE_STATUS(CalculateOpData(context, params, data_type, input,
                                        filter, bias, output, data));

  switch (filter->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      return EvalFloat(context, node, params, data, input, filter, bias,
                       output);
    case kTfLiteInt8:
      return EvalQuantizedInt8(context, node, params, data, input, filter, bias,
                               output);

    case kTfLiteUInt8:
      return EvalQuantized(context, node, params, data, input, filter, bias,
                           output);

    default:
      context->ReportError(context, "Type %d not currently supported.",
                           filter->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace fully_connected

TfLiteRegistration* Register_FULLY_CONNECTED() {
  static TfLiteRegistration r = {fully_connected::Init, fully_connected::Free,
                                 fully_connected::Prepare,
                                 fully_connected::Eval};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
