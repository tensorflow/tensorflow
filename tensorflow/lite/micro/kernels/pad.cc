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
#include "tensorflow/lite/kernels/internal/reference/pad.h"

#include <string.h>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace pad {
namespace {

struct OpData {
  PadParams params;
  int32_t output_zero_point;
};

}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  OpData* data = static_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE(context, NumInputs(node) == 2 || NumInputs(node) == 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, /*index=*/0);
  TF_LITE_ENSURE(context, input != nullptr);
  const TfLiteTensor* paddings = GetInput(context, node, /*index=*/1);
  TF_LITE_ENSURE(context, paddings != nullptr);
  const TfLiteTensor* constant_values =
      NumInputs(node) == 3 ? GetInput(context, node, /*index=*/2) : nullptr;
  TfLiteTensor* output = GetOutput(context, node, /*index=*/0);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  // Current implementations rely on the inputs being <= 4D.
  TF_LITE_ENSURE(context, NumDimensions(input) <=
                              reference_ops::PadKernelMaxDimensionCount());

  if (constant_values != nullptr) {
    TF_LITE_ENSURE_EQ(context, input->type, constant_values->type);
    // Ensure that constant_values is a scalar.
    TF_LITE_ENSURE_EQ(context, NumElements(constant_values), 1);
  }

  // There must be a pair of paddings for each output dimension.
  TF_LITE_ENSURE_EQ(context, GetTensorShape(paddings).FlatSize(),
                    output->dims->size * 2);

  // On Micro, outputs must be properly sized by the converter.
  // NOTE: This data is only available because the paddings buffer is stored in
  // the flatbuffer:
  TF_LITE_ENSURE(context, IsConstantTensor(paddings));
  const int32_t* paddings_data = GetTensorData<int32_t>(paddings);
  for (int i = 0; i < output->dims->size; i++) {
    int output_dim = output->dims->data[i];
    int expected_dim =
        input->dims->data[i] + paddings_data[i * 2] + paddings_data[i * 2 + 1];
    TF_LITE_ENSURE_EQ(context, output_dim, expected_dim);
  }

  // Calculate OpData:
  data->params.resizing_category = ResizingCategory::kGenericResize;
  const int paddings_total = GetTensorShape(paddings).FlatSize();
  if (paddings_total == 8 && (paddings_data[0] == 0 && paddings_data[1] == 0) &&
      (paddings_data[6] == 0 && paddings_data[7] == 0)) {
    data->params.resizing_category = ResizingCategory::kImageStyle;
  }

  const int num_input_dimensions = NumDimensions(input);
  data->params.left_padding_count = num_input_dimensions;
  data->params.right_padding_count = num_input_dimensions;

  for (int idx = num_input_dimensions - 1; idx >= 0; --idx) {
    data->params.left_padding[idx] = paddings_data[idx * 2];
    data->params.right_padding[idx] = paddings_data[idx * 2 + 1];
  }

  if (input->type == kTfLiteInt8 || input->type == kTfLiteUInt8) {
    if (constant_values == nullptr) {
      // Quantized Pad requires that 0 is represented in the quantized
      // range.
      if (input->type == kTfLiteUInt8) {
        TF_LITE_ENSURE(context, output->params.zero_point >=
                                    std::numeric_limits<uint8_t>::min());
        TF_LITE_ENSURE(context, output->params.zero_point <=
                                    std::numeric_limits<uint8_t>::max());
      } else {
        TF_LITE_ENSURE(context, output->params.zero_point >=
                                    std::numeric_limits<int8_t>::min());
        TF_LITE_ENSURE(context, output->params.zero_point <=
                                    std::numeric_limits<int8_t>::max());
      }
    } else {
      // Quantized Pad requires that 'constant_values' is represented in the
      // same quantized range as the input and output tensors.
      TF_LITE_ENSURE_EQ(context, output->params.zero_point,
                        constant_values->params.zero_point);
      TF_LITE_ENSURE_EQ(context, static_cast<double>(output->params.scale),
                        static_cast<double>(constant_values->params.scale));
    }
    data->output_zero_point = output->params.zero_point;
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData* data = static_cast<const OpData*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, /*index=*/0);
  const TfLiteEvalTensor* constant_values =
      NumInputs(node) == 3
          ? tflite::micro::GetEvalInput(context, node, /*index=*/2)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, /*index=*/0);

  switch (input->type) {
    case kTfLiteFloat32: {
      float pad_value =
          constant_values == nullptr
              ? 0.f
              : *tflite::micro::GetTensorData<float>(constant_values);
      if (data->params.resizing_category == ResizingCategory::kImageStyle) {
        reference_ops::PadImageStyle(
            data->params, tflite::micro::GetTensorShape(input),
            tflite::micro::GetTensorData<float>(input), &pad_value,
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<float>(output));
      } else {
        reference_ops::Pad(data->params, tflite::micro::GetTensorShape(input),
                           tflite::micro::GetTensorData<float>(input),
                           &pad_value, tflite::micro::GetTensorShape(output),
                           tflite::micro::GetTensorData<float>(output));
      }
    } break;
    case kTfLiteUInt8: {
      uint8_t pad_value;
      if (constant_values == nullptr) {
        pad_value = static_cast<uint8_t>(data->output_zero_point);
      } else {
        pad_value = *tflite::micro::GetTensorData<uint8_t>(constant_values);
      }
      if (data->params.resizing_category == ResizingCategory::kImageStyle) {
        reference_ops::PadImageStyle(
            data->params, tflite::micro::GetTensorShape(input),
            tflite::micro::GetTensorData<uint8_t>(input), &pad_value,
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<uint8_t>(output));
      } else {
        reference_ops::Pad(data->params, tflite::micro::GetTensorShape(input),
                           tflite::micro::GetTensorData<uint8_t>(input),
                           &pad_value, tflite::micro::GetTensorShape(output),
                           tflite::micro::GetTensorData<uint8_t>(output));
      }
    } break;
    case kTfLiteInt8: {
      int8_t pad_value;
      if (constant_values == nullptr) {
        pad_value = static_cast<uint8_t>(data->output_zero_point);
      } else {
        pad_value = *tflite::micro::GetTensorData<int8_t>(constant_values);
      }
      if (data->params.resizing_category == ResizingCategory::kImageStyle) {
        reference_ops::PadImageStyle(
            data->params, tflite::micro::GetTensorShape(input),
            tflite::micro::GetTensorData<int8_t>(input), &pad_value,
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<int8_t>(output));
      } else {
        reference_ops::Pad(data->params, tflite::micro::GetTensorShape(input),
                           tflite::micro::GetTensorData<int8_t>(input),
                           &pad_value, tflite::micro::GetTensorShape(output),
                           tflite::micro::GetTensorData<int8_t>(output));
      }
    } break;
    case kTfLiteInt32: {
      int32_t pad_value =
          constant_values == nullptr
              ? 0
              : *tflite::micro::GetTensorData<int32_t>(constant_values);
      reference_ops::Pad(data->params, tflite::micro::GetTensorShape(input),
                         tflite::micro::GetTensorData<int32_t>(input),
                         &pad_value, tflite::micro::GetTensorShape(output),
                         tflite::micro::GetTensorData<int32_t>(output));
    } break;
    default:

      TF_LITE_KERNEL_LOG(context, "Type %s not currently supported by Pad.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
#undef TF_LITE_PAD
  return kTfLiteOk;
}

}  // namespace pad

TfLiteRegistration Register_PAD() {
  return {/*init=*/pad::Init,
          /*free=*/nullptr,
          /*prepare=*/pad::Prepare,
          /*invoke=*/pad::Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

// Also register Pad as PadV2.
TfLiteRegistration Register_PADV2() {
  return {/*init=*/pad::Init,
          /*free=*/nullptr,
          /*prepare=*/pad::Prepare,
          /*invoke=*/pad::Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
