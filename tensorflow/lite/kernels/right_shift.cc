/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <climits>

#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/binary_function.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace right_shift {

// Input/output tensor index.
constexpr int kInputTensor1 = 0;
constexpr int kInputTensor2 = 1;
constexpr int kOutputTensor = 0;

// Op data for right shift op.
struct OpData {
  bool requires_broadcast = false;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new OpData;
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  TF_LITE_ENSURE_TYPES_EQ(context, input1->type, input2->type);

  output->type = input1->type;

  data->requires_broadcast = !HaveSameShapes(input1, input2);

  TfLiteIntArray* output_size = nullptr;
  if (data->requires_broadcast) {
    TF_LITE_ENSURE_OK(context, CalculateShapeForBroadcast(
                                   context, input1, input2, &output_size));
  } else {
    output_size = TfLiteIntArrayCopy(input1->dims);
  }

  return context->ResizeTensor(context, output, output_size);
}

template <typename T>
T RightShift(T x, T y) {
  // Avoids UB: don't shift by larger than the bitwidth of T.
  T y_clamped = y;
  if (y_clamped < 0) {
    y_clamped = 0;
  } else if (y_clamped > sizeof(T) * CHAR_BIT - 1) {
    y_clamped = sizeof(T) * CHAR_BIT - 1;
  }
  // Technically right shifts of signed integers are not necessarily
  // arithmetic shifts according to the C++ standard. However in practice most
  // implementations are arithmetic shifts. If this proves to be a problem in
  // practice, we may need to use an alternative implementation.
  return x >> y_clamped;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input1;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor1, &input1));
  const TfLiteTensor* input2;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputTensor2, &input2));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  const TfLiteType type = output->type;
  switch (type) {
    case kTfLiteUInt8: {
      if (data->requires_broadcast) {
        reference_ops::BroadcastBinaryFunction4DSlow<uint8_t, uint8_t, uint8_t>(
            GetTensorShape(input1), GetTensorData<uint8_t>(input1),
            GetTensorShape(input2), GetTensorData<uint8_t>(input2),
            GetTensorShape(output), GetTensorData<uint8_t>(output), RightShift);
      } else {
        reference_ops::BinaryFunction<uint8_t, uint8_t, uint8_t>(
            GetTensorShape(input1), GetTensorData<uint8_t>(input1),
            GetTensorShape(input2), GetTensorData<uint8_t>(input2),
            GetTensorShape(output), GetTensorData<uint8_t>(output), RightShift);
      }
      break;
    }
    case kTfLiteInt8: {
      if (data->requires_broadcast) {
        reference_ops::BroadcastBinaryFunction4DSlow<int8_t, int8_t, int8_t>(
            GetTensorShape(input1), GetTensorData<int8_t>(input1),
            GetTensorShape(input2), GetTensorData<int8_t>(input2),
            GetTensorShape(output), GetTensorData<int8_t>(output), RightShift);
      } else {
        reference_ops::BinaryFunction<int8_t, int8_t, int8_t>(
            GetTensorShape(input1), GetTensorData<int8_t>(input1),
            GetTensorShape(input2), GetTensorData<int8_t>(input2),
            GetTensorShape(output), GetTensorData<int8_t>(output), RightShift);
      }
      break;
    }
    case kTfLiteUInt16: {
      if (data->requires_broadcast) {
        reference_ops::BroadcastBinaryFunction4DSlow<uint16_t, uint16_t,
                                                     uint16_t>(
            GetTensorShape(input1), GetTensorData<uint16_t>(input1),
            GetTensorShape(input2), GetTensorData<uint16_t>(input2),
            GetTensorShape(output), GetTensorData<uint16_t>(output),
            RightShift);
      } else {
        reference_ops::BinaryFunction<uint16_t, uint16_t, uint16_t>(
            GetTensorShape(input1), GetTensorData<uint16_t>(input1),
            GetTensorShape(input2), GetTensorData<uint16_t>(input2),
            GetTensorShape(output), GetTensorData<uint16_t>(output),
            RightShift);
      }
      break;
    }
    case kTfLiteInt16: {
      if (data->requires_broadcast) {
        reference_ops::BroadcastBinaryFunction4DSlow<int16_t, int16_t, int16_t>(
            GetTensorShape(input1), GetTensorData<int16_t>(input1),
            GetTensorShape(input2), GetTensorData<int16_t>(input2),
            GetTensorShape(output), GetTensorData<int16_t>(output), RightShift);
      } else {
        reference_ops::BinaryFunction<int16_t, int16_t, int16_t>(
            GetTensorShape(input1), GetTensorData<int16_t>(input1),
            GetTensorShape(input2), GetTensorData<int16_t>(input2),
            GetTensorShape(output), GetTensorData<int16_t>(output), RightShift);
      }
      break;
    }
    case kTfLiteUInt32: {
      if (data->requires_broadcast) {
        reference_ops::BroadcastBinaryFunction4DSlow<uint32_t, uint32_t,
                                                     uint32_t>(
            GetTensorShape(input1), GetTensorData<uint32_t>(input1),
            GetTensorShape(input2), GetTensorData<uint32_t>(input2),
            GetTensorShape(output), GetTensorData<uint32_t>(output),
            RightShift);
      } else {
        reference_ops::BinaryFunction<uint32_t, uint32_t, uint32_t>(
            GetTensorShape(input1), GetTensorData<uint32_t>(input1),
            GetTensorShape(input2), GetTensorData<uint32_t>(input2),
            GetTensorShape(output), GetTensorData<uint32_t>(output),
            RightShift);
      }
      break;
    }
    case kTfLiteInt32: {
      if (data->requires_broadcast) {
        reference_ops::BroadcastBinaryFunction4DSlow<int32_t, int32_t, int32_t>(
            GetTensorShape(input1), GetTensorData<int32_t>(input1),
            GetTensorShape(input2), GetTensorData<int32_t>(input2),
            GetTensorShape(output), GetTensorData<int32_t>(output), RightShift);
      } else {
        reference_ops::BinaryFunction<int32_t, int32_t, int32_t>(
            GetTensorShape(input1), GetTensorData<int32_t>(input1),
            GetTensorShape(input2), GetTensorData<int32_t>(input2),
            GetTensorShape(output), GetTensorData<int32_t>(output), RightShift);
      }
      break;
    }
    default:
      TF_LITE_KERNEL_LOG(context,
                         "RightShift currently only supports "
                         "8-bit/16-bit/32-bit integer/unsigned integer, got %s",
                         TfLiteTypeGetName(type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace right_shift

TfLiteRegistration* Register_RIGHT_SHIFT() {
  static TfLiteRegistration r = {right_shift::Init, right_shift::Free,
                                 right_shift::Prepare, right_shift::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
