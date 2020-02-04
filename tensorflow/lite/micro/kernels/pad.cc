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

#include "tensorflow/lite/kernels/internal/types.h"

#ifdef MEMORY_SANITIZER
#include <sanitizer/msan_interface.h>
#else
#define __msan_check_mem_is_initialized(ptr, size)
#endif

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace micro {
namespace pad {

struct PadContext {
  PadContext(TfLiteContext* context, TfLiteNode* node) {
    input = GetInput(context, node, 0);
    paddings = GetInput(context, node, 1);
    constant_values = nullptr;
    if (NumInputs(node) == 3) {
      constant_values = GetOptionalInputTensor(context, node, 2);
    } else {
      constant_values = nullptr;
    }
    output = GetOutput(context, node, 0);
    dims = NumDimensions(input);

    resizing_category = ResizingCategory::kGenericResize;
    const int paddings_total = GetTensorShape(paddings).FlatSize();
    const int32* paddings_data = GetTensorData<int32>(paddings);
    // Paddings will be a n,2 array, and we need to detect 4D arrays with the
    // pattern { {0,0}, {a, b}, {c, d}, {0,0} }.
    if (IsConstantTensor(paddings) && paddings_total == 8 &&
        (paddings_data[0] == 0 && paddings_data[1] == 0) &&
        (paddings_data[6] == 0 && paddings_data[7] == 0)) {
      resizing_category = ResizingCategory::kImageStyle;
    }
  }
  const TfLiteTensor* constant_values;
  const TfLiteTensor* input;
  const TfLiteTensor* paddings;
  TfLiteTensor* output;
  int dims;
  ResizingCategory resizing_category;
};

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, NumInputs(node) == 2 || NumInputs(node) == 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  PadContext op_context(context, node);
  TF_LITE_ENSURE_EQ(context, op_context.input->type, op_context.output->type);
  if (op_context.constant_values != nullptr) {
    TF_LITE_ENSURE_EQ(context, op_context.input->type,
                      op_context.constant_values->type);
  }

  // There must be a pair of paddings for each output dimension.
  TF_LITE_ENSURE_EQ(context, GetTensorShape(op_context.paddings).FlatSize(),
                    op_context.output->dims->size * 2);

  // On Micro, outputs must be properly sized by the converter.
  const int32* paddings_data = GetTensorData<int32>(op_context.paddings);
  for (int i = 0; i < op_context.output->dims->size; i++) {
    int output_dim = op_context.output->dims->data[i];
    int expected_dim = op_context.input->dims->data[i] + paddings_data[i * 2] +
                       paddings_data[i * 2 + 1];
    TF_LITE_ENSURE_EQ(context, output_dim, expected_dim);
  }

  // Current implementations rely on the inputs being <= 4D.
  TF_LITE_ENSURE(
      context, op_context.dims <= reference_ops::PadKernelMaxDimensionCount());
  TF_LITE_ENSURE(context, IsConstantTensor(op_context.paddings));
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  PadContext op_context(context, node);

  if (op_context.constant_values != nullptr) {
    // Ensure that constant_values is a scalar.
    TF_LITE_ENSURE_EQ(context, NumElements(op_context.constant_values), 1);
  }

  // Create before and after padding arrays that are accepted by the kernel.
  const int32* paddings_data = GetTensorData<int32>(op_context.paddings);

  tflite::PadParams op_params;
  memset(&op_params, 0, sizeof(PadParams));
  op_params.left_padding_count = op_context.dims;
  op_params.right_padding_count = op_context.dims;

  for (int idx = op_context.dims - 1; idx >= 0; --idx) {
    op_params.left_padding[idx] = paddings_data[idx * 2];
    op_params.right_padding[idx] = paddings_data[idx * 2 + 1];
  }

#define TF_LITE_PAD(type, op_name, scalar, pad_value)                     \
  const scalar pad_value_copy = pad_value;                                \
                                                                          \
  type::op_name(op_params, GetTensorShape(op_context.input),              \
                GetTensorData<scalar>(op_context.input), &pad_value_copy, \
                GetTensorShape(op_context.output),                        \
                GetTensorData<scalar>(op_context.output))
  switch (op_context.input->type) {
    case kTfLiteFloat32: {
      float pad_value = op_context.constant_values == nullptr
                            ? 0.f
                            : *GetTensorData<float>(op_context.constant_values);
      if (op_context.resizing_category == ResizingCategory::kImageStyle) {
        TF_LITE_PAD(reference_ops, PadImageStyle, float, pad_value);
      } else {
        TF_LITE_PAD(reference_ops, Pad, float, pad_value);
      }
    } break;
    case kTfLiteUInt8: {
      uint8_t pad_value;
      if (op_context.constant_values == nullptr) {
        // Quantized Pad requires that 0 is represented in the quantized
        // range.
        TF_LITE_ENSURE(context, op_context.output->params.zero_point >=
                                    std::numeric_limits<uint8_t>::min());
        TF_LITE_ENSURE(context, op_context.output->params.zero_point <=
                                    std::numeric_limits<uint8_t>::max());
        pad_value = static_cast<uint8_t>(op_context.output->params.zero_point);
      } else {
        // Quantized Pad requires that 'constant_values' is represented in the
        // same quantized range as the input and output tensors.
        TF_LITE_ENSURE_EQ(context, op_context.output->params.zero_point,
                          op_context.constant_values->params.zero_point);
        TF_LITE_ENSURE_EQ(
            context, static_cast<double>(op_context.output->params.scale),
            static_cast<double>(op_context.constant_values->params.scale));
        pad_value = *GetTensorData<uint8_t>(op_context.constant_values);
      }
      if (op_context.resizing_category == ResizingCategory::kImageStyle) {
        TF_LITE_PAD(reference_ops, PadImageStyle, uint8_t, pad_value);
      } else {
        TF_LITE_PAD(reference_ops, Pad, uint8_t, pad_value);
      }
    } break;
    case kTfLiteInt8: {
      int8_t pad_value;
      if (op_context.constant_values == nullptr) {
        // Quantized Pad requires that 0 is represented in the quantized
        // range.
        TF_LITE_ENSURE(context, op_context.output->params.zero_point >=
                                    std::numeric_limits<int8_t>::min());
        TF_LITE_ENSURE(context, op_context.output->params.zero_point <=
                                    std::numeric_limits<int8_t>::max());
        pad_value = static_cast<int8_t>(op_context.output->params.zero_point);
      } else {
        // Quantized Pad requires that 'constant_values' is represented in the
        // same quantized range as the input and output tensors.
        TF_LITE_ENSURE_EQ(context, op_context.output->params.zero_point,
                          op_context.constant_values->params.zero_point);
        TF_LITE_ENSURE(context, op_context.output->params.scale ==
                                    op_context.constant_values->params.scale);
        pad_value = *GetTensorData<int8_t>(op_context.constant_values);
      }
      if (op_context.resizing_category == ResizingCategory::kImageStyle) {
        TF_LITE_PAD(reference_ops, PadImageStyle, int8_t, pad_value);
      } else {
        TF_LITE_PAD(reference_ops, Pad, int8_t, pad_value);
      }
    } break;
    case kTfLiteInt32: {
      int32_t pad_value =
          op_context.constant_values == nullptr
              ? 0
              : *GetTensorData<int32_t>(op_context.constant_values);
      TF_LITE_PAD(reference_ops, Pad, int32_t, pad_value);
    } break;
    default:

      context->ReportError(context, "Type %s not currently supported by Pad.",
                           TfLiteTypeGetName(op_context.input->type));
      return kTfLiteError;
  }
#undef TF_LITE_PAD
  return kTfLiteOk;
}

}  // namespace pad

TfLiteRegistration* Register_PAD() {
  static TfLiteRegistration r = {};
  r.prepare = pad::Prepare;
  r.invoke = pad::Eval;
  return &r;
}

// Also register Pad as PadV2.
TfLiteRegistration* Register_PADV2() {
  static TfLiteRegistration r = {};
  r.prepare = pad::Prepare;
  r.invoke = pad::Eval;
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
