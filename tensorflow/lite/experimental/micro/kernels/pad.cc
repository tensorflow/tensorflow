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
#include <string.h>
#include <vector>
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/reference/pad.h"
#include "tensorflow/lite/kernels/internal/optimized/pad.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace micro {
namespace pad {

// This file has two implementations of Pad.
enum KernelType {
  kReference,
  kGenericOptimized,
};

struct PadContext {
  PadContext(TfLiteContext* context, TfLiteNode* node) {
    input = GetInput(context, node, 0);
    paddings = GetInput(context, node, 1);
    if (NumInputs(node) == 3) {
      constant_values = GetInput(context, node, 2);
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
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  PadContext op_context(context, node);

  if (op_context.constant_values != nullptr) {
    // Ensure that constant_values is a scalar.
    TF_LITE_ENSURE_EQ(context, NumElements(op_context.constant_values), 1);
  }

  std::vector<int> before_padding;
  std::vector<int> after_padding;
  const int32* paddings_data = GetTensorData<int32>(op_context.paddings);

  for (int idx = op_context.dims - 1; idx >= 0; --idx) {
    before_padding.push_back(paddings_data[idx * 2]);
    after_padding.push_back(paddings_data[idx * 2 + 1]);
  }

#define TF_LITE_PAD(type, op_name, scalar, pad_value)                     \
  TF_LITE_ENSURE(context, before_padding.size() <= 4);                    \
  TF_LITE_ENSURE(context, after_padding.size() <= 4);                     \
  tflite::PadParams op_params;                                            \
  op_params.left_padding_count = before_padding.size();                   \
  op_params.right_padding_count = after_padding.size();                   \
  for (int i = 0; i < op_context.dims; ++i) {                             \
    op_params.left_padding[i] = before_padding[op_context.dims - 1 - i];  \
    op_params.right_padding[i] = after_padding[op_context.dims - 1 - i];  \
  }                                                                       \
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
      // float pad_value = 0.f;
      if (kernel_type == kReference) {
        if (op_context.resizing_category == ResizingCategory::kImageStyle) {
          TF_LITE_PAD(reference_ops, PadImageStyle, float, pad_value);
        } else {
          TF_LITE_PAD(reference_ops, Pad, float, pad_value);
        }
      } else if (kernel_type == kGenericOptimized) {
        if (op_context.resizing_category == ResizingCategory::kImageStyle) {
          TF_LITE_PAD(optimized_ops, PadImageStyle, float, pad_value);
        } else {
          TF_LITE_PAD(optimized_ops, Pad, float, pad_value);
        }
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
        TF_LITE_ENSURE_EQ(context, op_context.output->params.scale,
                          op_context.constant_values->params.scale);
        pad_value = *GetTensorData<uint8_t>(op_context.constant_values);
      }
      if (kernel_type == kReference) {
        if (op_context.resizing_category == ResizingCategory::kImageStyle) {
          TF_LITE_PAD(reference_ops, PadImageStyle, uint8_t, pad_value);
        } else {
          TF_LITE_PAD(reference_ops, Pad, uint8_t, pad_value);
        }
      } else if (kernel_type == kGenericOptimized) {
        if (op_context.resizing_category == ResizingCategory::kImageStyle) {
          TF_LITE_PAD(optimized_ops, PadImageStyle, uint8_t, pad_value);
        } else {
          TF_LITE_PAD(optimized_ops, Pad, uint8_t, pad_value);
        }
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
        TF_LITE_ENSURE_EQ(context, op_context.output->params.scale,
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
      if (kernel_type == kReference) {
        TF_LITE_PAD(reference_ops, Pad, int32_t, pad_value);
      } else if (kernel_type == kGenericOptimized) {
        TF_LITE_PAD(optimized_ops, Pad, int32_t, pad_value);
      }
    } break;
    case kTfLiteInt64: {
      int64_t pad_value =
          op_context.constant_values == nullptr
              ? 0L
              : *GetTensorData<int64_t>(op_context.constant_values);
      if (kernel_type == kReference) {
        TF_LITE_PAD(reference_ops, Pad, int64_t, pad_value);
      } else if (kernel_type == kGenericOptimized) {
        TF_LITE_PAD(optimized_ops, Pad, int64_t, pad_value);
      }
    } break;
    default:
      context->ReportError(context,
                           "Type %d is currently not supported by Pad.",
                           op_context.input->type);
      return kTfLiteError;
  }
#undef TF_LITE_PAD
  return kTfLiteOk;
}

}  // namespace pad

TfLiteRegistration* Register_PAD_REF() {
  static TfLiteRegistration r = {nullptr, nullptr, pad::Prepare,
                                 pad::Eval<pad::kReference>};
  return &r;
}

TfLiteRegistration* Register_PAD_GENERIC_OPT() {
  static TfLiteRegistration r = {nullptr, nullptr, pad::Prepare,
                                 pad::Eval<pad::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_PAD() { return Register_PAD_GENERIC_OPT(); }

// Also register Pad as PadV2.
TfLiteRegistration* Register_PADV2_REF() {
  static TfLiteRegistration r = {nullptr, nullptr, pad::Prepare,
                                 pad::Eval<pad::kReference>};
  return &r;
}

TfLiteRegistration* Register_PADV2_GENERIC_OPT() {
  static TfLiteRegistration r = {nullptr, nullptr, pad::Prepare,
                                 pad::Eval<pad::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_PADV2() { return Register_PADV2_GENERIC_OPT(); }

}  // namespace micro
}  // namespace ops
}  // namespace tflite
