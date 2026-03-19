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
#include "tensorflow/lite/kernels/internal/reference/pad.h"

#include <stdint.h>

#include <limits>
#include <type_traits>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
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
      constant_values = GetOptionalInputTensor(context, node, 2);
    } else {
      constant_values = nullptr;
    }
    output = GetOutput(context, node, 0);
    dims = NumDimensions(input);
    switch (paddings->type) {
      case kTfLiteInt64: {
        SetResizingCategory<int64_t>(context);
        break;
      }
      case kTfLiteInt32:
        SetResizingCategory<int32_t>(context);
        break;
      case kTfLiteInt8:
        SetResizingCategory<int8_t>(context);
        break;
      case kTfLiteInt16:
        SetResizingCategory<int16_t>(context);
        break;
      case kTfLiteBool:
        SetResizingCategory<bool>(context);
        break;
      default:
        TF_LITE_KERNEL_LOG(context,
                           "Padding type %s is currently not supported by Pad.",
                           TfLiteTypeGetName(paddings->type));
    }
  }

  template <typename padding_integer_type>
  void SetResizingCategory(TfLiteContext* context) {
    const padding_integer_type* paddings_data =
        GetTensorData<padding_integer_type>(paddings);
    resizing_category = ResizingCategory::kGenericResize;
    const int paddings_total = GetTensorShape(paddings).FlatSize();
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

bool CheckPaddingOverflow(PadContext* op_context) {
  if (op_context->paddings->type == kTfLiteInt64) {
    const int64_t* paddings_data = GetTensorData<int64_t>(op_context->paddings);
    if (paddings_data != nullptr) {
      int64_t int32_min =
          static_cast<int64_t>(std::numeric_limits<int32_t>::min());
      int64_t int32_max =
          static_cast<int64_t>(std::numeric_limits<int32_t>::max());
      const int paddings_total =
          GetTensorShape(op_context->paddings).FlatSize();
      for (int idx = 0; idx < paddings_total; ++idx) {
        int64_t padding = paddings_data[idx];
        if (padding < int32_min || padding > int32_max) {
          return true;
        }
      }
    }
  }
  return false;
}

// Helper template function for resizing output array based on the input size
// and padding size. Do not call this directly, call ResizeOutputTensor()
// instead.
template <typename PaddingIntegerType>
TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                PadContext* op_context) {
  if (op_context->paddings->type == kTfLiteInt64) {
    TF_LITE_ENSURE(context, (std::is_same_v<PaddingIntegerType, int64_t>));
  } else if (op_context->paddings->type == kTfLiteInt32) {
    TF_LITE_ENSURE(context, (std::is_same_v<PaddingIntegerType, int32_t>));
  } else if (op_context->paddings->type == kTfLiteInt8) {
    TF_LITE_ENSURE(context, (std::is_same_v<PaddingIntegerType, int8_t>));
  } else {
    TF_LITE_ENSURE(context, (std::is_same_v<PaddingIntegerType, int16_t>));
  }
  // Ensures the paddings array is dims x 2.
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(op_context->paddings, 0),
                    op_context->dims);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(op_context->paddings, 1), 2);

  // Right now we only support paddings between INT32_MIN and INT32_MAX, so
  // we are using int here and below.
  TfLiteIntArray* input_size = op_context->input->dims;
  TfLiteIntArray* output_size = TfLiteIntArrayCopy(input_size);
  const PaddingIntegerType* paddings_data =
      GetTensorData<PaddingIntegerType>(op_context->paddings);
  for (int idx = 0; idx < op_context->dims; ++idx) {
    // Paddings are between INT32_MIN and INT32_MAX.
    int before_padding = static_cast<int>(*paddings_data++);
    int after_padding = static_cast<int>(*paddings_data++);
    TF_LITE_ENSURE_MSG(context, (before_padding >= 0 && after_padding >= 0),
                       "Pad value has to be greater than equal to 0.");
  }
  paddings_data = GetTensorData<PaddingIntegerType>(op_context->paddings);
  for (int idx = 0; idx < op_context->dims; ++idx) {
    // Paddings are between INT32_MIN and INT32_MAX.
    int before_padding = static_cast<int>(*paddings_data++);
    int after_padding = static_cast<int>(*paddings_data++);
    output_size->data[idx] =
        (input_size->data[idx] + before_padding + after_padding);
  }
  return context->ResizeTensor(context, op_context->output, output_size);
}

// Resizes output array based on the input size and padding size. This function
// is callable from both Prepare() and Eval() as long as the caller ensures the
// paddings data is present.
TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                PadContext* op_context) {
  switch (op_context->paddings->type) {
    case kTfLiteInt64: {
      return ResizeOutputTensor<int64_t>(context, op_context);
    }
    case kTfLiteInt32: {
      return ResizeOutputTensor<int32_t>(context, op_context);
    }
    case kTfLiteInt8: {
      return ResizeOutputTensor<int8_t>(context, op_context);
    }
    case kTfLiteInt16: {
      return ResizeOutputTensor<int16_t>(context, op_context);
    }
    case kTfLiteBool: {
      return ResizeOutputTensor<bool>(context, op_context);
    }
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Padding type %s is currently not supported by Pad.",
                         TfLiteTypeGetName(op_context->paddings->type));
  }
  return kTfLiteError;
}

// Helper template function for getting pad params. Do not call this directly,
// call GetPadParams() instead.
template <typename PaddingIntegerType>
tflite::PadParams GetPadParams(TfLiteContext* context,
                               const PadContext& op_context) {
  tflite::PadParams op_params;
  if (!(op_context.paddings->type == kTfLiteInt64 &&
        std::is_same_v<PaddingIntegerType, int64_t>) &&
      !(op_context.paddings->type == kTfLiteInt32 &&
        std::is_same_v<PaddingIntegerType, int32_t>) &&
      !(op_context.paddings->type == kTfLiteInt8 &&
        std::is_same_v<PaddingIntegerType, int8_t>) &&
      !(op_context.paddings->type == kTfLiteInt16 &&
        std::is_same_v<PaddingIntegerType, int16_t>) &&
      !(op_context.paddings->type == kTfLiteBool &&
        std::is_same_v<PaddingIntegerType, bool>)) {
    TF_LITE_KERNEL_LOG(context, "Padding type %s doesn't match typename.",
                       TfLiteTypeGetName(op_context.paddings->type));
    return op_params;
  }
  const PaddingIntegerType* paddings_data =
      GetTensorData<PaddingIntegerType>(op_context.paddings);
  op_params.left_padding_count = op_context.dims;
  op_params.right_padding_count = op_context.dims;
  for (int idx = op_context.dims - 1; idx >= 0; --idx) {
    op_params.left_padding[idx] = static_cast<int32_t>(paddings_data[idx * 2]);
    op_params.right_padding[idx] =
        static_cast<int32_t>(paddings_data[idx * 2 + 1]);
  }
  return op_params;
}

tflite::PadParams GetPadParams(TfLiteContext* context,
                               const PadContext& op_context) {
  switch (op_context.paddings->type) {
    case kTfLiteInt64: {
      return GetPadParams<int64_t>(context, op_context);
    }
    case kTfLiteInt32: {
      return GetPadParams<int32_t>(context, op_context);
    }
    case kTfLiteInt8: {
      return GetPadParams<int8_t>(context, op_context);
    }
    case kTfLiteInt16: {
      return GetPadParams<int16_t>(context, op_context);
    }
    case kTfLiteBool: {
      return GetPadParams<bool>(context, op_context);
    }
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Padding type %s is currently not supported by Pad.",
                         TfLiteTypeGetName(op_context.paddings->type));
  }
  return tflite::PadParams();
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, NumInputs(node) == 2 || NumInputs(node) == 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  PadContext op_context(context, node);
  TF_LITE_ENSURE(context, op_context.output != nullptr);
  if (IsConstantTensor(op_context.paddings)) {
    TF_LITE_ENSURE_MSG(context, !CheckPaddingOverflow(&op_context),
                       "INT64 padding overflow. Only support value between "
                       "INT32_MIN and INT32_MAX.");
  }
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.input->type,
                          op_context.output->type);
  if (op_context.constant_values != nullptr) {
    TF_LITE_ENSURE_TYPES_EQ(context, op_context.input->type,
                            op_context.constant_values->type);
  }

  // Ensure we do not exceed maximum dimension count.
  TF_LITE_ENSURE(
      context, op_context.dims <= reference_ops::PadKernelMaxDimensionCount());

  // Exit early if paddings is a non-const tensor or the given input is an
  // unranked input. Set output tensor to dynamic so output size can be
  // determined in Eval.
  if (NumDimensions(op_context.input) == 0 ||
      !IsConstantOrPersistentTensor(op_context.paddings)) {
    SetTensorToDynamic(op_context.output);
    return kTfLiteOk;
  }
  return ResizeOutputTensor(context, &op_context);
}

template <typename integer_type>
TfLiteStatus EvalInt(TfLiteContext* context, const PadContext& op_context,
                     const tflite::PadParams& op_params) {
  integer_type pad_value;
  if (op_context.constant_values == nullptr) {
    // Quantized Pad requires that 0 is represented in the quantized
    // range.
    TF_LITE_ENSURE(context, op_context.output->params.zero_point >=
                                std::numeric_limits<integer_type>::min());
    TF_LITE_ENSURE(context, op_context.output->params.zero_point <=
                                std::numeric_limits<integer_type>::max());
    pad_value = static_cast<integer_type>(op_context.output->params.zero_point);
  } else {
    // Quantized Pad requires that 'constant_values' is represented in the
    // same quantized range as the input and output tensors.
    TF_LITE_ENSURE_EQ(context, op_context.output->params.zero_point,
                      op_context.constant_values->params.zero_point);
    TF_LITE_ENSURE_EQ(context, op_context.output->params.scale,
                      op_context.constant_values->params.scale);
    pad_value = *GetTensorData<integer_type>(op_context.constant_values);
  }
  const integer_type pad_value_copy = pad_value;
  if (op_context.resizing_category == ResizingCategory::kImageStyle) {
    optimized_ops::PadImageStyle(
        op_params, GetTensorShape(op_context.input),
        GetTensorData<integer_type>(op_context.input), &pad_value_copy,
        GetTensorShape(op_context.output),
        GetTensorData<integer_type>(op_context.output));
  } else {
    optimized_ops::Pad(op_params, GetTensorShape(op_context.input),
                       GetTensorData<integer_type>(op_context.input),
                       &pad_value_copy, GetTensorShape(op_context.output),
                       GetTensorData<integer_type>(op_context.output));
  }

  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  PadContext op_context(context, node);

  TF_LITE_ENSURE_MSG(context, !CheckPaddingOverflow(&op_context),
                     "INT64 padding overflow. Only support value between "
                     "INT32_MIN and INT32_MAX.");

  if (op_context.constant_values != nullptr) {
    // Ensure that constant_values is a scalar.
    TF_LITE_ENSURE_EQ(context, NumElements(op_context.constant_values), 1);
  }

  // Resize the output tensor if the output tensor is dynamic.
  if (IsDynamicTensor(op_context.output)) {
    TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, &op_context));
  }

  TF_LITE_ENSURE(
      context, op_context.dims <= reference_ops::PadKernelMaxDimensionCount());

  tflite::PadParams op_params = GetPadParams(context, op_context);

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
    case kTfLiteFloat16: {
      Eigen::half pad_value =
          op_context.constant_values == nullptr
              ? static_cast<Eigen::half>(0.f)
              : *GetTensorData<Eigen::half>(op_context.constant_values);
      if (kernel_type == kReference) {
        if (op_context.resizing_category == ResizingCategory::kImageStyle) {
          TF_LITE_PAD(reference_ops, PadImageStyle, Eigen::half, pad_value);
        } else {
          TF_LITE_PAD(reference_ops, Pad, Eigen::half, pad_value);
        }
      } else if (kernel_type == kGenericOptimized) {
        if (op_context.resizing_category == ResizingCategory::kImageStyle) {
          TF_LITE_PAD(optimized_ops, PadImageStyle, Eigen::half, pad_value);
        } else {
          TF_LITE_PAD(optimized_ops, Pad, Eigen::half, pad_value);
        }
      }
    } break;
    case kTfLiteBFloat16: {
      Eigen::bfloat16 pad_value =
          op_context.constant_values == nullptr
              ? static_cast<Eigen::bfloat16>(0.f)
              : *GetTensorData<Eigen::bfloat16>(op_context.constant_values);
      if (kernel_type == kReference) {
        if (op_context.resizing_category == ResizingCategory::kImageStyle) {
          TF_LITE_PAD(reference_ops, PadImageStyle, Eigen::bfloat16, pad_value);
        } else {
          TF_LITE_PAD(reference_ops, Pad, Eigen::bfloat16, pad_value);
        }
      } else if (kernel_type == kGenericOptimized) {
        if (op_context.resizing_category == ResizingCategory::kImageStyle) {
          TF_LITE_PAD(optimized_ops, PadImageStyle, Eigen::bfloat16, pad_value);
        } else {
          TF_LITE_PAD(optimized_ops, Pad, Eigen::bfloat16, pad_value);
        }
      }
    } break;
    case kTfLiteUInt8: {
      EvalInt<uint8_t>(context, op_context, op_params);
    } break;
    case kTfLiteInt8: {
      if (op_context.input->quantization.type != kTfLiteNoQuantization) {
        EvalInt<int8_t>(context, op_context, op_params);
      } else {
        int8_t pad_value =
            op_context.constant_values == nullptr
                ? 0
                : *GetTensorData<int8_t>(op_context.constant_values);
        if (kernel_type == kReference) {
          TF_LITE_PAD(reference_ops, Pad, int8_t, pad_value);
        } else if (kernel_type == kGenericOptimized) {
          TF_LITE_PAD(optimized_ops, Pad, int8_t, pad_value);
        }
      }
    } break;
    case kTfLiteInt16: {
      if (op_context.input->quantization.type != kTfLiteNoQuantization) {
        EvalInt<int16_t>(context, op_context, op_params);
      } else {
        int16_t pad_value =
            op_context.constant_values == nullptr
                ? 0
                : *GetTensorData<int16_t>(op_context.constant_values);
        if (kernel_type == kReference) {
          TF_LITE_PAD(reference_ops, Pad, int16_t, pad_value);
        } else if (kernel_type == kGenericOptimized) {
          TF_LITE_PAD(optimized_ops, Pad, int16_t, pad_value);
        }
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
    case kTfLiteBool: {
      bool pad_value = op_context.constant_values == nullptr
                           ? false
                           : *GetTensorData<bool>(op_context.constant_values);
      if (kernel_type == kReference) {
        TF_LITE_PAD(reference_ops, Pad, bool, pad_value);
      } else if (kernel_type == kGenericOptimized) {
        TF_LITE_PAD(optimized_ops, Pad, bool, pad_value);
      }
    } break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s is currently not supported by Pad.",
                         TfLiteTypeGetName(op_context.input->type));
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

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
