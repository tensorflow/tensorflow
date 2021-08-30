/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <stdint.h>
#include <stdlib.h>

#include <cmath>
#include <limits>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/elementwise_portable.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace elementwise {
namespace {

typedef bool (*IsSupportedType)(TfLiteType);
TfLiteStatus GenericPrepareLite(TfLiteContext* context, TfLiteNode* node,
                                IsSupportedType is_supported_type,
                                const char* op_name) {
  TF_LITE_ENSURE_OK(context,
                    GenericPrepare(context, node, is_supported_type, op_name));

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

void* ElementWiseQuantizedInit(TfLiteContext* context, const char* buffer,
                               size_t length) {
  return new OpData();
}

void ElementWiseQuantizedFree(TfLiteContext* context, void* buffer) {
  delete static_cast<OpData*>(buffer);
}

}  // namespace
}  // namespace elementwise

// Given a function...
// template<int T>
// int Foo(int b)
//
// typedef int(*Bar)(int);
//
// MSVC2015 will not see Foo<10> as the same type as Bar.
//
// This works around the issue by instantiating wrapper methods around
// elementwise::GenericPrepare() rather than using a templated
// elementwise::GenericPrepare method.
#define GENERIC_PREPARE(function_name, is_supported_type_function, type_name) \
  static TfLiteStatus function_name(TfLiteContext* context,                   \
                                    TfLiteNode* node) {                       \
    return elementwise::GenericPrepareLite(                                   \
        context, node, is_supported_type_function, type_name);                \
  }

GENERIC_PREPARE(PrepareAbs, elementwise::IsAbsSupportedType,
                elementwise::kAbsName)

TfLiteRegistration* Register_ABS() {
  static TfLiteRegistration r = {elementwise::ElementWiseQuantizedInit,
                                 elementwise::ElementWiseQuantizedFree,
                                 PrepareAbs, elementwise::AbsEval};
  return &r;
}

GENERIC_PREPARE(PrepareSin, elementwise::IsNumericSupportedType, "Sin")

TfLiteRegistration* Register_SIN() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr, PrepareSin,
                                 elementwise::SinEval};
  return &r;
}

GENERIC_PREPARE(PrepareCos, elementwise::IsNumericSupportedType, "Cos")

TfLiteRegistration* Register_COS() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr, PrepareCos,
                                 elementwise::CosEval};
  return &r;
}

GENERIC_PREPARE(PrepareLog, elementwise::IsNumericSupportedType, "Log")

TfLiteRegistration* Register_LOG() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr, PrepareLog,
                                 elementwise::LogEval};
  return &r;
}

GENERIC_PREPARE(PrepareSqrt, elementwise::IsNumericSupportedType, "Sqrt")

TfLiteRegistration* Register_SQRT() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 PrepareSqrt, elementwise::SqrtEval};
  return &r;
}

GENERIC_PREPARE(PrepareRsqrt, elementwise::IsRsqrtSupportedType,
                elementwise::kRsqrtName)

TfLiteRegistration* Register_RSQRT() {
  static TfLiteRegistration r = {elementwise::ElementWiseQuantizedInit,
                                 elementwise::ElementWiseQuantizedFree,
                                 PrepareRsqrt, elementwise::RsqrtEval};
  return &r;
}

GENERIC_PREPARE(PrepareSquare, elementwise::IsNumericSupportedType, "Square")

TfLiteRegistration* Register_SQUARE() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 PrepareSquare, elementwise::SquareEval};
  return &r;
}

GENERIC_PREPARE(PrepareNot, elementwise::IsLogicalSupportedType, "Not")

TfLiteRegistration* Register_LOGICAL_NOT() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr, PrepareNot,
                                 elementwise::LogicalNotEval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
