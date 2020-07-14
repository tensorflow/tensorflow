/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/cl/kernels/quantize_and_dequantize.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/types/variant.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace cl {

QuantizeAndDequantize::QuantizeAndDequantize(
    const OperationDef& definition, const QuantizeAndDequantizeAttributes& attr,
    CalculationsPrecision scalar_precision)
    : ElementwiseOperation(definition) {
  if (definition.precision == CalculationsPrecision::F32) {
    args_.AddFloat("min", attr.min);
    args_.AddFloat("max", attr.max);
    args_.AddFloat("scale", attr.scale);
  } else {
    args_.AddHalf("min", half(attr.min));
    args_.AddHalf("max", half(attr.max));
    args_.AddHalf("scale", half(attr.scale));
  }
  code_ = R"(
FLT4 clamped_value = min((FLT4)(args.max), max((FLT4)(args.min), in_out_value));
FLT4 quantized_value = round((clamped_value - (FLT4)(args.min)) / (FLT4)(args.scale));
FLT4 dequantized_value = quantized_value * (FLT4)(args.scale) + (FLT4)(args.min);
in_out_value = dequantized_value;)";
}

QuantizeAndDequantize::QuantizeAndDequantize(QuantizeAndDequantize&& operation)
    : ElementwiseOperation(std::move(operation)) {}

QuantizeAndDequantize& QuantizeAndDequantize::operator=(
    QuantizeAndDequantize&& operation) {
  if (this != &operation) {
    ElementwiseOperation::operator=(std::move(operation));
  }
  return *this;
}

absl::Status CreateQuantizeAndDequantize(
    const CreationContext& creation_context, const OperationDef& definition,
    const QuantizeAndDequantizeAttributes& attr,
    QuantizeAndDequantize* result) {
  const auto scalar_precision = creation_context.device->IsPowerVR()
                                    ? CalculationsPrecision::F32
                                    : definition.precision;
  const bool is_fp16 = definition.precision == CalculationsPrecision::F16 ||
                       definition.precision == CalculationsPrecision::F32_F16;
  if (is_fp16 && attr.scale < 0.000062f) {
    // The smallest positive normal number for Half-precision floating-point
    // format is 2^-14 ~ 0.000062f. Therefore, if the scale is lesser than this
    // number, we just reset it accordingly.
    QuantizeAndDequantizeAttributes adjusted_attr = attr;
    adjusted_attr.scale = 0.000062f;
    *result =
        QuantizeAndDequantize(definition, adjusted_attr, scalar_precision);
  } else {
    *result = QuantizeAndDequantize(definition, attr, scalar_precision);
  }
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
