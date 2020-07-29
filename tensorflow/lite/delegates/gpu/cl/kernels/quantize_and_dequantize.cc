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
  min_ = FLT(scalar_precision, attr.min);
  max_ = FLT(scalar_precision, attr.max);
  scale_ = FLT(scalar_precision, attr.scale);
}

QuantizeAndDequantize::QuantizeAndDequantize(QuantizeAndDequantize&& operation)
    : ElementwiseOperation(std::move(operation)),
      min_(std::move(operation.min_)),
      max_(std::move(operation.max_)),
      scale_(std::move(operation.scale_)) {}

QuantizeAndDequantize& QuantizeAndDequantize::operator=(
    QuantizeAndDequantize&& operation) {
  if (this != &operation) {
    min_ = std::move(operation.min_);
    max_ = std::move(operation.max_);
    scale_ = std::move(operation.scale_);
    ElementwiseOperation::operator=(std::move(operation));
  }
  return *this;
}

void QuantizeAndDequantize::SetLinkIndex(int index) {
  min_.SetName(absl::StrCat("quantize_and_dequantize_min_", index));
  max_.SetName(absl::StrCat("quantize_and_dequantize_max_", index));
  scale_.SetName(absl::StrCat("quantize_and_dequantize_scale_", index));
}

std::string QuantizeAndDequantize::GetCoreCode(
    const LinkingContext& context) const {
  std::string scale_string, max_string, min_string;
  if (!scale_.Active()) {
    scale_string = "(FLT4)(1.0f)";
  } else {
    scale_string = absl::StrCat("(FLT4)(", scale_.GetName(), ")");
  }
  if (!max_.Active()) {
    max_string = "(FLT4)(0.0f)";
  } else {
    max_string = absl::StrCat("(FLT4)(", max_.GetName(), ")");
  }
  if (!min_.Active()) {
    min_string = "(FLT4)(0.0f)";
  } else {
    min_string = absl::StrCat("(FLT4)(", min_.GetName(), ")");
  }
  std::string clamped_value = absl::StrCat(
      "min(", max_string, ", max(", min_string, ", ", context.var_name, "))");
  std::string quantized_value = absl::StrCat(
      "round((", clamped_value, " - ", min_string, ") / ", scale_string, ")");
  std::string dequantized_value =
      absl::StrCat(quantized_value, " * ", scale_string, " + ", min_string);

  return absl::StrCat(context.var_name, " = ", dequantized_value, ";\n");
}

std::string QuantizeAndDequantize::GetArgsDeclaration() const {
  return absl::StrCat(",\n    ", min_.GetDeclaration(), ",\n    ",
                      max_.GetDeclaration(), ",\n    ",
                      scale_.GetDeclaration());
}

absl::Status QuantizeAndDequantize::BindArguments(CLKernel* kernel) {
  RETURN_IF_ERROR(kernel->SetBytesAuto(min_));
  RETURN_IF_ERROR(kernel->SetBytesAuto(max_));
  RETURN_IF_ERROR(kernel->SetBytesAuto(scale_));
  return absl::OkStatus();
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
  result->SetLinkIndex(0);
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
