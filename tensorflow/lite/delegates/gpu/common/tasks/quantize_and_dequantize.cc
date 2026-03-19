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

#include "tensorflow/lite/delegates/gpu/common/tasks/quantize_and_dequantize.h"

#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/types/variant.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {

GPUOperation CreateQuantizeAndDequantize(
    const OperationDef& definition,
    const QuantizeAndDequantizeAttributes& attr) {
  QuantizeAndDequantizeAttributes adjusted_attr = attr;
  const bool is_fp16 = definition.precision == CalculationsPrecision::F16 ||
                       definition.precision == CalculationsPrecision::F32_F16;
  if (is_fp16 && attr.scale < 0.000062f) {
    // The smallest positive normal number for Half-precision floating-point
    // format is 2^-14 ~ 0.000062f. Therefore, if the scale is lesser than this
    // number, we just reset it accordingly.
    adjusted_attr.scale = 0.000062f;
  }

  ElementwiseDescriptor op_desc;
  if (definition.precision == CalculationsPrecision::F32) {
    op_desc.args.AddFloat("min", adjusted_attr.min);
    op_desc.args.AddFloat("max", adjusted_attr.max);
    op_desc.args.AddFloat("scale", adjusted_attr.scale);
  } else {
    op_desc.args.AddHalf("min", half(adjusted_attr.min));
    op_desc.args.AddHalf("max", half(adjusted_attr.max));
    op_desc.args.AddHalf("scale", half(adjusted_attr.scale));
  }
  op_desc.code = R"(
FLT4 clamped_value = min(INIT_FLT4(args.max), max(INIT_FLT4(args.min), in_value));
FLT4 quantized_value = round((clamped_value - INIT_FLT4(args.min)) / INIT_FLT4(args.scale));
FLT4 dequantized_value = quantized_value * INIT_FLT4(args.scale) + INIT_FLT4(args.min);
out_value = dequantized_value;)";

  return CreateGpuOperation(definition, std::move(op_desc));
}

}  // namespace gpu
}  // namespace tflite
