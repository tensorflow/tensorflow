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

#include "tensorflow/lite/delegates/gpu/common/tasks/relu.h"

#include <string>
#include <utility>

#include "absl/strings/str_cat.h"

namespace tflite {
namespace gpu {

ElementwiseDescriptor CreateReLU(const ReLUAttributes& attr,
                                 CalculationsPrecision precision) {
  ElementwiseDescriptor result;
  std::string min_func;
  if (attr.alpha != 0.0f) {
    min_func = "min(in_value * args.alpha, INIT_FLT(0.0f))";
    if (precision == CalculationsPrecision::F32) {
      result.args.AddFloat("alpha", attr.alpha);
    } else {
      result.args.AddHalf("alpha", half(attr.alpha));
    }
  } else {
    min_func = "INIT_FLT4(args.activation_min)";
    if (precision == CalculationsPrecision::F32) {
      result.args.AddFloat("activation_min", attr.activation_min);
    } else {
      result.args.AddHalf("activation_min", half(attr.activation_min));
    }
  }
  if (attr.activation_max != 0.0f) {
    if (precision == CalculationsPrecision::F32) {
      result.args.AddFloat("activation_max", attr.activation_max);
    } else {
      result.args.AddHalf("activation_max", half(attr.activation_max));
    }
    result.code = absl::StrCat("out_value = clamp(in_value, " + min_func +
                               ", INIT_FLT4(args.activation_max));");
  } else {
    result.code = absl::StrCat("out_value = max(in_value, ", min_func, ");");
  }
  return result;
}

GPUOperation CreateReLU(const OperationDef& definition,
                        const ReLUAttributes& attr) {
  ElementwiseDescriptor op_desc = CreateReLU(attr, definition.precision);
  return CreateGpuOperation(definition, std::move(op_desc));
}

}  // namespace gpu
}  // namespace tflite
