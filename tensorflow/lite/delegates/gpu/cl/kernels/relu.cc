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

#include "tensorflow/lite/delegates/gpu/cl/kernels/relu.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/delegates/gpu/cl/precision.h"

namespace tflite {
namespace gpu {
namespace cl {

ReLU::ReLU(const OperationDef& definition, const ReLUAttributes& attr,
           CalculationsPrecision scalar_precision)
    : ElementwiseOperation(definition) {
  std::string min_func;
  if (attr.alpha != 0.0f) {
    min_func = "min(in_out_value * args.alpha, (FLT)(0.0f))";
    if (definition.precision == CalculationsPrecision::F32) {
      args_.AddFloat("alpha", attr.alpha);
    } else {
      args_.AddHalf("alpha", half(attr.alpha));
    }
  } else {
    min_func = "(FLT)(0.0f)";
  }
  if (attr.clip != 0.0f) {
    if (definition.precision == CalculationsPrecision::F32) {
      args_.AddFloat("clip", attr.clip);
    } else {
      args_.AddHalf("clip", half(attr.clip));
    }
    code_ = absl::StrCat("in_out_value = clamp(in_out_value, " + min_func +
                         ", args.clip);");
  } else {
    code_ = absl::StrCat("in_out_value = max(in_out_value, ", min_func, ");");
  }
}

ReLU::ReLU(ReLU&& operation) : ElementwiseOperation(std::move(operation)) {}

ReLU& ReLU::operator=(ReLU&& operation) {
  if (this != &operation) {
    ElementwiseOperation::operator=(std::move(operation));
  }
  return *this;
}

ReLU CreateReLU(const CreationContext& creation_context,
                const OperationDef& definition, const ReLUAttributes& attr) {
  const auto scalar_precision = creation_context.device->IsPowerVR()
                                    ? CalculationsPrecision::F32
                                    : definition.precision;
  ReLU operation(definition, attr, scalar_precision);
  return operation;
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
