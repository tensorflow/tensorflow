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
  if (attr.alpha != 0.0f) {
    alpha_ = FLT(scalar_precision, attr.alpha);
  }
  if (attr.clip != 0.0f) {
    clip_ = FLT(scalar_precision, attr.clip);
  }
}

ReLU::ReLU(ReLU&& operation)
    : ElementwiseOperation(std::move(operation)),
      alpha_(std::move(operation.alpha_)),
      clip_(std::move(operation.clip_)) {}

ReLU& ReLU::operator=(ReLU&& operation) {
  if (this != &operation) {
    alpha_ = std::move(operation.alpha_);
    clip_ = std::move(operation.clip_);
    ElementwiseOperation::operator=(std::move(operation));
  }
  return *this;
}

void ReLU::SetLinkIndex(int index) {
  alpha_.SetName(absl::StrCat("relu_alpha", index));
  clip_.SetName(absl::StrCat("relu_clip", index));
}

std::string ReLU::GetCoreCode(const LinkingContext& context) const {
  std::string min_func;
  if (!alpha_.Active()) {
    min_func = "(FLT)(0.0f)";
  } else {
    min_func = absl::StrCat("min(", context.var_name, " * (FLT)(",
                            alpha_.GetName(), "), (FLT)(0.0f))");
  }
  if (!clip_.Active()) {
    return absl::StrCat(context.var_name, " = max(", context.var_name, ", ",
                        min_func, ");\n");
  } else {
    return absl::StrCat(context.var_name, " = clamp(", context.var_name,
                        ", " + min_func + ", (FLT)(", clip_.GetName(), "));\n");
  }
}

std::string ReLU::GetArgsDeclaration() const {
  std::string args;
  if (alpha_.Active()) {
    absl::StrAppend(&args, ",\n    ", alpha_.GetDeclaration());
  }
  if (clip_.Active()) {
    absl::StrAppend(&args, ",\n    ", clip_.GetDeclaration());
  }
  return args;
}

absl::Status ReLU::BindArguments(CLKernel* kernel) {
  if (alpha_.Active()) {
    RETURN_IF_ERROR(kernel->SetBytesAuto(alpha_));
  }
  if (clip_.Active()) {
    RETURN_IF_ERROR(kernel->SetBytesAuto(clip_));
  }
  return absl::OkStatus();
}

ReLU CreateReLU(const CreationContext& creation_context,
                const OperationDef& definition, const ReLUAttributes& attr) {
  const auto scalar_precision = creation_context.device->IsPowerVR()
                                    ? CalculationsPrecision::F32
                                    : definition.precision;
  ReLU operation(definition, attr, scalar_precision);
  operation.SetLinkIndex(0);
  return operation;
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
