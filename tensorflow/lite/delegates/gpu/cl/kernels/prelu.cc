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

#include "tensorflow/lite/delegates/gpu/cl/kernels/prelu.h"

#include "absl/strings/str_cat.h"
#include "absl/types/variant.h"
#include "tensorflow/lite/delegates/gpu/cl/kernels/util.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace cl {

PReLU::PReLU(const OperationDef& definition, const PReLUAttributes& attr,
             CalculationsPrecision scalar_precision)
    : ElementwiseOperation(definition) {
  if (attr.clip != 0) {
    clip_ = FLT(scalar_precision, attr.clip);
    if (definition.precision == CalculationsPrecision::F32) {
      args_.AddFloat("clip", attr.clip);
    } else {
      args_.AddHalf("clip", half(attr.clip));
    }
    code_ =
        "in_out_value = clamp(in_out_value, (FLT4)(0.0f), (FLT4)(args.clip)) + "
        "min((FLT4)(0.0f), in_out_value) * args.alpha.Read(S_COORD);";
  } else {
    code_ =
        "in_out_value = max((FLT4)(0.0f), in_out_value) + min((FLT4)(0.0f), "
        "in_out_value) * args.alpha.Read(S_COORD);";
  }
}

PReLU::PReLU(PReLU&& operation)
    : ElementwiseOperation(std::move(operation)),
      clip_(std::move(operation.clip_)),
      alpha_(std::move(operation.alpha_)) {}

PReLU& PReLU::operator=(PReLU&& operation) {
  if (this != &operation) {
    clip_ = std::move(operation.clip_);
    alpha_ = std::move(operation.alpha_);
    ElementwiseOperation::operator=(std::move(operation));
  }
  return *this;
}

void PReLU::SetLinkIndex(int index) {
  clip_.SetName(absl::StrCat("prelu_clip", index));
  alpha_.SetName(absl::StrCat("prelu_alpha_", index));
}

std::string PReLU::GetCoreCode(const LinkingContext& context) const {
  if (!clip_.Active()) {
    return absl::StrCat(context.var_name, " = max((FLT4)(0.0f), ",
                        context.var_name, ") + min((FLT4)(0.0f), ",
                        context.var_name, ") * ",
                        alpha_.ReadLinearFLT4(context.s_coord), ";\n");
  } else {
    return absl::StrCat(context.var_name, " = clamp(", context.var_name,
                        ", (FLT4)(0.0f), (FLT4)(", clip_.GetName(),
                        ")) + min((FLT4)(0.0f), ", context.var_name, ") * ",
                        alpha_.ReadLinearFLT4(context.s_coord), ";\n");
  }
}

std::string PReLU::GetArgsDeclaration() const {
  std::string args = absl::StrCat(",\n    ", alpha_.GetDeclaration());
  if (clip_.Active()) {
    absl::StrAppend(&args, ",\n    ", clip_.GetDeclaration());
  }
  return args;
}

absl::Status PReLU::BindArguments(CLKernel* kernel) {
  RETURN_IF_ERROR(kernel->SetMemoryAuto(alpha_.GetMemoryPtr()));
  if (clip_.Active()) {
    RETURN_IF_ERROR(kernel->SetBytesAuto(clip_));
  }
  return absl::OkStatus();
}

absl::Status CreatePReLU(const CreationContext& creation_context,
                         const OperationDef& definition,
                         const PReLUAttributes& attr, PReLU* result) {
  auto alpha =
      absl::get_if<tflite::gpu::Tensor<Linear, DataType::FLOAT32>>(&attr.alpha);
  if (!alpha) {
    return absl::InvalidArgumentError("Alpha is missing");
  }
  const auto scalar_precision = creation_context.device->IsPowerVR()
                                    ? CalculationsPrecision::F32
                                    : definition.precision;
  *result = PReLU(definition, attr, scalar_precision);
  RETURN_IF_ERROR(result->UploadParameters(*alpha, creation_context.context));
  result->SetLinkIndex(0);
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
