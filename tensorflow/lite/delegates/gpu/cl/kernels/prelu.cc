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

PReLU::PReLU(PReLU&& operation) : ElementwiseOperation(std::move(operation)) {}

PReLU& PReLU::operator=(PReLU&& operation) {
  if (this != &operation) {
    ElementwiseOperation::operator=(std::move(operation));
  }
  return *this;
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
  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
