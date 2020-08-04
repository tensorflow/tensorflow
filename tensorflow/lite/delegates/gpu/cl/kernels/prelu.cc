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

absl::Status CreatePReLU(const CreationContext& creation_context,
                         const OperationDef& definition,
                         const PReLUAttributes& attr, GPUOperation* result) {
  *result = GPUOperation(definition);
  result->elementwise_ = true;
  if (attr.clip != 0) {
    if (definition.precision == CalculationsPrecision::F32) {
      result->args_.AddFloat("clip", attr.clip);
    } else {
      result->args_.AddHalf("clip", half(attr.clip));
    }
    result->code_ =
        "in_out_value = clamp(in_out_value, (FLT4)(0.0f), (FLT4)(args.clip)) + "
        "min((FLT4)(0.0f), in_out_value) * args.alpha.Read(S_COORD);";
  } else {
    result->code_ =
        "in_out_value = max((FLT4)(0.0f), in_out_value) + min((FLT4)(0.0f), "
        "in_out_value) * args.alpha.Read(S_COORD);";
  }

  auto alpha =
      absl::get_if<tflite::gpu::Tensor<Linear, DataType::FLOAT32>>(&attr.alpha);
  if (!alpha) {
    return absl::InvalidArgumentError("Alpha is missing");
  }
  TensorLinearDescriptor desc;
  desc.storage_type =
      DeduceLinearStorageType(definition.GetPrimaryStorageType());
  desc.element_type = definition.GetPrimaryDataType();

  LinearStorage lt;
  RETURN_IF_ERROR(
      CreateLinearStorage(desc, *alpha, creation_context.context, &lt));
  result->args_.AddObject("alpha", AccessType::READ,
                          absl::make_unique<LinearStorage>(std::move(lt)),
                          absl::make_unique<TensorLinearDescriptor>(desc));

  return absl::OkStatus();
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
