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

#include "tensorflow/lite/delegates/gpu/cl/kernels/abs.h"

#include <string>

#include "absl/strings/str_cat.h"

namespace tflite {
namespace gpu {
namespace cl {

Abs::Abs(Abs&& operation) : ElementwiseOperation(std::move(operation)) {}

Abs& Abs::operator=(Abs&& operation) {
  if (this != &operation) {
    ElementwiseOperation::operator=(std::move(operation));
  }
  return *this;
}

std::string Abs::GetCoreCode(const LinkingContext& context) const {
  return absl::StrCat(context.var_name, " = fabs(", context.var_name, ");\n");
}

Abs CreateAbs(const OperationDef& definition) {
  Abs operation(definition);
  operation.SetLinkIndex(0);
  return operation;
}

}  // namespace cl
}  // namespace gpu
}  // namespace tflite
