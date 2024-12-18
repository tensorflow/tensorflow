// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/vendors/examples/example_conversion_impl.h"

#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_element_type.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/backend_ir.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/conversion.h"
#include "tensorflow/lite/experimental/litert/vendors/examples/example_ir.h"

namespace litert::example {

TensorConverter<ExampleTensor> MakeTensorConverter(
    TensorAllocator<ExampleTensor> alloc) {
  return [alloc](const Tensor& litert_tensor) -> Expected<ExampleTensor*> {
    auto& tensor = *alloc();
    tensor.name = litert_tensor.Name();

    auto litert_type = litert_tensor.RankedTensorType();
    if (!litert_type) {
      return Error(litert_type.Error().Status());
    }

    const auto litert_dims = litert_type->Layout().Dimensions();

    tensor.dims.assign(litert_dims.cbegin(), litert_dims.cend());

    switch (litert_tensor.RankedTensorType()->ElementType()) {
      case ElementType::Float32:
        tensor.type = ExampleTensorType::FLOAT;
        break;
      case ElementType::Int32:
        tensor.type = ExampleTensorType::INT;
        break;
      default:
        return Error(kLiteRtStatusErrorInvalidArgument);
    }

    return &tensor;
  };
}

ExampleTypes::Legalizations MakeAllLegalizations() {
  ExampleTypes::Legalizations legalizations;
  legalizations.push_back(ExampleLegalizeMul::Make());
  legalizations.push_back(ExampleLegalizeAdd::Make());
  return legalizations;
}

}  // namespace litert::example
