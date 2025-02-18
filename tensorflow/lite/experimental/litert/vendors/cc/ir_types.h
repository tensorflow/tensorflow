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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_IR_TYPES_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_IR_TYPES_H_

#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/backend_ir.h"
#include "tensorflow/lite/experimental/litert/vendors/cc/conversion.h"

namespace litert {

// Holds particular backends IR template aliases for convenience.
template <class BackendOp, class BackendTensor>
struct IrTypes {
  using Op = BackendOp;
  using Tensor = BackendTensor;
  using OpAllocator = OpAllocator<Op>;
  using TensorAllocator = TensorAllocator<Tensor>;
  using GraphBuilder = BackendGraphBuilder<Op, Tensor>;
  using GeneralConversionResult = GeneralConversionResult<Op, Tensor>;
  using SimpleConversionResult = SimpleConversionResult<Op>;
  using ConversionResult = Expected<ConversionResult<Op, Tensor>>;
  using Legalization = Legalization<Op, Tensor>;
  using Legalizations = Legalizations<Op, Tensor>;
  using LegalizationMap = LegalizationMap<Op, Tensor>;
  using TensorConverter = TensorConverter<Tensor>;
  using TensorResult = Expected<Tensor*>;
  using TensorConverterFactory = TensorConverterFactory<Tensor>;
  using TensorMap = TensorMap<Tensor>;
  using Capability = Capability<Op>;
  // NOLINTNEXTLINE
  inline static auto MakeLegalizationMap =
      litert::MakeLegalizationMap<Op, Tensor>;
};

}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_CC_IR_TYPES_H_
