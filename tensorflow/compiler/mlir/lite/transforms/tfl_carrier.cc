/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/transforms/tfl_carrier.h"

#include <optional>

#include "mlir/AsmParser/AsmParser.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace TFL {

DictionaryAttr ParseSerializedTFLOpAttributes(
    std::optional<Attribute> backend_config, MLIRContext* ctx) {
  if (!backend_config) return nullptr;
  auto serialized = dyn_cast_or_null<StringAttr>(*backend_config);
  if (!serialized) return nullptr;
  return dyn_cast_or_null<DictionaryAttr>(
      parseAttribute(serialized.getValue(), ctx));
}

}  // namespace TFL
}  // namespace mlir
