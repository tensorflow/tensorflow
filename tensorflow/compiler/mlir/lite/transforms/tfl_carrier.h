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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_TFL_CARRIER_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_TFL_CARRIER_H_

#include <optional>

#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project

namespace mlir {
namespace TFL {

// The StableHLO carrier convention for TFL ops: a `custom_call` whose
// call_target_name is the TFL op name (`tfl.*`) and whose backend_config is
// the op's attribute dictionary serialized in MLIR syntax (the serialized
// empty dictionary if the op has no attributes). Shared by HLO->TFL
// legalization (legalize_hlo_conversions/custom_call.cc) and passes that need
// to read carrier attributes before legalization.
//
// Returns the parsed dictionary, or null if `backend_config` is absent, not a
// string, or does not parse as a dictionary.
DictionaryAttr ParseSerializedTFLOpAttributes(
    std::optional<Attribute> backend_config, MLIRContext* ctx);

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_TFL_CARRIER_H_
