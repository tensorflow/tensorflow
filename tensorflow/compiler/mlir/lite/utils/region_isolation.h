/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_UTILS_REGION_ISOLATION_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_UTILS_REGION_ISOLATION_H_

#include <optional>

#include "llvm/ADT/SetVector.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace TFL {

// Isolates op's contained regions. Replaces all references to values defined
// above these (single block) regions with a block argument. The union of all
// values referenced this way is returned. Each region will have an identical
// signature, which is the types of the returned vector in the same order.
// NOTE: Critically, llvm::SetVector iterates deterministically in order of
// insertion.
std::optional<llvm::SetVector<Value>> IsolateRegions(Operation* op_with_regions,
                                                     OpBuilder& b);

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_REGION_ISOLATION_H_
