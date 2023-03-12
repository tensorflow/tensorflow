/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef MLIR_HLO_GML_ST_TRANSFORMS_TRANSFORMS_H
#define MLIR_HLO_GML_ST_TRANSFORMS_TRANSFORMS_H

#include "mlir/IR/Operation.h"

namespace mlir {
namespace gml_st {

constexpr llvm::StringRef kPerfectlyTiledLoopLabel =
    "__perfectly_tiled_loop_label__";

template <typename ShapedTy>
bool hasSingleElement(ShapedTy type) {
  return type.hasStaticShape() && type.getNumElements() == 1;
}
bool hasSingleElementOperandsAndResults(Operation *op);

// Sets the attribute to the `op` that indicates that the op was transformed.
void setLabel(Operation *op, StringRef name);

// Removes the attribute that indicates that it was transformed.
void removeLabel(Operation *op, StringRef name);

// Checks if `op` has the attribute that indicates that it was transformed.
bool hasLabel(Operation *op, StringRef name);

// Checks if `op` has the matching label attribute.
bool hasMatchingLabel(Operation *op, StringRef label);

}  // namespace gml_st
}  // namespace mlir

#endif  // MLIR_HLO_GML_ST_TRANSFORMS_TRANSFORMS_H
