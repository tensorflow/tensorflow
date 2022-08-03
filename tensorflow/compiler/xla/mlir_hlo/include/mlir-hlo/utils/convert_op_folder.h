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

#ifndef MLIR_HLO_UTILS_CONVERT_OP_FOLDER_H
#define MLIR_HLO_UTILS_CONVERT_OP_FOLDER_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace hlo {

// Converts the given elements attr to the specified elements type.
// Requires type of the elements and new_type to be either integer or float
// type.
mlir::ElementsAttr convertElementsAttr(const mlir::ElementsAttr& elements,
                                       mlir::Type newType);
}  // namespace hlo
}  // namespace mlir

#endif  // MLIR_HLO_UTILS_CONVERT_OP_FOLDER_H
