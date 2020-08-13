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

#ifndef TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_UTILS_CONVERT_OP_FOLDER_H_
#define TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_UTILS_CONVERT_OP_FOLDER_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/StandardTypes.h"

namespace mlir {
namespace hlo {

// Converts the given elements attr to the specified elements type.
// Requires type of the elements and new_type to be either integer or float
// type.
mlir::ElementsAttr ConvertElementsAttr(const mlir::ElementsAttr& elements,
                                       mlir::Type new_type);
}  // namespace hlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_UTILS_CONVERT_OP_FOLDER_H_
