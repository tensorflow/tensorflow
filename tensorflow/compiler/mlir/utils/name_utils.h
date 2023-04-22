/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_UTILS_NAME_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_UTILS_NAME_UTILS_H_

#include <string>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Location.h"  // from @llvm-project

namespace mlir {

// Converts characters in name that are considered illegal in TensorFlow Node
// name to '.'.
void LegalizeNodeName(std::string& name);

// Creates a TensorFlow node name from a location.
std::string GetNameFromLoc(Location loc);

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_UTILS_NAME_UTILS_H_
