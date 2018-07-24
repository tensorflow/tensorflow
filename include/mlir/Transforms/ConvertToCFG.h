//===- ConvertToCFG.h - Convert ML functions to CFG ones --------*- C++ -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file defines APIs to convert ML functions into CFG functions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_CONVERTTOCFG_H
#define MLIR_TRANSFORMS_CONVERTTOCFG_H

namespace mlir {
class Module;

/// Replaces all ML functions in the module with equivalent CFG functions.
/// Function references are appropriately patched to refer only
/// to CFG functions.
void convertToCFG(Module *module);

} // namespace mlir
#endif // MLIR_TRANSFORMS_CONVERTTOCFG_H
