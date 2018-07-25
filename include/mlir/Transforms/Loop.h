//===- Loop.h - Loop Transformations ----------------------------*- C++ -*-===//
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
// This header file defines prototypes that expose passes in the loop
// transformation library.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_LOOP_H
#define MLIR_TRANSFORMS_LOOP_H

namespace mlir {

class MLFunctionPass;

/// A loop unrolling pass.
MLFunctionPass *createLoopUnrollPass();

} // end namespace mlir

#endif // MLIR_TRANSFORMS_LOOP_H
