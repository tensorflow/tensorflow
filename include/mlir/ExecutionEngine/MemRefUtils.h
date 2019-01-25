//===- MemRefUtils.h - MLIR runtime utilities for memrefs -------*- C++ -*-===//
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
// This is a set of utilities to working with objects of memref type in an JIT
// context using the MLIR execution engine.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_EXECUTIONENGINE_MEMREFUTILS_H_
#define MLIR_EXECUTIONENGINE_MEMREFUTILS_H_

#include "mlir/Support/LLVM.h"

namespace llvm {
template <typename T> class Expected;
}

namespace mlir {

class Function;

/// Simple memref descriptor class compatible with the ABI of functions emitted
/// by MLIR to LLVM IR conversion for statically-shaped memrefs of float type.
struct StaticFloatMemRef {
  float *data;
};

/// Given an MLIR function that takes only statically-shaped memrefs with
/// element type f32, allocate the memref descriptor and the data storage for
/// each of the arguments, initialize the storage with `initialValue`, and
/// return a list of type-erased descriptor pointers.
llvm::Expected<SmallVector<void *, 8>>
allocateMemRefArguments(const Function *func, float initialValue = 0.0);

/// Free a list of type-erased descriptors to statically-shaped memrefs with
/// element type f32.
void freeMemRefArguments(ArrayRef<void *> args);

} // namespace mlir

#endif // MLIR_EXECUTIONENGINE_MEMREFUTILS_H_
