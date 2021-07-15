/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "llvm/ADT/StringRef.h"

#ifndef TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_UTILS_PLACEMENT_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_UTILS_PLACEMENT_UTIL_H_

namespace mlir {
namespace mhlo {
namespace placement_utils {

constexpr llvm::StringRef c_cpu = "cpu";
constexpr llvm::StringRef c_gpu = "gpu";

// Return true if the memref is on GPU
inline bool isGpuMemRef(Value memref) {
  assert(memref.getType().isa<MemRefType>());
  auto memory_space = memref.getType().cast<MemRefType>().getMemorySpace();
  return memory_space && memory_space.isa<StringAttr>() &&
         memory_space.cast<StringAttr>().getValue() == c_gpu;
}

}  // namespace placement_utils
}  // namespace mhlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_HLO_INCLUDE_MLIR_HLO_UTILS_PLACEMENT_UTIL_H_
