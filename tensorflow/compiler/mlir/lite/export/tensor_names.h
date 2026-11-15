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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPORT_TENSOR_NAMES_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPORT_TENSOR_NAMES_H_

#include <string>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project

namespace mlir::TFL::fb_export {

// Returns a hierarchical, human readable name for the result of `op` (or
// nullptr for block arguments) at `loc`, or "" if `loc` has no name. It
//  - combines the caller and callee scopes of inlined call sites, dropping
//    JAX's internal `jit(_where)` wrappers,
//  - skips JAX op type wrappers (e.g. "pjit:", "composite:"),
//  - replaces a trailing "composite" scope with the composite name,
//  - joins fused locations with ';' after dropping CSE outliers from other
//    scopes.
// PyTorch locations (which use ';' as their own delimiter) are kept as is.
std::string GetHierarchicalName(Location loc, Operation* op);

// Returns true if the hierarchical name of `value` is more informative than
// the flat signature name `signature_name` an entry function gives to it.
bool PrefersHierarchicalName(Value value, llvm::StringRef signature_name);

// Names tensors, uniquely across the whole model.
class TensorNamer {
 public:
  // Names `value` as `name`, unless it is already named. Names set this way
  // are used verbatim, even if not unique.
  void SetName(Value value, llvm::StringRef name);

  // Returns the name of `value`, deriving a unique one from its location if it
  // isn't named yet.
  llvm::StringRef GetName(Value value);

  // Like `GetName`, but derives the name from that of `source`. The name of
  // `source` is reused if `source` isn't named yet.
  llvm::StringRef GetName(Value value, Value source);

 private:
  llvm::StringRef MakeUnique(llvm::StringRef name);

  llvm::DenseMap<Value, llvm::StringRef> names_;
  // Number of times each name was requested; owns the name strings.
  llvm::StringMap<int> name_counts_;
};

}  // namespace mlir::TFL::fb_export

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPORT_TENSOR_NAMES_H_
