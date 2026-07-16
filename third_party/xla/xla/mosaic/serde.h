/* Copyright 2025 The JAX Authors.

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

#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_SERDE_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_SERDE_H_

#include <functional>

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace jaxlib::mosaic {

struct SerdeOptions {
  llvm::StringRef dialect_prefix;  // mangled dialect prefix
  int highest_version;             // the highest supported version
  llvm::StringRef version_attr_name;
  int serialize_version;  // target version for serialization, must be -1 when
                          // deserializing
};

// A rule for upgrading or downgrading an operation.
//
// The first argument is the operation to upgrade/downgrade.
// The second argument is the target version.
//
// The function should return success if the upgrade/downgrade was successful,
// or an error otherwise.
using SerdeRuleType =
    std::function<::mlir::LogicalResult(::mlir::Operation *, int)>;

// Run serialization or deserialization on the given module.
::mlir::LogicalResult RunSerde(
    ::mlir::ModuleOp module,
    const llvm::StringMap<SerdeRuleType> &upgrade_rules,
    const llvm::StringMap<SerdeRuleType> &downgrade_rules, bool serialize,
    SerdeOptions options);

}  // namespace jaxlib::mosaic

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_SERDE_H_
