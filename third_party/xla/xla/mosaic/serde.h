/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_MOSAIC_SERDE_H_
#define XLA_MOSAIC_SERDE_H_

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

// A rule for upgrading an operation.
//
// The first argument is the operation to upgrade.
// The second argument is the source version.
// The third argument is a boolean that the upgrade rule will set to true if it
// happens to erase the operation.
//
// The function should return success if the upgrade was successful, or an error
// otherwise.
using UpgradeRuleType =
    std::function<::mlir::LogicalResult(::mlir::Operation*, int, bool&)>;

// A rule for downgrading an operation.
//
// The first argument is the operation to downgrade.
// The second argument is the target version.
//
// The function should return success if the downgrade was successful, or an
// error otherwise.
using DowngradeRuleType =
    std::function<::mlir::LogicalResult(::mlir::Operation*, int)>;

// Run serialization or deserialization on the given module.
::mlir::LogicalResult RunSerde(
    ::mlir::ModuleOp module,
    const llvm::StringMap<UpgradeRuleType>& upgrade_rules,
    const llvm::StringMap<DowngradeRuleType>& downgrade_rules, bool serialize,
    SerdeOptions options, bool keep_version_attr = false);

// Deprecated: use `UpgradeRuleType` or `DowngradeRuleType` instead.
// TODO(tlongeri): Remove once JAX no longer uses it.
using SerdeRuleType = UpgradeRuleType;

// Deprecated: pass `DowngradeRuleType` downgrade rules instead. The `bool&`
// argument of each downgrade rule is ignored.
// TODO(tlongeri): Remove once JAX no longer uses it.
::mlir::LogicalResult RunSerde(
    ::mlir::ModuleOp module,
    const llvm::StringMap<UpgradeRuleType>& upgrade_rules,
    const llvm::StringMap<SerdeRuleType>& downgrade_rules, bool serialize,
    SerdeOptions options, bool keep_version_attr = false);

}  // namespace jaxlib::mosaic

#endif  // XLA_MOSAIC_SERDE_H_
