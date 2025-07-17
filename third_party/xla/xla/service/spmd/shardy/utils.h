/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_SPMD_SHARDY_UTILS_H_
#define XLA_SERVICE_SPMD_SHARDY_UTILS_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/log/check.h"
#include "absl/strings/escaping.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Support/LLVM.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace xla {
namespace sdy {

absl::string_view toStringView(mlir::StringRef sr);

// Gets the "frontend_attributes" `DictionaryAttr` from `op`. If it doesn't
// exist, return nullptr.
mlir::DictionaryAttr getFrontendAttrs(mlir::Operation* op);

// Gets the `frontend_attributes` `DictionaryAttr` from `funcOp`'s arg at
// `index`. If it doesn't exist, return nullptr.
mlir::DictionaryAttr getFuncArgFrontendAttrs(mlir::func::FuncOp funcOp,
                                             unsigned int index);

// Adds `name` into the frontend attributes of `op` with value `value`. If
// `name` already exists, it will be overwritten. Note that `value` will be
// turned into a `StringAttr`.
void setFrontendAttribute(mlir::Operation* op, mlir::StringRef name,
                          mlir::Attribute value);

// Adds `name` into the argument at `argNum`'s frontend attributes of `funcOp`
// with value `value`. If `name` already exists, it will be overwritten. Note
// that `value` will be turned into a `StringAttr`.
void setFrontendAttribute(mlir::func::FuncOp funcOp, mlir::StringRef name,
                          mlir::Attribute value, int64_t argNum);

// Remove `attributeName` from the frontend attributes of `op`.
void removeFrontendAttribute(mlir::Operation* op,
                             mlir::StringRef attributeName);

// Remove `attributeName` from the argument at `argNum`'s frontend attributes
// of `funcOp`.
void removeFrontendAttribute(mlir::func::FuncOp funcOp,
                             mlir::StringRef attributeName, int64_t argNum);

// Checks if "frontend_attributes" `DictionaryAttr` from `op` contains `key`.
bool hasFrontendAttr(mlir::Operation* op, mlir::StringRef key);

// Checks if `dictAttr` exists and contains `key`.
bool hasKey(mlir::DictionaryAttr dictAttr, mlir::StringRef key);

void loadAllRequiredDialects(mlir::MLIRContext* context);

// Adjusts the output sharding based on allowSpmdShardingPropagationToOutput
// flag.
void adjustOutputSharding(
    mlir::func::FuncOp func, int idx, mlir::sdy::TensorShardingAttr sharding,
    int64_t rank, absl::Span<const bool> allowSpmdShardingPropagationToOutput);

// Parses `escapedValue` to an attribute of type `AttrTy`.
template <typename AttrTy>
AttrTy parseStringAttr(llvm::StringRef escapedValue,
                       mlir::MLIRContext* context) {
  std::string unescapedValue;
  std::string error;
  CHECK(absl::CUnescape(
      absl::string_view(escapedValue.data(), escapedValue.size()),
      &unescapedValue, &error))
      << error;
  return mlir::cast<AttrTy>(mlir::parseAttribute(unescapedValue, context));
}

// Parses `attrName` from `dictAttr` to an attribute of type `AttrTy`.
template <typename AttrTy>
AttrTy parseStringAttr(mlir::DictionaryAttr dictAttr,
                       llvm::StringRef attrName) {
  if (mlir::Attribute stringAttr = dictAttr.get(attrName)) {
    return parseStringAttr<AttrTy>(
        mlir::cast<mlir::StringAttr>(stringAttr).getValue(),
        stringAttr.getContext());
  }
  return nullptr;
}

// Checks if `op`'s "frontend_attributes" `DictionaryAttr` contains `attrName`
// and parses it to an attribute of type `AttrTy`. If it doesn't exist, then
// returns std::nullopt.
template <typename AttrTy>
std::optional<AttrTy> tryGetFrontendAttr(mlir::Operation* op,
                                         mlir::StringRef attrName) {
  mlir::DictionaryAttr dictAttr = getFrontendAttrs(op);
  if (hasKey(dictAttr, attrName)) {
    return parseStringAttr<AttrTy>(dictAttr, attrName);
  }
  return std::nullopt;
}

// Builds a new `stablehlo.custom_call` with the same operands and attributes
// as `op` but with new `resultTypes`.
mlir::stablehlo::CustomCallOp cloneCustomCallWithNewResultTypes(
    mlir::stablehlo::CustomCallOp op, mlir::TypeRange resultTypes,
    mlir::IRRewriter& rewriter);

// Whether `op` is a Python callback custom call.
bool isPythonCallbackCustomCall(mlir::stablehlo::CustomCallOp op);

// Parses `shardingsFrontendAttr` as a `TensorShardingPerValueAttr`, duplicates
// the shardings at the specified indices, and returns the result as a string.
std::string duplicateShardingsAtIndices(
    mlir::StringRef shardingsFrontendAttr,
    const llvm::BitVector& indicesToDuplicate);

// Return all axes or sub-axes in the `mesh`, such that sub-axes are derived
// from `shardingOrAxisList` and sorted by their order in the mesh. For example,
// given mesh <"x"=2, "y"=16, "z"=4> and axis refs [{"x"}, {"y":2(2)}], we
// would return ["x", "y":1(2), "y":2(2), "y":4(4), "z"].
mlir::SmallVector<mlir::sdy::AxisRefAttr> getOrderedAxisRefs(
    mlir::Attribute shardingOrAxisList, mlir::sdy::MeshAttr mesh);

// Returns true if the module has at least one GSPMD attribute or op, like an
// `mhlo.sharding` attribute or `Sharding` custom call.
// TODO(b/420837831): delete this once we don't fall back to GSPMD.
bool hasGspmdAttrsOrOps(mlir::ModuleOp module);

// Check if the module has any sort of Shardy mesh:
// - `mesh`
// - `maximal_mesh_{X}`
// - `empty_mesh`
// TODO(b/420837831): delete this once we don't fall back to GSPMD.
bool hasShardyMesh(mlir::ModuleOp module);

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDY_UTILS_H_
