/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_SPMD_SHARDONNAY_UTILS_H_
#define XLA_SERVICE_SPMD_SHARDONNAY_UTILS_H_

#include <cstdint>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace xla {
namespace sdy {

// Converts `attr` to string.
// TODO(bartchr): use the one from Shardonnay when its open sourced.
std::string attributeToString(mlir::Attribute attr);

// Converts `attr` to a `StringAttr` using the `builder`.
mlir::StringAttr getStringAttribute(mlir::Attribute attr,
                                    mlir::OpBuilder& builder);

// Gets the "frontend_attributes" `DictionaryAttr` from `op`. If it doesn't
// exist, return nullptr.
mlir::DictionaryAttr getFrontendAttrs(mlir::Operation* op);

// Gets the `frontend_attributes` `DictionaryAttr` from `funcOp`'s arg at
// `index`. If it doesn't exist, return nullptr.
mlir::DictionaryAttr getFuncArgFrontendAttrs(mlir::func::FuncOp funcOp,
                                             unsigned int index);

// Add `name` into the frontend attributes of `op` with value `value`. Note that
// `value` will be turned into a `StringAttr`.
void addFrontendAttribute(mlir::Operation* op, mlir::StringRef name,
                          mlir::Attribute value);

// Add `name` into the argument at `argNum`'s frontend attributes of `funcOp`
// with value `value`. Note that `value` will be turned into a `StringAttr`.
void addFrontendAttribute(mlir::func::FuncOp funcOp, mlir::StringRef name,
                          mlir::Attribute value, int64_t argNum);

// Remove `attributeName` from the frontend attributes of `op`.
void removeFrontendAttribute(mlir::Operation* op,
                             mlir::StringRef attributeName);

// Remove `attributeName` from the argument at `argNum`'s frontend attributes
// of `funcOp`.
void removeFrontendAttribute(mlir::func::FuncOp funcOp,
                             mlir::StringRef attributeName, int64_t argNum);

}  // namespace sdy
}  // namespace xla

#endif  // XLA_SERVICE_SPMD_SHARDONNAY_UTILS_H_
