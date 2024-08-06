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

#ifndef XLA_TRANSLATE_HLO_TO_MHLO_ASYNC_IMPORTER_H_
#define XLA_TRANSLATE_HLO_TO_MHLO_ASYNC_IMPORTER_H_

#include <functional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace xla {

// Op Converters
absl::StatusOr<mlir::Operation*> ImportSend(
    const HloInstruction* instruction, mlir::Location loc,
    const llvm::SmallVectorImpl<mlir::Value>& operands,
    llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes,
    mlir::Type result_type, mlir::OpBuilder* builder,
    mlir::SymbolTable& symbol_table);

absl::StatusOr<mlir::Operation*> ImportRecv(
    const HloInstruction* instruction, mlir::Location loc,
    const llvm::SmallVectorImpl<mlir::Value>& operands,
    llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes,
    mlir::Type result_type, mlir::OpBuilder* builder,
    mlir::SymbolTable& symbol_table);

// Async Collectives
absl::StatusOr<mlir::Operation*> ImportAllGatherStart(
    const HloInstruction* instruction, mlir::Location loc,
    const llvm::SmallVectorImpl<mlir::Value>& operands,
    llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes,
    mlir::Type result_type, mlir::OpBuilder* builder,
    mlir::SymbolTable& symbol_table);

absl::StatusOr<mlir::Operation*> ImportAllReduceStart(
    const HloInstruction* instruction, mlir::Location loc,
    const llvm::SmallVectorImpl<mlir::Value>& operands,
    llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes,
    mlir::Type result_type, mlir::OpBuilder* builder,
    std::function<absl::Status(mlir::mhlo::AllReduceOp)> mutate_op,
    mlir::SymbolTable& symbol_table);

absl::StatusOr<mlir::Operation*> ImportCollectivePermuteStart(
    const HloInstruction* instruction, mlir::Location loc,
    const llvm::SmallVectorImpl<mlir::Value>& operands,
    llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes,
    mlir::Type result_type, mlir::OpBuilder* builder,
    mlir::SymbolTable& symbol_table);

absl::StatusOr<mlir::Operation*> ImportCopyStart(
    const HloInstruction* instruction, mlir::Location loc,
    const llvm::SmallVectorImpl<mlir::Value>& operands,
    llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes,
    mlir::Type result_type, mlir::OpBuilder* builder,
    mlir::SymbolTable& symbol_table);

absl::StatusOr<mlir::Operation*> ImportAsyncOpDone(
    const HloInstruction* instruction, mlir::Location loc,
    const llvm::SmallVectorImpl<mlir::Value>& operands,
    llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes,
    mlir::Type result_type, mlir::OpBuilder* builder);

}  // namespace xla

#endif  // XLA_TRANSLATE_HLO_TO_MHLO_ASYNC_IMPORTER_H_
