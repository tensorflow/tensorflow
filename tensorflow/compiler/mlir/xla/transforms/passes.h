/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_PASSES_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_PASSES_H_

#include <memory>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {

class FuncOp;
class ModuleOp;
class Operation;
template <typename T>
class OperationPass;
class Pass;

namespace mhlo {

/// Lowers from TF dialect to HLO dialect. When allow_partial_conversion is
/// false, emits an error if there is any operation that can't be legalized.
/// When `tf2xla_fallback_device_type` is not `None`, also uses legalization
/// patterns from TF2XLA fallback for provided device type (see
/// legalize_tf_with_tf2xla.cc for details). By default, TF2XLA fallback is not
/// used.
std::unique_ptr<OperationPass<FuncOp>> createLegalizeTFPass(
    bool allow_partial_conversion = false, bool legalize_chlo = true,
    llvm::Optional<StringRef> tf2xla_fallback_device_type = llvm::None);

/// Lowers from TF dialect to HLO dialect using tf2xla op kernels for the
/// specified device type.
std::unique_ptr<OperationPass<FuncOp>> createLegalizeTfWithTf2XlaPass(
    llvm::StringRef device_type);

/// Replaces types that do not exist in MHLO with equivalent types that do
/// exist.
std::unique_ptr<OperationPass<void>> CreateLegalizeTfTypesPass();

/// Adds the TF to XLA via TF2XLA rewrite patterns to the pattern list.
void PopulateLegalizeTfWithTf2XlaPatterns(llvm::StringRef device_type,
                                          OwningRewritePatternList& patterns);

/// Adds the TF to TF lowerings and TF to XLA rewrite patterns to the pattern
/// list.
void PopulateLegalizeTfPatterns(MLIRContext* context,
                                OwningRewritePatternList* patterns);

/// Checks whether the op is supported by the Tf2Xla fallback for legalization.
bool IsOpAllowedTf2XlaFallback(Operation* op);

/// Lowers from TF dialect's control flow to HLO dialect's control flow.
std::unique_ptr<OperationPass<ModuleOp>> createLegalizeTFControlFlowPass();

/// Converts the provided Operation as well as all nested operations into HLO
/// dialect using the conversion patterns registered by the HLO dialect. When
/// allow_partial_conversion is false, emits an error if there is any operation
/// that can't be legalized.
/// When `tf2xla_fallback_device_type` is not `None`, also uses legalization
/// patterns from TF2XLA fallback for provided device type (see
/// legalize_tf_with_tf2xla.cc for details). By default, TF2XLA fallback is not
/// used.
LogicalResult legalizeTF(
    Operation* op, bool allow_partial_conversion = false,
    bool legalize_chlo = true,
    llvm::Optional<StringRef> tf2xla_fallback_device_type = llvm::None);

// Legalizes TF/XLA communication ops (TF dialect) to HLO dialect communication
// ops.
std::unique_ptr<OperationPass<ModuleOp>> CreateLegalizeTFCommunicationPass();

// Prepare module for export to XLA HLO protos/instruction.
std::unique_ptr<OperationPass<FuncOp>> CreatePrepareForExport();

}  // namespace mhlo
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_TRANSFORMS_PASSES_H_
