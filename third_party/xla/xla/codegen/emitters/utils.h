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

#ifndef XLA_CODEGEN_EMITTERS_UTILS_H_
#define XLA_CODEGEN_EMITTERS_UTILS_H_

#include "absl/container/flat_hash_map.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/xla_data.pb.h"

namespace xla::emitters {

mlir::DenseElementsAttr GetZeroDenseElementsAttr(mlir::ShapedType shaped_type);

// Evaluates the epilogue of the fusion. Returns the results for each epilogue
// root.
absl::flat_hash_map<const HloInstruction*, mlir::ValueRange> EmitEpilogue(
    int epilogue_index, const emitters::PartitionedComputations& computations,
    mlir::func::FuncOp entry_fn,
    const absl::flat_hash_map<const HloInstruction*,
                              llvm::SmallVector<mlir::Value>>& injected,
    mlir::ValueRange output_indices, mlir::ImplicitLocOpBuilder& builder);

}  // namespace xla::emitters

#endif  // XLA_CODEGEN_EMITTERS_UTILS_H_
