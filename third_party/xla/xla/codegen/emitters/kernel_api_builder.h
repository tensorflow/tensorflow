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

#ifndef XLA_CODEGEN_EMITTERS_KERNEL_API_BUILDER_H_
#define XLA_CODEGEN_EMITTERS_KERNEL_API_BUILDER_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/runtime/work_dimensions.h"
#include "xla/runtime/work_group.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"

namespace xla::emitters {

// Emits and return the kernel entry function into the provided module it for
// the given HLO instruction.
// The function will be given the provided name and the arguments will be
// annotated with the buffer indices.
absl::StatusOr<mlir::func::FuncOp> EmitKernelApi(
    mlir::ModuleOp module, const HloInstruction& hlo_instruction,
    const BufferAssignment* buffer_assignment,
    const KernelArguments::BufferAlignment& buffer_alignment,
    absl::string_view entry_function_name);

void SetIndexDataLayout(mlir::ModuleOp module,
                        const HloInstruction& hlo_instruction,
                        bool force_64_bit = false);

// Get the default indexing map for the given work dimensions, unroll factor,
// and output shape.
IndexingMap GetDefaultWorkItemIndexingMap(const WorkDimensions& work_dimensions,
                                          int unroll_factor, const Shape& shape,
                                          mlir::MLIRContext* ctx);

// Emits the work group id ops annotated with the range of each dimension.
llvm::SmallVector<mlir::Value> EmitWorkGroupIds(
    mlir::ImplicitLocOpBuilder& builder, const NumWorkGroups& num_work_groups);

absl::StatusOr<CallTargetProvider> EmitPartitionedComputations(
    mlir::ModuleOp module, const PartitionedComputations& computations);

}  // namespace xla::emitters

#endif  // XLA_CODEGEN_EMITTERS_KERNEL_API_BUILDER_H_
