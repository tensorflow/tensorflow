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

#include "xla/backends/cpu/codegen/emitters/cpu_fusion_emitter.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/backends/cpu/codegen/symbol_name_util.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/elemental_hlo_to_mlir.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/kernel_api_builder.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/util.h"

namespace xla {
namespace cpu {

absl::StatusOr<mlir::func::FuncOp> EmitEntryFunctionApi(
    mlir::ModuleOp fusion_module, const HloFusionInstruction& fusion,
    const std::string& entry_function_name,
    const BufferAssignment& buffer_assignment) {
  mlir::MLIRContext* context = fusion_module.getContext();

  if (fusion_module.getName().value().empty()) {
    return Internal("Fusion module name shouldn't be empty.");
  }

  emitters::KernelArguments::BufferAlignment alignment{MinAlign(), MinAlign(),
                                                       MinAlign()};
  ABSL_ASSIGN_OR_RETURN(
      mlir::func::FuncOp entry_func,
      emitters::EmitKernelApi(fusion_module, fusion, &buffer_assignment,
                              alignment, entry_function_name));
  // TODO(willfroom): Remove backend kind.
  SetBackendKind(context, entry_func, xla::BackendKind::kCpu);

  return entry_func;
}

absl::StatusOr<emitters::CallTargetProvider> EmitCallTargets(
    mlir::ModuleOp module, const HloFusionInstruction& fusion,
    const emitters::PartitionedComputations& computations,
    const std::vector<emitters::EpilogueSpecification>& epilogues) {
  auto subgraph_to_mlir_fn = computations.DeclareFunctions(module);

  // Erase subgraphs for all heroes that aren't used anywhere else. This is
  // necessary because the instructions may not have elemental implementations
  // (scatter).
  for (const auto& epilogue : epilogues) {
    for (auto* custom : epilogue.heroes) {
      if (custom->user_count() == 0) {
        subgraph_to_mlir_fn.extract(&computations.FindSubgraph(custom))
            .mapped()
            .erase();
      }
    }
  }

  // The epilogue functions replace the root tuple.
  auto* root = fusion.fused_instructions_computation()->root_instruction();
  if (root->opcode() == HloOpcode::kTuple && !epilogues.empty()) {
    subgraph_to_mlir_fn.extract(&computations.FindSubgraph(root))
        .mapped()
        .erase();
  }

  auto call_targets =
      computations.CreateCallTargetProvider(subgraph_to_mlir_fn);
  for (const auto& comp : computations.partitioned_computations()) {
    for (const auto& subgraph : comp.subgraphs()) {
      if (subgraph_to_mlir_fn.contains(&subgraph)) {
        ABSL_RETURN_IF_ERROR(emitters::SubgraphToMlirFunction(
            comp, subgraph, subgraph_to_mlir_fn[&subgraph], call_targets,
            computations.mlir_context()));
      }
    }
  }
  for (const auto& epilogue : computations.epilogues()) {
    if (epilogue.roots.empty()) {
      continue;
    }
    ABSL_RETURN_IF_ERROR(emitters::SubgraphToMlirFunction(
        computations.FindPartitionedComputation(
            fusion.fused_instructions_computation()),
        epilogue, subgraph_to_mlir_fn[&epilogue], call_targets,
        computations.mlir_context()));
  }

  return call_targets;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateNamedMlirModuleOp(
    const HloFusionInstruction& fusion, mlir::Builder& builder) {
  ABSL_ASSIGN_OR_RETURN(std::string fusion_name, GetFusionName(fusion));
  auto loc = mlir::NameLoc::get(builder.getStringAttr(fusion_name));
  return llvm_ir::CreateMlirModuleOp(loc, fusion_name);
}

absl::StatusOr<std::string> GetFusionName(const HloFusionInstruction& fusion) {
  std::string fusion_name(fusion.name());
  if (fusion.parent()
          ->parent()
          ->config()
          .debug_options()
          .xla_cpu_generate_unique_c_style_kernel_entry_points()) {
    ABSL_ASSIGN_OR_RETURN(fusion_name, ConvertToCName(absl::StrCat(
                                      fusion.parent()->parent()->name(), "_",
                                      fusion.name())));
  }
  return fusion_name;
}

}  // namespace cpu
}  // namespace xla
