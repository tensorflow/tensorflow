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
#include "xla/service/gpu/fusions/mlir/computation_partitioner.h"

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/DataLayoutInterfaces.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/fusions/mlir/type_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/translate/hlo_to_mhlo/hlo_utils.h"
#include "xla/union_find.h"

namespace xla {
namespace gpu {
namespace mlir_converter {
namespace {

absl::flat_hash_map<const HloInstruction*, int> PartitionGraphByIndexing(
    const HloComputation& computation,
    std::function<bool(const HloInstruction*)> is_subgraph_root) {
  constexpr int kRootIndexing = 0;
  int next_indexing = 1;
  absl::flat_hash_map<const HloInstruction*, int> indexing;

  std::function<int(const HloInstruction*)> indexing_for_instr;
  indexing_for_instr = [&](const HloInstruction* instr) -> int {
    auto it = indexing.find(instr);
    if (it != indexing.end()) return it->second;

    if (is_subgraph_root(instr)) {
      return indexing[instr] = next_indexing++;
    }
    if (instr->opcode() != HloOpcode::kTuple &&
        !HloInstruction::IsOpElementwise(instr->opcode())) {
      return indexing[instr] = next_indexing++;
    }
    if (instr->user_count() == 0) {
      return indexing[instr] = kRootIndexing;
    }
    // If all users have the same indexing, we can reuse it.
    std::optional<int> instr_indexing = std::nullopt;
    for (auto* user : instr->users()) {
      auto user_indexing = indexing_for_instr(user);
      if (user->opcode() == HloOpcode::kConcatenate ||
          (instr_indexing && user_indexing != *instr_indexing)) {
        instr_indexing = std::nullopt;
        break;
      }
      instr_indexing = user_indexing;
    }
    return indexing[instr] = instr_indexing ? *instr_indexing : next_indexing++;
  };
  for (auto* instr : computation.instructions()) {
    indexing_for_instr(instr);
  }
  return indexing;
}

}  // namespace

PartitionedComputation::PartitionedComputation(
    const HloComputation* computation,
    std::function<bool(const HloInstruction*)> is_subgraph_root)
    : computation_(computation) {
  if (!is_subgraph_root) {
    is_subgraph_root = [](const HloInstruction*) { return false; };
  }

  // For each instruction, figure out what function it goes in. Parameters don't
  // count.
  absl::node_hash_map<const HloInstruction*,
                      tensorflow::UnionFind<const HloInstruction*>>
      disjoint_sets;
  auto indexing = PartitionGraphByIndexing(*computation, is_subgraph_root);
  for (auto* instruction : computation->instructions()) {
    if (instruction->opcode() == HloOpcode::kParameter) continue;
    disjoint_sets[instruction].Get() = instruction;
  }
  for (auto* instruction : computation->instructions()) {
    if (instruction->opcode() == HloOpcode::kParameter) continue;

    // If the instruction has to become a subgraph root, then we do not merge.
    bool can_merge = !is_subgraph_root(instruction);
    can_merge &=
        instruction->user_count() == 1 ||
        (instruction->user_count() > 1 &&
         absl::c_all_of(instruction->users(), [&](const HloInstruction* user) {
           return indexing.at(user) == indexing.at(instruction);
         }));
    auto is_bad_gather = [&](const HloInstruction* user) {
      // Don't merge into a gather that would evaluate the index more than once.
      return user->opcode() == HloOpcode::kGather &&
             user->operand_index(instruction) == 1 &&
             instruction->shape().dimensions(1) > 1;
    };
    auto is_concat = [&](const HloInstruction* user) {
      // Concat codegen doesn't work if any of a concat's transitive inputs is
      // reused. Instead of checking, we just cut the function at the concat,
      // which has the benefit of leading to slightly easier to read IR.
      return user->opcode() == HloOpcode::kConcatenate;
    };
    can_merge &= absl::c_none_of(instruction->users(), is_bad_gather);
    can_merge &= absl::c_none_of(instruction->users(), is_concat);
    if (can_merge) {
      auto& set = disjoint_sets[instruction];
      for (auto* user : instruction->users()) {
        set.Merge(&disjoint_sets[user]);
      }
    }
  }

  ConstHloInstructionMap<std::vector<const HloInstruction*>> functions;
  for (auto* instruction : computation->MakeInstructionPostOrder()) {
    if (instruction->opcode() == HloOpcode::kParameter) continue;
    functions[disjoint_sets[instruction].Get()].push_back(instruction);
  }

  subgraphs_.reserve(functions.size());
  for (auto& [cluster_id, instructions] : functions) {
    std::vector<const HloInstruction*> roots;
    for (auto* instruction : instructions) {
      if (instruction->user_count() == 0 ||
          absl::c_any_of(instruction->users(),
                         [cluster_id = cluster_id, &disjoint_sets](auto* user) {
                           return disjoint_sets[user].Get() != cluster_id;
                         })) {
        roots.push_back(instruction);
      }
    }
    CHECK(!roots.empty()) << "No roots found";
    std::string name = llvm_ir::SanitizeFunctionName(absl::StrCat(
        roots.front()->parent()->name(), "_",
        absl::StrJoin(roots, "_", [](std::string* out, const auto* root) {
          absl::StrAppend(out, root->name());
        })));
    subgraphs_.push_back(
        Subgraph{.name = std::move(name),
                 .instructions_post_order = std::move(instructions),
                 .roots = std::move(roots)});
  }

  for (const auto& subgraph : subgraphs_) {
    for (const auto* instruction : subgraph.instructions_post_order) {
      instructions_to_subgraphs_[instruction] = &subgraph;
    }
  }
}

PartitionedComputations::PartitionedComputations(const HloComputation* fusion) {
  // Collect all transitively called computations (including the fusion itself).
  absl::flat_hash_set<const HloComputation*> seen;
  std::vector<const HloComputation*> computations;
  std::function<void(const HloComputation*)> visit;
  visit = [&](const HloComputation* computation) {
    if (!seen.insert(computation).second) return;
    computations.push_back(computation);
    for (auto* instr : computation->instructions()) {
      absl::c_for_each(instr->called_computations(), visit);
    }
  };
  visit(fusion);

  partitioned_computations_.reserve(computations.size());
  for (auto* computation : computations) {
    computation_to_partitioning_[computation] =
        &partitioned_computations_.emplace_back(
            PartitionedComputation{computation});
  }
}

absl::flat_hash_map<const PartitionedComputation::Subgraph*, mlir::func::FuncOp>
PartitionedComputations::DeclareFunctions(mlir::ModuleOp module) const {
  absl::flat_hash_map<const PartitionedComputation::Subgraph*,
                      mlir::func::FuncOp>
      mapping;
  mlir::ImplicitLocOpBuilder builder(module.getLoc(), module->getContext());
  builder.setInsertionPointToEnd(module.getBody());
  for (const auto& computation : partitioned_computations_) {
    for (const auto& subgraph : computation.subgraphs()) {
      auto func_op = CreateSubgraphMlirFunction(subgraph, builder);
      func_op->setAttr("llvm.linkage", mlir::LLVM::LinkageAttr::get(
                                           module->getContext(),
                                           mlir::LLVM::Linkage::Internal));
      mapping[&subgraph] = func_op;
    }
  }
  return mapping;
}

mlir::func::FuncOp CreateSubgraphMlirFunction(
    const PartitionedComputation::Subgraph& subgraph,
    mlir::ImplicitLocOpBuilder& b) {
  auto* computation = subgraph.roots.front()->parent();
  llvm::SmallVector<mlir::Type> parameter_types;
  llvm::SmallVector<mlir::Type> result_types;

  auto element_type = [&](const auto& shape) {
    return *ConvertPrimitiveTypeToMLIRType(shape.element_type(), b);
  };

  const xla::Shape* one_root_shape = nullptr;
  for (auto* root : subgraph.roots) {
    if (root->shape().IsTuple()) {
      for (auto& shape : root->shape().tuple_shapes()) {
        one_root_shape = &shape;
        result_types.push_back(element_type(shape));
      }
    } else {
      one_root_shape = &root->shape();
      result_types.push_back(element_type(root->shape()));
    }
  }

  llvm::SmallVector<mlir::DictionaryAttr> arg_attrs;
  // We support the entry computation here for convenience of testing. The entry
  // computation is never code generated here.
  if (computation->IsFusionComputation() || computation->IsEntryComputation()) {
    for (auto* param : computation->parameter_instructions()) {
      parameter_types.push_back(TensorShapeToMlirType(param->shape(), b));
      arg_attrs.emplace_back();
    }
    for (int dim = 0; dim < one_root_shape->rank(); ++dim) {
      parameter_types.push_back(b.getIndexType());
      arg_attrs.emplace_back(mlir::DictionaryAttr::get(
          b.getContext(),
          {b.getNamedAttr(
              "xla.range",
              b.getIndexArrayAttr({0, one_root_shape->dimensions(dim) - 1}))}));
    }
  } else {
    for (auto* param : computation->parameter_instructions()) {
      parameter_types.push_back(element_type(param->shape()));
    }
  }
  auto ty = b.getFunctionType(parameter_types, result_types);
  return b.create<mlir::func::FuncOp>(
      subgraph.name, ty,
      /*attrs=*/llvm::ArrayRef<mlir::NamedAttribute>{}, arg_attrs);
}

}  // namespace mlir_converter
}  // namespace gpu
}  // namespace xla
