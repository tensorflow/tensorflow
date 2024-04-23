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

#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/DataLayoutInterfaces.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/fusions/mlir/type_util.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/translate/hlo_to_mhlo/hlo_utils.h"
#include "xla/union_find.h"

namespace xla {
namespace gpu {
namespace mlir_converter {
namespace {

absl::flat_hash_map<const HloInstruction*, int> PartitionGraphByIndexing(
    const HloComputation& computation) {
  constexpr int kRootIndexing = 0;
  int next_indexing = 1;
  absl::flat_hash_map<const HloInstruction*, int> indexing;

  std::function<int(const HloInstruction*)> indexing_for_instr;
  indexing_for_instr = [&](const HloInstruction* instr) -> int {
    auto it = indexing.find(instr);
    if (it != indexing.end()) return it->second;

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

EpilogueSpecification EpilogueSpecification::FromIdentityIndexing(
    const HloInstruction* hero, const HloInstruction* root,
    mlir::MLIRContext* mlir_context) {
  EpilogueSpecification result;
  absl::c_copy(root->shape().dimensions(),
               std::back_inserter(result.index_ranges));
  result.root_indexing.push_back(mlir::AffineMap::getMultiDimIdentityMap(
      root->shape().rank(), mlir_context));
  result.heroes.push_back(hero);
  return result;
}

EpilogueSpecification EpilogueSpecification::FromOutputIndexing(
    const HloFusionAnalysis& analysis,
    const std::vector<const HloInstruction*>& heroes,
    const KernelFusionInterface& fusion, mlir::MLIRContext* mlir_context) {
  EpilogueSpecification result;

  for (auto [index, hero] : llvm::enumerate(analysis.fusion_heroes())) {
    auto indexing = fusion.ComputeThreadIdToOutputIndexing(index, mlir_context);
    if (index == 0) {
      result.index_ranges.reserve(indexing->GetDimensionCount() +
                                  indexing->GetSymbolCount());
      for (const auto& dim : indexing->GetDimensionBounds()) {
        result.index_ranges.push_back(dim.upper + 1);
      }
      for (const auto& sym : indexing->GetSymbolBounds()) {
        result.index_ranges.push_back(sym.upper + 1);
      }
    }

    auto epilogue_indexing =
        ComputeEpilogueInputToOutputIndexing(hero, mlir_context);
    auto root_indexing = ComposeIndexingMaps(*indexing, epilogue_indexing);

    result.root_indexing.push_back(root_indexing.GetAffineMap());
  }
  result.heroes = heroes;
  return result;
}

std::string PartitionedComputation::Subgraph::ToString() const {
  std::ostringstream ss;
  ss << "SUBGRAPH " << name << " {\n";
  for (auto* instr :
       (*instructions.begin())->parent()->MakeInstructionPostOrder()) {
    if (!instructions.contains(instr)) continue;
    ss << "  ";
    if (absl::c_linear_search(roots, instr)) {
      ss << "ROOT ";
    }
    ss << instr->ToString() << "\n";
  }
  ss << "}";
  return ss.str();
}

std::string PartitionedComputation::ToString() const {
  std::ostringstream ss;
  ss << "PartitionedComputation " << computation_->name() << ":";
  for (const Subgraph& subgraph : subgraphs_) {
    ss << "\n" << subgraph.ToString();
  }
  return ss.str();
}

std::string PartitionedComputations::ToString() const {
  std::ostringstream ss;
  ss << "PartitionedComputations:";
  for (const auto& partitioned_computation : partitioned_computations_) {
    ss << "\n" << partitioned_computation.ToString();
  }
  return ss.str();
}

PartitionedComputation::PartitionedComputation(
    const HloComputation* computation, mlir::MLIRContext* mlir_context,
    std::function<bool(const HloInstruction*)> is_subgraph_root)
    : computation_(computation) {
  CHECK_NE(computation, nullptr);
  // For each instruction, figure out what function it goes in. Parameters don't
  // count.
  absl::node_hash_map<const HloInstruction*,
                      tensorflow::UnionFind<const HloInstruction*>>
      disjoint_sets;
  auto indexing = PartitionGraphByIndexing(*computation);
  for (auto* instruction : computation->instructions()) {
    disjoint_sets[instruction].Get() = instruction;
  }
  for (auto* instruction : computation->instructions()) {
    // If the instruction has to become a subgraph root, then we do not merge.
    bool can_merge = !is_subgraph_root(instruction);
    if (instruction->user_count() > 0) {
      // If all users have the same indexing, we can merge.
      int64_t one_user_indexing = indexing.at(instruction->users().front());
      can_merge &=
          absl::c_all_of(instruction->users(), [&](const HloInstruction* user) {
            return indexing.at(user) == one_user_indexing;
          });
    }
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
    functions[disjoint_sets[instruction].Get()].push_back(instruction);
  }

  subgraphs_.reserve(functions.size());
  for (auto& [cluster_id, instructions] : functions) {
    auto is_different_cluster = [cluster_id = cluster_id,
                                 &disjoint_sets](auto* user) {
      auto it = disjoint_sets.find(user);
      if (it == disjoint_sets.end()) {
        return true;
      }
      return it->second.Get() != cluster_id;
    };

    std::vector<const HloInstruction*> roots;
    std::vector<mlir::AffineMap> root_indexing;
    const xla::Shape* first_root_shape = nullptr;
    for (auto* instruction : instructions) {
      if (instruction->user_count() == 0 ||
          absl::c_any_of(instruction->users(), is_different_cluster)) {
        roots.push_back(instruction);
        if (first_root_shape) {
          CHECK(!instruction->shape().IsTuple())
              << "Internal tuples are not supported";
          if (ShapeUtil::EqualIgnoringElementType(*first_root_shape,
                                                  instruction->shape())) {
            root_indexing.push_back(root_indexing.front());
          } else {
            // Bitcast from the first root to the target shape.
            auto bitcast = GetBitcastMap(*first_root_shape,
                                         instruction->shape(), mlir_context);
            root_indexing.push_back(bitcast.GetAffineMap());
          }
        } else {
          first_root_shape = &instruction->shape();
          if (first_root_shape->IsTuple()) {
            first_root_shape = &first_root_shape->tuple_shapes()[0];
          }
          root_indexing.push_back(mlir::AffineMap::getMultiDimIdentityMap(
              first_root_shape->rank(), mlir_context));
        }
      }
    }

    std::vector<int64_t> ranges{first_root_shape->dimensions().begin(),
                                first_root_shape->dimensions().end()};

    CHECK(!roots.empty()) << "No roots found";
    std::string name = llvm_ir::SanitizeFunctionName(absl::StrCat(
        roots.front()->parent()->name(), "_",
        absl::StrJoin(roots, "_", [](std::string* out, const auto* root) {
          absl::StrAppend(out, root->name());
        })));
    subgraphs_.push_back(
        Subgraph{.name = std::move(name),
                 .instructions = {instructions.begin(), instructions.end()},
                 .roots = std::move(roots),
                 .index_ranges = std::move(ranges),
                 .root_indexing = std::move(root_indexing)});
  }

  for (const auto& subgraph : subgraphs_) {
    for (const auto* instruction : subgraph.instructions) {
      instructions_to_subgraphs_[instruction] = &subgraph;
    }
  }
}

std::optional<PartitionedComputation::Subgraph>
PartitionedComputation::Subgraph::ForEpilogue(
    const std::optional<EpilogueSpecification>& epilogue) {
  if (!epilogue) {
    return std::nullopt;
  }
  const auto* computation = epilogue->heroes.front()->parent();
  if ((epilogue->heroes.size() == 1 &&
       epilogue->heroes[0] == computation->root_instruction())) {
    return std::nullopt;
  }

  PartitionedComputation::Subgraph subgraph;
  subgraph.name = llvm_ir::SanitizeFunctionName(
      absl::StrCat(computation->name(), "__epilogue__"));
  if (computation->root_instruction()->opcode() == HloOpcode::kTuple) {
    absl::c_copy(computation->root_instruction()->operands(),
                 std::back_inserter(subgraph.roots));
  } else {
    subgraph.roots = {computation->root_instruction()};
  }

  for (auto* hero : epilogue->heroes) {
    if (!subgraph.injected_values.contains(hero)) {
      int index = subgraph.injected_values.size();
      subgraph.injected_values[hero] = index;
    }
  }

  absl::flat_hash_set<const HloInstruction*> seen;
  std::function<void(const HloInstruction*)> visit;
  visit = [&](const HloInstruction* instruction) {
    if (!seen.insert(instruction).second) return;
    for (auto [index, operand] : llvm::enumerate(instruction->operands())) {
      if (!subgraph.injected_values.contains(operand)) {
        visit(operand);
      }
    }
  };

  visit(computation->root_instruction());
  subgraph.instructions = std::move(seen);
  subgraph.index_ranges = epilogue->index_ranges;
  subgraph.root_indexing = epilogue->root_indexing;
  return subgraph;
}

PartitionedComputations::PartitionedComputations(
    const HloComputation* fusion, mlir::MLIRContext* mlir_context,
    std::optional<EpilogueSpecification> epilogue)
    : fusion_(fusion),
      epilogue_(PartitionedComputation::Subgraph::ForEpilogue(epilogue)) {
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

  absl::flat_hash_set<const HloInstruction*> roots;
  if (epilogue) {
    roots = {epilogue->heroes.begin(), epilogue->heroes.end()};
    for (auto* instruction : epilogue->heroes) {
      roots.insert(instruction->operands().begin(),
                   instruction->operands().end());
    }
  }
  auto is_root = [&](const HloInstruction* instruction) {
    return roots.contains(instruction);
  };

  partitioned_computations_.reserve(computations.size());
  for (auto* computation : computations) {
    computation_to_partitioning_[computation] =
        &partitioned_computations_.emplace_back(
            PartitionedComputation{computation, mlir_context, is_root});
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
  if (epilogue_) {
    auto func_op = CreateSubgraphMlirFunction(*epilogue_, builder);
    func_op->setAttr("llvm.linkage",
                     mlir::LLVM::LinkageAttr::get(
                         module->getContext(), mlir::LLVM::Linkage::Internal));
    mapping[&*epilogue_] = func_op;
  }
  return mapping;
}

const PartitionedComputation::Subgraph& PartitionedComputations::FindSubgraph(
    const HloInstruction* instr) const {
  return FindPartitionedComputation(instr->parent()).FindSubgraph(instr);
}

CallTargetProvider PartitionedComputations::CreateCallTargetProvider(
    const absl::flat_hash_map<const PartitionedComputation::Subgraph*,
                              mlir::func::FuncOp>& subgraph_to_func) const {
  return [&, this](const HloInstruction* instr) {
    const auto& subgraph = FindSubgraph(instr);
    CHECK(subgraph_to_func.contains(&subgraph))
        << "No function found for subgraph with instruction "
        << instr->ToString();
    return subgraph_to_func.at(&subgraph);
  };
}

mlir::func::FuncOp CreateSubgraphMlirFunction(
    const PartitionedComputation::Subgraph& subgraph,
    mlir::ImplicitLocOpBuilder& b) {
  auto* computation = subgraph.roots.front()->parent();
  llvm::SmallVector<mlir::Type> parameter_types;
  llvm::SmallVector<mlir::Type> result_types;

  auto element_type = [&](const auto& shape) {
    return *ConvertPrimitiveTypeToMlirType(shape.element_type(), b);
  };

  for (auto* root : subgraph.roots) {
    if (root->shape().IsTuple()) {
      for (auto& shape : root->shape().tuple_shapes()) {
        result_types.push_back(element_type(shape));
      }
    } else {
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
    for (int64_t size : subgraph.index_ranges) {
      parameter_types.push_back(b.getIndexType());
      arg_attrs.emplace_back(mlir::DictionaryAttr::get(
          b.getContext(),
          {b.getNamedAttr("xla.range", b.getIndexArrayAttr({0, size - 1}))}));
    }

    // Populate arguments for injected parameters (values that are computed
    // outside the function and are passed into it).
    int operand_offset = parameter_types.size();
    parameter_types.resize(operand_offset + subgraph.injected_values.size());
    arg_attrs.resize(parameter_types.size());

    for (auto [value, index] : subgraph.injected_values) {
      parameter_types[operand_offset + index] = element_type(value->shape());
    }
  } else {
    for (auto* param : computation->parameter_instructions()) {
      parameter_types.push_back(element_type(param->shape()));
    }
  }
  auto ty = b.getFunctionType(parameter_types, result_types);
  auto func_op = b.create<mlir::func::FuncOp>(
      subgraph.name, ty,
      /*attrs=*/llvm::ArrayRef<mlir::NamedAttribute>{}, arg_attrs);
  // Needed so that the function can potentially be inlined in-place.
  func_op.setPrivate();
  return func_op;
}

}  // namespace mlir_converter
}  // namespace gpu
}  // namespace xla
