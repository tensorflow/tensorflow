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
#include "xla/codegen/emitters/computation_partitioner.h"

#include <cstdint>
#include <functional>
#include <iterator>
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
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/emitters/type_util.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace emitters {
namespace {

int Arity(const Shape& shape) {
  return shape.IsTuple() ? shape.tuple_shapes_size() : 1;
}

const Shape& TupleShape(const Shape& shape, int index) {
  return shape.IsTuple() ? shape.tuple_shapes(index) : shape;
}

std::vector<IndexingMapSet> ComputeOperandIndexingMaps(
    const HloInstruction* instr, mlir::MLIRContext* mlir_context) {
  std::vector<IndexingMapSet> indexing_maps_per_operand;
  // For some ops, there is no indexing map implemented for the operands (e.g.
  // scatter) or there are multiple results and the common iteration space is
  // not known. In these cases, we return identity indexing for the operands.
  if (HloPredicateIsOp<HloOpcode::kTuple, HloOpcode::kScatter,
                       HloOpcode::kParameter>(instr)) {
    int64_t num_operands = instr->operand_count();
    indexing_maps_per_operand.reserve(num_operands);
    for (int64_t i = 0; i < num_operands; ++i) {
      indexing_maps_per_operand.push_back(
          {CreateIdentityMap(instr->operand(i)->shape(), mlir_context)});
    }
  } else {
    auto operands_indexing =
        ComputeOutputToInputIndexing(instr, /*output_id=*/0, mlir_context);
    operands_indexing.Simplify();
    indexing_maps_per_operand = std::move(operands_indexing.indexing_maps);
  }
  return indexing_maps_per_operand;
}

}  // namespace

EpilogueSpecification EpilogueSpecification::FromIdentityIndexing(
    const HloInstruction* hero, const HloInstruction* root,
    mlir::MLIRContext* mlir_context) {
  EpilogueSpecification result;
  absl::c_copy(root->shape().dimensions(),
               std::back_inserter(result.index_ranges));
  result.roots.push_back(root);
  result.root_indexing.push_back(
      CreateIdentityMap(root->shape(), mlir_context));
  result.heroes.push_back(hero);
  return result;
}

std::string PartitionedComputation::Subgraph::ToString(int indentation) const {
  std::string indent(indentation, ' ');
  std::ostringstream ss;
  ss << indent << "SUBGRAPH " << name << " {\n";
  for (auto* instr :
       (*instructions.begin())->parent()->MakeInstructionPostOrder()) {
    if (!instructions.contains(instr)) continue;
    ss << indent << "  ";
    if (absl::c_linear_search(roots, instr)) {
      ss << "ROOT ";
    }
    ss << instr->ToString() << "\n";
  }
  ss << indent << "}";
  return ss.str();
}

std::string PartitionedComputation::ToString(int indentation) const {
  std::ostringstream ss;
  ss << "PartitionedComputation " << computation_->name() << ":";
  for (const Subgraph& subgraph : subgraphs_) {
    ss << "\n" << subgraph.ToString(indentation);
  }
  return ss.str();
}

template <typename C, typename F>
bool AllIdentical(const C& c, F&& f) {
  auto begin = std::begin(c);
  auto end = std::end(c);
  if (begin == end || begin + 1 == end) {
    return true;
  }
  auto v = f(*begin);
  ++begin;
  for (; begin != end; ++begin) {
    if (f(*begin) != v) {
      return false;
    }
  }
  return true;
}

// Whether the instruction is evaluated more than once by any of its direct
// users. By 'more than once', we mean with a different indexing - for example,
// reduce will evaluate its argument more than once (in a loop), but with only
// one indexing.
bool IsEvaluatedMoreThanOnce(const HloInstruction* instr) {
  return absl::c_any_of(instr->users(), [&](const HloInstruction* user) {
    if (user->opcode() == HloOpcode::kGather &&
        absl::c_linear_search(user->OperandIndices(instr), 1) &&
        instr->shape().dimensions_size() >= 2 &&
        instr->shape().dimensions(1) > 1) {
      return true;
    }
    if (user->opcode() == HloOpcode::kConcatenate &&
        user->OperandIndices(instr).size() > 1) {
      return true;
    }
    return false;
  });
}

using SubgraphId = int;

constexpr int kMaxHloOpsPerSubgraph = 2000;

// HloSubgraphData is associated with a single HLO instruction and contains
// the necessary information to partition the computation into subgraphs.
struct HloSubgraphData {
  // Indexing map from the root of the function to the input of the instruction.
  IndexingMapSet indexings;
  // Subgraph IDs of the users of the instruction.
  absl::flat_hash_set<SubgraphId> user_subgraph_ids;
  // ID of the subgraph that this instruction belongs to.
  SubgraphId subgraph_id = -1;
  // Whether the instruction is a root of the subgraph.
  bool is_root = false;
  // Number of users.
  int num_users = 0;
};

PartitionedComputation::PartitionedComputation(
    const HloComputation* computation, mlir::MLIRContext* mlir_context,
    std::function<bool(const HloInstruction*)> is_subgraph_root)
    : computation_(computation) {
  CHECK_NE(computation, nullptr);

  auto pre_order = computation->MakeInstructionPostOrder();
  absl::c_reverse(pre_order);
  absl::flat_hash_map<const HloInstruction*, int> instr_to_id;
  for (auto [i, instr] : llvm::enumerate(pre_order)) {
    instr_to_id[instr] = i;
  }

  SubgraphId subgraph_count = 0;
  std::vector<HloSubgraphData> id_to_subgraph_data(pre_order.size());
  std::vector<int> num_ops_per_subgraph;
  // Iterate over the use-def chains and check if the instruction should be
  // placed in a separate function.
  for (auto [instr_index, instr] : llvm::enumerate(pre_order)) {
    // If the instruction is a subgraph root, or if it has users with different
    // function IDs users, or if it has multiple non-trivial indexings, we need
    // to create a new function.
    auto& instr_subgraph_data = id_to_subgraph_data[instr_index];
    instr_subgraph_data.is_root =
        is_subgraph_root(instr) ||
        instr_subgraph_data.user_subgraph_ids.size() != 1 ||
        instr_subgraph_data.indexings.size() > 1;
    bool is_large_subgraph =
        instr_subgraph_data.subgraph_id > -1 &&
        num_ops_per_subgraph[instr_subgraph_data.subgraph_id] >=
            kMaxHloOpsPerSubgraph;
    if (instr_subgraph_data.is_root || is_large_subgraph) {
      instr_subgraph_data.subgraph_id = subgraph_count++;
      instr_subgraph_data.indexings.clear();
      num_ops_per_subgraph.push_back(1);
    } else {
      instr_subgraph_data.subgraph_id =
          *instr_subgraph_data.user_subgraph_ids.begin();
      ++num_ops_per_subgraph.at(instr_subgraph_data.subgraph_id);
    }
    if (num_ops_per_subgraph.at(instr_subgraph_data.subgraph_id) >
            kMaxHloOpsPerSubgraph &&
        instr_subgraph_data.num_users == 1) {
      instr_subgraph_data.subgraph_id = subgraph_count++;
      instr_subgraph_data.is_root = true;
      instr_subgraph_data.indexings.clear();
      num_ops_per_subgraph.push_back(1);
    }
    auto operands_indexing = ComputeOperandIndexingMaps(instr, mlir_context);
    // Iterate over the operands and add the func_ids of the current instruction
    // to their HloSubgraphIndexing and compute the indexing maps.
    for (auto [operand_instr, operand_maps] :
         llvm::zip(instr->operands(), operands_indexing)) {
      auto& operand_subgraph_data =
          id_to_subgraph_data[instr_to_id[operand_instr]];
      ++operand_subgraph_data.num_users;
      IndexingMap instr_indexing = instr_subgraph_data.indexings.empty()
                                       ? IndexingMap::GetUndefined()
                                       : *instr_subgraph_data.indexings.begin();
      IndexingMap composed_indexing =
          instr_subgraph_data.is_root
              ? *operand_maps.begin()
              : ComposeIndexingMaps(instr_indexing, *operand_maps.begin());
      composed_indexing.Simplify();

      operand_subgraph_data.user_subgraph_ids.insert(
          instr_subgraph_data.subgraph_id);
      if (!composed_indexing.IsUndefined() &&
          !composed_indexing.IsKnownEmpty()) {
        operand_subgraph_data.indexings.insert(std::move(composed_indexing));
      }
    }
  }

  std::vector<std::vector<const HloInstruction*>> functions(subgraph_count);
  for (const auto& [subgraph_data, instr] :
       llvm::reverse(llvm::zip(id_to_subgraph_data, pre_order))) {
    CHECK_NE(subgraph_data.subgraph_id, -1) << "Uninitialized subgraph ID";
    functions[subgraph_data.subgraph_id].push_back(instr);
  }

  subgraphs_.reserve(functions.size());
  for (auto&& [function_id, instructions] : llvm::enumerate(functions)) {
    std::vector<const HloInstruction*> roots;
    std::vector<IndexingMap> root_indexing;
    const xla::Shape* first_root_shape = nullptr;
    bool has_no_compute = true;
    for (auto* instruction : instructions) {
      has_no_compute &=
          HloPredicateIsOp<HloOpcode::kParameter, HloOpcode::kIota,
                           HloOpcode::kConstant>(instruction);
      if (id_to_subgraph_data[instr_to_id[instruction]].is_root) {
        roots.push_back(instruction);
        if (first_root_shape) {
          CHECK(!instruction->shape().IsTuple())
              << "Internal tuples are not supported";
          if (ShapeUtil::EqualIgnoringElementType(*first_root_shape,
                                                  instruction->shape())) {
            root_indexing.push_back(root_indexing.front());
          } else {
            // Bitcast from the first root to the target shape.
            root_indexing.push_back(GetBitcastMap(
                *first_root_shape, instruction->shape(), mlir_context));
          }
        } else {
          first_root_shape = &instruction->shape();
          while (first_root_shape->IsTuple()) {
            first_root_shape = &first_root_shape->tuple_shapes()[0];
          }
          root_indexing.push_back(
              CreateIdentityMap(*first_root_shape, mlir_context));
        }
      }
    }

    CHECK(first_root_shape);
    std::vector<int64_t> ranges{first_root_shape->dimensions().begin(),
                                first_root_shape->dimensions().end()};

    CHECK(!roots.empty()) << "No roots found";
    std::string name = llvm_ir::SanitizeFunctionName(absl::StrCat(
        roots.front()->parent()->name(), "_",
        absl::StrJoin(roots, "_", [](std::string* out, const auto* root) {
          absl::StrAppend(out, root->name());
        })));
    subgraphs_.push_back(Subgraph{
        /* .name = */ std::move(name),
        /* .instructions = */ {instructions.begin(), instructions.end()},
        /* .roots = */ std::move(roots),
        /* .index_ranges = */ std::move(ranges),
        /* .root_indexing = */ std::move(root_indexing),
        /* .has_no_compute = */ has_no_compute});
  }

  for (const auto& subgraph : subgraphs_) {
    for (const auto* instruction : subgraph.instructions) {
      instructions_to_subgraphs_[instruction] = &subgraph;
    }
  }
}

PartitionedComputation::Subgraph PartitionedComputation::Subgraph::ForEpilogue(
    const EpilogueSpecification& epilogue) {
  if (epilogue.roots.empty()) return {};
  const auto* computation = epilogue.heroes.front()->parent();
  PartitionedComputation::Subgraph subgraph;
  subgraph.name = llvm_ir::SanitizeFunctionName(
      absl::StrCat(computation->name(), "__epilogue__",
                   absl::StrJoin(epilogue.roots, "_",
                                 [](std::string* out, const auto* root) {
                                   absl::StrAppend(out, root->name());
                                 })));
  subgraph.roots = epilogue.roots;

  int index = 0;
  for (auto* hero : epilogue.heroes) {
    if (subgraph.injected_value_starts.insert({hero, index}).second) {
      index += Arity(hero->shape());
    }
  }
  subgraph.num_injected_values = index;

  absl::flat_hash_set<const HloInstruction*> seen;
  std::function<void(const HloInstruction*)> visit;
  visit = [&](const HloInstruction* instruction) {
    if (subgraph.injected_value_starts.contains(instruction)) return;
    if (!seen.insert(instruction).second) return;
    for (auto [index, operand] : llvm::enumerate(instruction->operands())) {
      visit(operand);
    }
  };

  visit(computation->root_instruction());
  subgraph.instructions = std::move(seen);
  subgraph.index_ranges = epilogue.index_ranges;
  subgraph.root_indexing = epilogue.root_indexing;
  return subgraph;
}

PartitionedComputations::PartitionedComputations(
    const HloComputation* fusion, mlir::MLIRContext* mlir_context,
    std::vector<EpilogueSpecification> epilogues)
    : fusion_(fusion) {
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
  epilogues_.reserve(epilogues.size());
  for (const auto& epilogue : epilogues) {
    epilogues_.push_back(
        PartitionedComputation::Subgraph::ForEpilogue(epilogue));
    roots.insert(epilogue.heroes.begin(), epilogue.heroes.end());
    for (auto* instruction : epilogue.heroes) {
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
  auto create_funcs =
      [&](absl::Span<const PartitionedComputation::Subgraph> subgraphs) {
        for (const auto& subgraph : subgraphs) {
          if (subgraph.roots.empty()) continue;
          auto func_op = CreateSubgraphMlirFunction(subgraph, builder);
          func_op->setAttr("llvm.linkage", mlir::LLVM::LinkageAttr::get(
                                               module->getContext(),
                                               mlir::LLVM::Linkage::Internal));
          mapping[&subgraph] = func_op;
        }
      };
  for (const auto& computation : partitioned_computations_) {
    create_funcs(computation.subgraphs());
  }
  create_funcs(epilogues_);
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
    return PrimitiveTypeToMlirType(shape.element_type(), b);
  };

  for (auto* root : subgraph.roots) {
    for (auto ty : ShapeToMlirTypes(root->shape(), b)) {
      result_types.push_back(
          mlir::cast<mlir::RankedTensorType>(ty).getElementType());
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
    parameter_types.resize(operand_offset + subgraph.num_injected_values);
    arg_attrs.resize(parameter_types.size());

    for (auto [value, start] : subgraph.injected_value_starts) {
      for (int index = 0; index < Arity(value->shape()); ++index) {
        parameter_types[operand_offset + start + index] =
            element_type(TupleShape(value->shape(), index));
      }
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
  if (subgraph.has_no_compute) {
    func_op->setAttr(kHasNoCompute, b.getBoolAttr(true));
  }
  // Needed so that the function can potentially be inlined in-place.
  func_op.setPrivate();
  return func_op;
}

}  // namespace emitters
}  // namespace xla
