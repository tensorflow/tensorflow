/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/codegen/tiling/experimental/tiled_hlo.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/codegen/tiling/experimental/tile.h"
#include "xla/codegen/tiling/experimental/tile_propagation.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/name_uniquer.h"
#include "xla/util.h"

namespace xla::gpu::experimental {

using ::llvm::ArrayRef;
using ::llvm::SmallVector;

TiledHloRegion::TiledHloRegion(
    std::vector<TiledHloInstruction* absl_nonnull> instructions,
    llvm::SmallVector<const TiledHloInstruction* absl_nonnull, 4> roots)
    : instructions_(std::move(instructions)), roots_(std::move(roots)) {
  for (const TiledHloInstruction* root : roots_) {
    CHECK(absl::c_linear_search(instructions_, root))
        << "Root instruction " << root->ToString()
        << " must be present in the region.";
  }
}

std::string TiledHloInstruction::ToString(
    absl::string_view field_separator) const {
  std::stringstream ss;
  ss << "hlo: " << hlo_->ToString() << field_separator;
  if (tiles_.size() == 1) {
    ss << "tile: " << tile().ToString();
  } else {
    for (const auto& [i, t] : llvm::enumerate(tiles_)) {
      if (i > 0) {
        ss << field_separator;
      }
      ss << "tile #" << i << ": " << t.ToString();
    }
  }
  for (const auto& [index, region] : llvm::enumerate(regions_)) {
    ss << field_separator << "region #" << index << " {";
    for (const TiledHloInstruction* instruction : region.instructions()) {
      ss << field_separator << instruction->ToString(field_separator);
    }
    ss << field_separator << "}";
  }
  return ss.str();
}

llvm::SmallVector<const TiledHloInstruction*, 2>
TiledHloInstruction::runtime_variables() const {
  llvm::SmallVector<const TiledHloInstruction*, 2> runtime_variables;
  if (auto dyn_slice = DynCast<HloDynamicSliceInstruction>(hlo_)) {
    // `operands_` might be empty and inconsistent with `hlo_->operand_count()`
    // if the instruction lies outside the fusion boundary (we skip populating
    // its operands during traversal).
    for (int i = dyn_slice->first_index_operand_number(); i < operands_.size();
         ++i) {
      runtime_variables.push_back(operands_[i]);
    }
  }
  return runtime_variables;
}
namespace {

// A hash set of TiledHloInstructions.
//
// This set adds a few key features on top of
// absl::flat_hash_set<TiledHloInstruction*>:
// * Elements are inserted as (hlo, tiles) pairs. The instruction is
//   constructed in `instructions` (shared by all regions of a computation) and
//   dropped again if an equivalent element is already in the set.
// * Elements are compared by (hlo, tiles), not by pointer.
// * Elements are stored in the order of insertion.
class OrderedTiledHloPtrSet {
 public:
  explicit OrderedTiledHloPtrSet(std::deque<TiledHloInstruction>& instructions)
      : instructions_(instructions) {}

  // Inserts an element into the set.
  // Returns a pair of a non-owning raw pointer to the element that was inserted
  // (or the element that prevented insertion) and a bool indicating whether the
  // element was inserted.
  std::pair<TiledHloInstruction*, bool> Insert(const HloInstruction* hlo,
                                               Tile tile) {
    return Insert(hlo,
                  llvm::SmallVector<experimental::Tile, 2>{std::move(tile)});
  }

  // Same as above for an instruction with one tile per result.
  std::pair<TiledHloInstruction*, bool> Insert(
      const HloInstruction* hlo,
      llvm::SmallVector<experimental::Tile, 2> tiles) {
    TiledHloInstruction& candidate =
        instructions_.emplace_back(hlo, std::move(tiles));
    auto [it, inserted] = hash_set_.insert(&candidate);
    if (!inserted) {
      instructions_.pop_back();
      return {*it, false};
    }
    insertion_order_.push_back(&candidate);
    return {&candidate, true};
  }

  // Consumes the set, returning pointers to its elements in insertion order.
  std::vector<TiledHloInstruction*> ConsumeInInsertionOrder() && {
    return std::move(insertion_order_);
  }

 private:
  struct PtrHash {
    size_t operator()(const TiledHloInstruction* v) const {
      return absl::HashOf(v->hlo(), v->tiles());
    }
  };

  struct PtrEqual {
    bool operator()(const TiledHloInstruction* lhs,
                    const TiledHloInstruction* rhs) const {
      return lhs == rhs ||
             (lhs->hlo() == rhs->hlo() && lhs->tiles() == rhs->tiles());
    }
  };

  // Stores non-owning pointers to the elements in the set. Elements are
  // compared by the value behind the pointer, not the pointer itself.
  absl::flat_hash_set<TiledHloInstruction*, PtrHash, PtrEqual> hash_set_;

  // Owned by the TiledHloComputation and shared with the sets of other regions.
  std::deque<TiledHloInstruction>& instructions_;

  // Pointers to the elements of this set, in insertion order.
  std::vector<TiledHloInstruction*> insertion_order_;
};

// Sorts tiled hlo instructions in def-before-use order, starting from
// `roots_with_no_users`.
//
// Precondition: all `tiled_hlo_instructions` are reachable from
// `roots_with_no_users`.
void SortTiledHloInstructionsInPostOrder(
    std::vector<TiledHloInstruction*>& tiled_hlo_instructions,
    ArrayRef<const TiledHloInstruction*> roots_with_no_users) {
  absl::flat_hash_map<const TiledHloInstruction*, int64_t> topological_order;

  std::function<void(const TiledHloInstruction*)> visit_instruction;
  visit_instruction = [&](const TiledHloInstruction* instruction) {
    if (topological_order.contains(instruction)) {
      return;
    }
    for (const TiledHloInstruction* rt_operand :
         instruction->runtime_variables()) {
      visit_instruction(rt_operand);
    }
    for (const TiledHloInstruction* operand : instruction->operands()) {
      visit_instruction(operand);
    }
    topological_order[instruction] = topological_order.size();
  };

  for (const TiledHloInstruction* root_with_no_user : roots_with_no_users) {
    visit_instruction(root_with_no_user);
  }
  absl::c_sort(tiled_hlo_instructions, [&](const TiledHloInstruction* t1,
                                           const TiledHloInstruction* t2) {
    auto it1 = topological_order.find(t1);
    auto it2 = topological_order.find(t2);
    CHECK(it1 != topological_order.end())
        << "Unexpected stray instruction: " << t1->ToString();
    CHECK(it2 != topological_order.end())
        << "Unexpected stray instruction: " << t2->ToString();
    return it1->second < it2->second;
  });

  VLOG(4) << "Sorted symbolic tiled HLO instructions in def-before-use order:\n"
          << absl::StrJoin(
                 tiled_hlo_instructions, "\n",
                 [](std::string* out, const TiledHloInstruction* instruction) {
                   absl::StrAppend(out, instruction->ToString("; "));
                 });
}

bool IsReductionLoopRequired(const TiledHloInstruction& tiled_hlo) {
  const TilingSpace& tiling_space = tiled_hlo.tile().tiling_space();
  return absl::c_any_of(
      tiling_space.dimensions(), [&](const TilingSpace::DimensionInfo& dim) {
        return dim.type == TilingSpace::DimensionSemantics::kSequential &&
               dim.hlo == tiled_hlo.hlo() && dim.tile_size.has_value() &&
               *dim.tile_size < dim.dimension_size;
      });
}

bool IsScanLoopRequired(const TiledHloInstruction& tiled_hlo) {
  const auto* scan = Cast<HloScanInstruction>(tiled_hlo.hlo());
  const TilingSpace& tiling_space = tiled_hlo.tile().tiling_space();
  const auto& dim_info =
      tiling_space.GetDimensionInfo(*scan, scan->scan_dimension());
  return dim_info.type == TilingSpace::DimensionSemantics::kSequential &&
         dim_info.tile_size.has_value() &&
         *dim_info.tile_size < dim_info.dimension_size;
}

// Returns the variable IDs of the sequential dimensions of `tiled_hlo` whose
// tile does not cover the whole dimension, i.e. those iterated by a loop in the
// emitted code.
llvm::SmallVector<VariableID, 2> GetLoopDimIds(
    const TiledHloInstruction& tiled_hlo) {
  const TilingSpace& tiling_space = tiled_hlo.tile().tiling_space();
  llvm::SmallVector<VariableID, 2> loop_dim_ids;
  for (const TilingSpace::DimensionInfo& dim : tiling_space.dimensions()) {
    if (dim.type == TilingSpace::DimensionSemantics::kSequential &&
        dim.hlo == tiled_hlo.hlo() &&
        (!dim.tile_size.has_value() || *dim.tile_size < dim.dimension_size)) {
      loop_dim_ids.push_back(dim.id.value());
    }
  }
  return loop_dim_ids;
}

// Defines how the operands of a TiledHloInstruction are partitioned during
// region reconstruction.
//
// Examples:
// - For a  reduction, e.g., `reduce(input, init)`, the `input` is wrapped in a
//   nested region if and only if the reduction dimension tile sizes are
//   symbolic or smaller than the dimension size. `init` is never wrapped in a
//   nested region:
//
//   RegionSchema {
//     region_roots = {{0}}
//     operand_ids = {1}
//     loop_dim_ids = {<reduction dimension IDs>}
//   }
// - For a dot product (e.g., `dot(lhs, rhs)`), both operands are grouped
//   together as region roots to represent the sub-computation:
//
//   RegionSchema {
//     region_roots = {{0, 1}}
//     operand_ids = {}
//     loop_dim_ids = {<contracting dimension ID>}
//   }
struct RegionSchema {
  using OperandIDs = llvm::SmallVector<int64_t>;

  // Groups of operand indices. Each group represents the roots of a new
  // nested HLO region (e.g., loop bodies, dot product sub-computations).
  std::vector<OperandIDs> region_roots;

  // Operand indices that are regular inputs to the instruction and should
  // remain within the current region.
  OperandIDs operand_ids;

  // Sequential dimensions iterated by the loop wrapping each region in
  // `region_roots`. Empty for regions that are not loop bodies.
  llvm::SmallVector<VariableID, 2> loop_dim_ids;
};

// Determines the partitioning specification for the operands of a tiled HLO
// instruction.
//
// This specification dictates the boundaries of the tiled HLO regions. Operands
// that represent roots of nested computations (such as reduction bodies, dot
// product inputs, or concatenate inputs) are categorized under `region_roots`
// to initiate the creation of nested `TiledHloRegion`s. Regular operands that
// are simply inputs to the current computation level are kept under
// `operand_ids` to be processed in the current region.
RegionSchema GetRegionSchema(const TiledHloInstruction& tiled_hlo) {
  const HloOpcode opcode = tiled_hlo.hlo()->opcode();
  const int64_t num_operands = tiled_hlo.hlo()->operand_count();

  auto iota = [](int64_t start, int64_t end) {
    return llvm::to_vector(llvm::seq<int64_t>(start, end));
  };
  switch (opcode) {
    case HloOpcode::kDot:
    case HloOpcode::kScaledDot: {
      return RegionSchema{/*region_roots=*/{iota(0, num_operands)},
                          /*operand_ids=*/{},
                          /*loop_dim_ids=*/GetLoopDimIds(tiled_hlo)};
    }
    case HloOpcode::kRaggedDot: {
      // No hoisting: the ragged dot emitter generates its own control flow.
      return RegionSchema{/*region_roots=*/{iota(0, num_operands)},
                          /*operand_ids=*/{},
                          /*loop_dim_ids=*/{}};
    }
    case HloOpcode::kReduce: {
      if (IsReductionLoopRequired(tiled_hlo)) {
        int64_t num_inputs = num_operands / 2;
        return RegionSchema{/*region_roots=*/{iota(0, num_inputs)},
                            /*operand_ids=*/{iota(num_inputs, num_operands)},
                            /*loop_dim_ids=*/GetLoopDimIds(tiled_hlo)};
      }
      break;
    }
    case HloOpcode::kScan: {
      if (IsScanLoopRequired(tiled_hlo)) {
        const auto* scan = Cast<HloScanInstruction>(tiled_hlo.hlo());
        int64_t num_inputs = scan->inputs().size();
        return RegionSchema{/*region_roots=*/{iota(0, num_inputs)},
                            /*operand_ids=*/{iota(num_inputs, num_operands)},
                            /*loop_dim_ids=*/GetLoopDimIds(tiled_hlo)};
      }
      break;
    }
    case HloOpcode::kConcatenate: {
      RegionSchema schema;
      schema.region_roots.reserve(num_operands);
      for (int64_t operand_id = 0; operand_id < num_operands; ++operand_id) {
        schema.region_roots.push_back({operand_id});
      }
      return schema;
    }
    default:
      break;
  }
  return RegionSchema{/*region_roots=*/{},
                      /*operand_ids=*/iota(0, num_operands),
                      /*loop_dim_ids=*/{}};
}

// Recursively populates `tile_names` with unique names for `tiled_hlo` and
// all instructions within its regions.
void PrepopulateTileNames(
    const TiledHloInstruction* tiled_hlo, NameUniquer& name_uniquer,
    absl::flat_hash_map<const TiledHloInstruction*, std::string>& tile_names) {
  auto [_, inserted] = tile_names.try_emplace(
      tiled_hlo, name_uniquer.GetUniqueName(
                     absl::StrCat(tiled_hlo->hlo()->name(), ".tile_0")));
  if (!inserted) {
    return;
  }
  for (const auto& region : tiled_hlo->hlo_regions()) {
    for (const TiledHloInstruction* region_instruction :
         region.instructions()) {
      PrepopulateTileNames(region_instruction, name_uniquer, tile_names);
    }
  }
}

std::string TiledHloOperandsToString(
    const TiledHloInstruction* tiled_hlo,
    const absl::flat_hash_map<const TiledHloInstruction*, std::string>&
        tile_names) {
  const HloInstruction* hlo = tiled_hlo->hlo();
  if (auto parameter = DynCast<HloParameterInstruction>(hlo)) {
    return std::to_string(parameter->parameter_number());
  }
  absl::InlinedVector<std::string, 4> operand_names;
  for (const auto& operand : tiled_hlo->operands()) {
    CHECK(tile_names.contains(operand)) << operand->hlo()->name();
    operand_names.push_back(tile_names.at(operand));
  }
  return absl::StrJoin(operand_names, ", ");
}

// Recursively prints `tiled_hlo` and all instructions within its regions.
void PrintTiledHloInstruction(
    const TiledHloInstruction* tiled_hlo,
    const absl::flat_hash_map<const TiledHloInstruction*, std::string>&
        tile_names,
    std::stringstream& ss, int indent) {
  std::string indentation(indent, ' ');
  ss << indentation << tile_names.at(tiled_hlo) << " = "
     << HloOpcodeString(tiled_hlo->hlo()->opcode()) << "("
     << TiledHloOperandsToString(tiled_hlo, tile_names) << ")";
  if (tiled_hlo->tiles().size() == 1) {
    ss << " " << tiled_hlo->tile().ToString(false) << "\n";
  } else {
    ss << "\n";
    for (const auto& [i, tile] : llvm::enumerate(tiled_hlo->tiles())) {
      ss << indentation << "  #" << i << " " << tile.ToString(false) << "\n";
    }
  }

  for (auto const& [i, region] : llvm::enumerate(tiled_hlo->hlo_regions())) {
    ss << indentation << "region #" << i << " {\n";
    for (const TiledHloInstruction* instruction : region.instructions()) {
      PrintTiledHloInstruction(instruction, tile_names, ss, indent + 2);
    }
    ss << indentation << "}\n";
  }
}

// Extracts `HloInstruction`s from a span of `HloInstructionAdaptor`s.
absl::InlinedVector<const HloInstruction*, 2> ToInstructions(
    absl::Span<const HloInstructionAdaptor> instruction_adaptors) {
  absl::InlinedVector<const HloInstruction*, 2> hlo_instructions;
  hlo_instructions.reserve(instruction_adaptors.size());
  absl::c_transform(
      instruction_adaptors, std::back_inserter(hlo_instructions),
      [&](const HloInstructionAdaptor& instr) { return &instr.instruction(); });
  return hlo_instructions;
}

// State of a region under construction. Linked to the enclosing region's
// context so that loop-invariant instructions can be inserted into an outer
// region while a nested one is being built.
struct RegionContext {
  // Sequential dimensions iterated by the loop wrapping this region, if any.
  llvm::SmallVector<VariableID, 2> loop_dim_ids;
  OrderedTiledHloPtrSet* instructions_set = nullptr;
  std::vector<TiledHloInstruction*>* worklist = nullptr;
  RegionContext* parent = nullptr;
};

// Returns the outermost region into which an instruction with the given `tile`
// can be hoisted from `current`.
RegionContext* FindTargetRegion(RegionContext* current, const Tile& tile) {
  // Regions that are not loop bodies (e.g. concatenate branches) are barriers.
  while (current->parent != nullptr && !current->loop_dim_ids.empty() &&
         !tile.DependsOnVariables(current->loop_dim_ids)) {
    current = current->parent;
  }
  return current;
}

// Recursively constructs a tiled HLO region starting from a set of root
// instructions.
//
// Performs a backward topological traversal (from roots to parameters) within
// the fusion boundary to reconstruct the tiled HLO dependency graph and any
// nested computation regions (e.g., reduction bodies or dot computations).
//
// Instructions are not ordered.
absl::StatusOr<TiledHloRegion> CreateHloRegion(
    llvm::SmallVector<std::pair<const HloInstruction*, experimental::Tile>, 4>
        roots,
    const HloFusionAdaptor& fusion, TilingSpace& tiling_space,
    std::deque<TiledHloInstruction>& instruction_storage,
    absl::flat_hash_map<int64_t,
                        std::pair<const TiledHloInstruction*, Interval>>&
        rt_symbol_to_tiled_hlo,
    llvm::SmallVector<VariableID, 2> loop_dim_ids,
    RegionContext* parent_context) {
  std::vector<TiledHloInstruction*> worklist;
  OrderedTiledHloPtrSet tiled_hlo_instructions_set(instruction_storage);
  RegionContext current_context{std::move(loop_dim_ids),
                                &tiled_hlo_instructions_set, &worklist,
                                parent_context};

  llvm::SmallVector<const TiledHloInstruction*, 4> canonical_roots;
  canonical_roots.reserve(roots.size());
  for (auto& [root_hlo, root_tile] : roots) {
    auto [raw_root, inserted] =
        tiled_hlo_instructions_set.Insert(root_hlo, std::move(root_tile));
    canonical_roots.push_back(raw_root);
    if (inserted) {
      worklist.push_back(raw_root);
    }
  }

  while (!worklist.empty()) {
    TiledHloInstruction* tiled_hlo = worklist.back();
    worklist.pop_back();
    const HloInstruction* hlo = tiled_hlo->hlo();
    if (!fusion.ContainsInstruction(hlo) || hlo->operand_count() == 0) {
      continue;
    }

    ABSL_ASSIGN_OR_RETURN(
        auto operands_tiles,
        PropagateTileToInput(tiling_space, *hlo, tiled_hlo->tile(), 0));

    RegionSchema spec = GetRegionSchema(*tiled_hlo);

    HloInstructionAdaptor instruction_adaptor(*hlo, &fusion);
    absl::InlinedVector<HloInstructionAdaptor, 2> operands =
        instruction_adaptor.GetOperands();

    llvm::SmallVector<const TiledHloInstruction*, 4> resolved_operands(
        hlo->operand_count(), nullptr);

    for (const auto& region_root_ids : spec.region_roots) {
      llvm::SmallVector<std::pair<const HloInstruction*, experimental::Tile>, 4>
          region_roots;
      region_roots.reserve(region_root_ids.size());
      for (int64_t id : region_root_ids) {
        region_roots.emplace_back(&operands[id].instruction(),
                                  std::move(operands_tiles[id]));
      }

      ABSL_ASSIGN_OR_RETURN(
          TiledHloRegion res,
          CreateHloRegion(std::move(region_roots), fusion, tiling_space,
                          instruction_storage, rt_symbol_to_tiled_hlo,
                          spec.loop_dim_ids, &current_context));
      for (const auto& [i, id] : llvm::enumerate(region_root_ids)) {
        resolved_operands[id] = res.roots()[i];
      }

      tiled_hlo->AddHloRegion(std::move(res));
    }

    for (int64_t id : spec.operand_ids) {
      // Hoists and deduplicates loop-invariant instructions.
      RegionContext* target_region =
          FindTargetRegion(&current_context, operands_tiles[id]);
      auto [operand_tiled_hlo, inserted] =
          target_region->instructions_set->Insert(
              &operands[id].instruction(), std::move(operands_tiles[id]));
      resolved_operands[id] = operand_tiled_hlo;
      if (inserted) {
        target_region->worklist->push_back(operand_tiled_hlo);
      }

      std::optional<const TilingSpace::RTVarInfo*> rt_var_info =
          tiling_space.GetRTVarInfo(*hlo, id);
      if (rt_var_info.has_value()) {
        rt_symbol_to_tiled_hlo.insert(std::make_pair(
            rt_var_info.value()->id + tiling_space.num_dimensions(),
            std::make_pair(operand_tiled_hlo, rt_var_info.value()->bounds)));
      }
    }

    for (int64_t i = 0; i < hlo->operand_count(); ++i) {
      CHECK(resolved_operands[i] != nullptr);
      tiled_hlo->AddOperand(resolved_operands[i]);
    }
  }

  return TiledHloRegion{
      std::move(tiled_hlo_instructions_set).ConsumeInInsertionOrder(),
      std::move(canonical_roots)};
}

}  // namespace

void TiledHloRegion::Simplify() {
  for (TiledHloInstruction* instruction : instructions_) {
    for (Tile& tile : instruction->tiles()) {
      tile.Simplify();
    }
    for (auto& region : instruction->hlo_regions()) {
      region.Simplify();
    }
  }
}

void TiledHloRegion::SortInstructionsPostOrder() {
  for (TiledHloInstruction* instruction : instructions_) {
    for (auto& region : instruction->hlo_regions()) {
      region.SortInstructionsPostOrder();
    }
  }
  SortTiledHloInstructionsInPostOrder(instructions_, roots_);
}

/*static*/ absl::StatusOr<TiledHloComputation> TiledHloComputation::Tile(
    const HloFusionAdaptor& fusion, std::unique_ptr<TilingSpace> tiling_space) {
  llvm::SmallVector<std::pair<const HloInstruction*, experimental::Tile>, 4>
      tiled_roots;
  tiled_roots.reserve(fusion.GetRoots().size());
  for (const auto& [root, tile] :
       llvm::zip(fusion.GetRoots(), tiling_space->tiled_roots())) {
    tiled_roots.emplace_back(&root.instruction(), tile);
  }

  std::deque<TiledHloInstruction> instruction_storage;
  absl::flat_hash_map<int64_t, std::pair<const TiledHloInstruction*, Interval>>
      rt_symbol_to_tiled_hlo;
  ABSL_ASSIGN_OR_RETURN(
      TiledHloRegion region,
      CreateHloRegion(std::move(tiled_roots), fusion, *tiling_space,
                      instruction_storage, rt_symbol_to_tiled_hlo,
                      /*loop_dim_ids=*/{}, /*parent_context=*/nullptr));

  return TiledHloComputation(std::move(tiling_space),
                             std::move(instruction_storage), std::move(region),
                             std::move(rt_symbol_to_tiled_hlo));
}

void TiledHloComputation::Simplify() { region_.Simplify(); }

void TiledHloComputation::SortInstructionsPostOrder() {
  region_.SortInstructionsPostOrder();
}

std::string TiledHloComputation::ToString() const {
  std::stringstream ss;

  ss << tiling_space_->ToString() << "\n";

  NameUniquer name_uniquer("_");
  absl::flat_hash_map<const TiledHloInstruction*, std::string> tile_names;
  for (const TiledHloInstruction* tiled_hlo : region_.instructions()) {
    PrepopulateTileNames(tiled_hlo, name_uniquer, tile_names);
  }

  ss << "Tiled HLO:\n";
  for (const TiledHloInstruction* tiled_hlo : region_.instructions()) {
    PrintTiledHloInstruction(tiled_hlo, tile_names, ss, /*indent=*/2);
  }
  return ss.str();
}

}  // namespace xla::gpu::experimental
