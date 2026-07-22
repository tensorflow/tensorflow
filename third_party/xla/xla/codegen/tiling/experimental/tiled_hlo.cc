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
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/codegen/tiling/experimental/tile.h"
#include "xla/codegen/tiling/experimental/tile_propagation.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/name_uniquer.h"
#include "xla/util.h"

namespace xla::gpu::experimental {

using ::llvm::ArrayRef;
using ::llvm::SmallVector;

TiledHloRegion::TiledHloRegion(
    std::vector<absl_nonnull std::unique_ptr<TiledHloInstruction>> instructions,
    llvm::SmallVector<const TiledHloInstruction* absl_nonnull, 4> roots)
    : instructions_(std::move(instructions)), roots_(std::move(roots)) {
  for (const TiledHloInstruction* root : roots_) {
    CHECK(absl::c_any_of(
        instructions_,
        [root](const auto& instruction) { return instruction.get() == root; }))
        << "Root instruction " << root->ToString()
        << " must be present in the region.";
  }
}

std::string TiledHloInstruction::ToString(
    absl::string_view field_separator) const {
  std::stringstream ss;
  ss << "hlo: " << hlo_->ToString() << field_separator;
  ss << "tile: " << tile().ToString();
  for (const auto& [index, region] : llvm::enumerate(regions_)) {
    ss << field_separator << "region #" << index << " {";
    for (const auto& instruction : region.instructions()) {
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

// A hash set of unique pointers to TiledHloInstructions.
//
// This set add a few key features on top of
// absl::flat_hash_set<TiledHloInstruction*>:
// * The set takes ownership of the object and deletes the object if an
//   equivalent element is already in the set.
// * Values are compared by the value behind the pointer, not the pointer
//   itself.
// * This set provides a convenient method to extract the unique pointers into a
//   vector.
// * Values are stored in the order of insertion. This is useful when we have
//   information about the order in which we process elements.
class OrderedTiledHloPtrSet {
 public:
  // Inserts an element into the set.
  // Returns a pair of a non-owning raw pointer to the element that was inserted
  // (or the element that prevented insertion) and a bool indicating whether the
  // element was inserted.
  std::pair<TiledHloInstruction*, bool> Insert(
      std::unique_ptr<TiledHloInstruction> elem) {
    auto [it, inserted] = hash_set_.insert(elem.get());
    if (inserted) {
      data_.push_back(std::move(elem));
    }
    return {*it, inserted};
  }

  void Reserve(int64_t n) {
    hash_set_.reserve(n);
    data_.reserve(n);
  }

  // Moves data out of the set.
  std::vector<std::unique_ptr<TiledHloInstruction>> ExtractData() {
    return std::move(data_);
  }

 private:
  struct PtrHash {
    size_t operator()(const TiledHloInstruction* v) const {
      return absl::HashOf(v->hlo(), v->tile());
    }
  };

  struct PtrEqual {
    bool operator()(const TiledHloInstruction* lhs,
                    const TiledHloInstruction* rhs) const {
      return lhs == rhs ||
             (lhs->hlo() == rhs->hlo() && lhs->tile() == rhs->tile());
    }
  };

  // Stores non-owning pointers to the elements in the set. Elements are
  // compared by the value behind the pointer, not the pointer itself.
  absl::flat_hash_set<TiledHloInstruction*, PtrHash, PtrEqual> hash_set_;

  // Stores owning pointers to the elements in the set.
  std::vector<std::unique_ptr<TiledHloInstruction>> data_;
};

// Sorts tiled hlo instructions in def-before-use order, starting from
// `roots_with_no_users`. If instruction is not reachable from the root then it
// might be put in an arbitrary position.
void SortTiledHloInstructionsInPostOrder(
    std::vector<std::unique_ptr<TiledHloInstruction>>& tiled_hlo_instructions,
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
  absl::c_sort(
      tiled_hlo_instructions,
      [&](const std::unique_ptr<TiledHloInstruction>& t1,
          const std::unique_ptr<TiledHloInstruction>& t2) {
        auto it1 = topological_order.find(t1.get());
        auto it2 = topological_order.find(t2.get());
        int64_t order1 = (it1 != topological_order.end()) ? it1->second : -1;
        int64_t order2 = (it2 != topological_order.end()) ? it2->second : -1;
        return order1 < order2;
      });

  VLOG(4) << "Sorted symbolic tiled HLO instructions in def-before-use order:\n"
          << absl::StrJoin(
                 tiled_hlo_instructions, "\n",
                 [](std::string* out,
                    const std::unique_ptr<TiledHloInstruction>& instruction) {
                   absl::StrAppend(out, instruction->ToString("; "));
                 });
}
// Defines how the operands of a TiledHloInstruction are partitioned during
// region reconstruction.
struct OperandsSpec {
  using OperandIDs = llvm::SmallVector<int64_t>;
  // Groups of operand indices. Each group represents the roots of a new
  // nested HLO region (e.g., loop bodies, dot product sub-computations).
  std::vector<OperandIDs> region_roots;
  // Operand indices that are regular inputs to the instruction and should
  // remain within the current region.
  OperandIDs operand_ids;
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
OperandsSpec GetSpec(const TiledHloInstruction& tiled_hlo,
                     const TilingSpace& tiling_space) {
  // Helper to generate a contiguous sequence of operand indices.
  auto iota = [](int64_t size, int64_t start = 0) {
    return llvm::to_vector(llvm::iota_range<int64_t>(start, start + size,
                                                     /*Inclusive=*/false));
  };
  const HloOpcode opcode = tiled_hlo.hlo()->opcode();
  const int64_t num_operands = tiled_hlo.hlo()->operand_count();
  OperandsSpec spec;
  if (opcode == HloOpcode::kDot || opcode == HloOpcode::kScaledDot) {
    spec.region_roots.push_back(iota(num_operands));
  } else if (opcode == HloOpcode::kConcatenate) {
    spec.region_roots.reserve(num_operands);
    for (int64_t i = 0; i < num_operands; ++i) {
      spec.region_roots.push_back({i});
    }
  } else {
    spec.operand_ids = iota(num_operands);
  }
  return spec;
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
    for (const auto& region_instruction : region.instructions()) {
      PrepopulateTileNames(region_instruction.get(), name_uniquer, tile_names);
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
     << TiledHloOperandsToString(tiled_hlo, tile_names) << ") "
     << tiled_hlo->tile().ToString(false) << "\n";

  for (auto const& [i, region] : llvm::enumerate(tiled_hlo->hlo_regions())) {
    ss << indentation << "region #" << i << " {\n";
    for (const auto& instruction : region.instructions()) {
      PrintTiledHloInstruction(instruction.get(), tile_names, ss, indent + 2);
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

}  // namespace

void TiledHloRegion::Simplify() {
  for (auto& instruction : instructions_) {
    Tile tile = instruction->tile();
    tile.Simplify();
    instruction->set_tile(std::move(tile));
    for (auto& region : instruction->hlo_regions()) {
      region.Simplify();
    }
  }
}

void TiledHloRegion::SortInstructionsPostOrder() {
  for (auto& instruction : instructions_) {
    for (auto& region : instruction->hlo_regions()) {
      region.SortInstructionsPostOrder();
    }
  }
  SortTiledHloInstructionsInPostOrder(instructions_, roots_);
}

// Recursively constructs a tiled HLO region starting from a set of root
// instructions.
//
// Performs a backward topological traversal (from roots to parameters) within
// the fusion boundary to reconstruct the tiled HLO dependency graph and any
// nested computation regions (e.g., reduction bodies or dot computations).
//
// Instructions are not ordered.
absl::StatusOr<TiledHloRegion> TiledHloComputation::CreateHloRegion(
    std::vector<std::unique_ptr<TiledHloInstruction>> roots,
    const HloFusionAdaptor& fusion, TilingSpace& tiling_space,
    absl::flat_hash_map<int64_t,
                        std::pair<const TiledHloInstruction*, Interval>>&
        rt_symbol_to_tiled_hlo) {
  std::vector<TiledHloInstruction*> worklist;
  OrderedTiledHloPtrSet tiled_hlo_instructions_set;

  llvm::SmallVector<const TiledHloInstruction*, 4> canonical_roots;
  canonical_roots.reserve(roots.size());
  for (auto& root : roots) {
    auto [raw_root, inserted] =
        tiled_hlo_instructions_set.Insert(std::move(root));
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

    ASSIGN_OR_RETURN(
        auto operands_tiles,
        PropagateTileToInput(tiling_space, *hlo, tiled_hlo->tile(), 0));

    OperandsSpec spec = GetSpec(*tiled_hlo, tiling_space);

    HloInstructionAdaptor instruction_adaptor(*hlo, &fusion);
    std::vector<std::unique_ptr<TiledHloInstruction>> tiled_operands;
    tiled_operands.reserve(hlo->operand_count());
    for (const auto& [i, operand] :
         llvm::enumerate(instruction_adaptor.GetOperands())) {
      tiled_operands.push_back(std::make_unique<TiledHloInstruction>(
          &operand.instruction(), operands_tiles[i]));
    }

    std::vector<const TiledHloInstruction*> resolved_operands(
        hlo->operand_count(), nullptr);

    for (const auto& region_root_ids : spec.region_roots) {
      std::vector<std::unique_ptr<TiledHloInstruction>> region_roots;
      region_roots.reserve(region_root_ids.size());
      for (int64_t id : region_root_ids) {
        region_roots.push_back(std::move(tiled_operands[id]));
      }

      ASSIGN_OR_RETURN(TiledHloRegion res,
                       CreateHloRegion(std::move(region_roots), fusion,
                                       tiling_space, rt_symbol_to_tiled_hlo));
      for (const auto& [i, id] : llvm::enumerate(region_root_ids)) {
        resolved_operands[id] = res.roots()[i];
      }

      tiled_hlo->AddHloRegion(std::move(res));
    }

    for (int64_t id : spec.operand_ids) {
      auto [operand_tiled_hlo, inserted] =
          tiled_hlo_instructions_set.Insert(std::move(tiled_operands[id]));
      resolved_operands[id] = operand_tiled_hlo;
      if (inserted) {
        worklist.push_back(operand_tiled_hlo);
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

  std::vector<std::unique_ptr<TiledHloInstruction>> tiled_hlo_instructions =
      tiled_hlo_instructions_set.ExtractData();

  return TiledHloRegion{std::move(tiled_hlo_instructions),
                        std::move(canonical_roots)};
}

/*static*/ absl::StatusOr<TiledHloComputation> TiledHloComputation::Tile(
    const HloFusionAdaptor& fusion, std::unique_ptr<TilingSpace> tiling_space) {
  std::vector<std::unique_ptr<TiledHloInstruction>> tiled_roots;
  tiled_roots.reserve(fusion.GetRoots().size());
  for (const auto& [root, tile] :
       llvm::zip(fusion.GetRoots(), tiling_space->tiled_roots())) {
    tiled_roots.push_back(
        std::make_unique<TiledHloInstruction>(&root.instruction(), tile));
  }

  absl::flat_hash_map<int64_t, std::pair<const TiledHloInstruction*, Interval>>
      rt_symbol_to_tiled_hlo;
  ASSIGN_OR_RETURN(TiledHloRegion region,
                   CreateHloRegion(std::move(tiled_roots), fusion,
                                   *tiling_space, rt_symbol_to_tiled_hlo));

  return TiledHloComputation(std::move(tiling_space), std::move(region),
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
  for (const auto& tiled_hlo : region_.instructions()) {
    PrepopulateTileNames(tiled_hlo.get(), name_uniquer, tile_names);
  }

  ss << "Tiled HLO:\n";
  for (const auto& tiled_hlo : region_.instructions()) {
    PrintTiledHloInstruction(tiled_hlo.get(), tile_names, ss, /*indent=*/2);
  }
  return ss.str();
}

}  // namespace xla::gpu::experimental
