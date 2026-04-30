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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
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
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"
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
#include "xla/service/instruction_fusion.h"
#include "xla/service/name_uniquer.h"
#include "xla/util.h"

namespace xla::gpu::experimental {

using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::mlir::MLIRContext;

std::string TiledHloInstruction::ToString(
    absl::string_view field_separator) const {
  std::stringstream ss;
  ss << "hlo: " << hlo_->ToString() << field_separator;
  ss << "tile: " << tile().ToString();
  for (const auto& [index, region] : llvm::enumerate(regions_)) {
    ss << field_separator << "region #" << index << " {";
    for (const auto& instruction : region) {
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
    for (int i = dyn_slice->first_index_operand_number();
         i < hlo_->operand_count(); ++i) {
      runtime_variables.push_back(operands_[i]);
    }
  }
  return runtime_variables;
}

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
//   information about the order in which we process elements. For example,
//   during the construction of TiledHloComputation from
//   TiledHloInstructions, we know that instruction are already sorted
//   in def-before-use order.
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
  absl::c_sort(tiled_hlo_instructions,
               [&](const std::unique_ptr<TiledHloInstruction>& t1,
                   const std::unique_ptr<TiledHloInstruction>& t2) {
                 return topological_order[t1.get()] <
                        topological_order[t2.get()];
               });
  if (VLOG_IS_ON(4)) {
    VLOG(4)
        << "Sorted symbolic tiled HLO instructions in def-before-use order:\n"
        << absl::StrJoin(
               tiled_hlo_instructions, "\n",
               [](std::string* out,
                  const std::unique_ptr<TiledHloInstruction>& instruction) {
                 absl::StrAppend(out, instruction->ToString("; "));
               });
  }
}

bool IsControlFlowLoop(const TiledHloInstruction& tiled_hlo) {
  const HloOpcode hlo_opcode = tiled_hlo.hlo()->opcode();
  return hlo_opcode == HloOpcode::kDot || hlo_opcode == HloOpcode::kScaledDot;
}

bool IsControlFlowCondition(const TiledHloInstruction& tiled_hlo) {
  const HloOpcode hlo_opcode = tiled_hlo.hlo()->opcode();
  return hlo_opcode == HloOpcode::kConcatenate;
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
    for (const auto& region_instruction : region) {
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
    for (const auto& instruction : region) {
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

// Returns a region with `tiled_root` and subgraph HLOs (in tiled_root.regions
// or in the returned vector). E.g.,
// * If tiled_root is a dot/scaled_dot/reduce
//   returns {tiled_root}, and tiled_root has one region including all
//   dependencies.
// * If tiled_root is a concat,
//   returns {tiled_root}, and tiled_root has one region per operand.
// * Otherwise, returns a region including tiled_root and all dependencies.
/*static*/ absl::StatusOr<TiledHloRegion> TiledHloComputation::CreateHloRegion(
    std::unique_ptr<TiledHloInstruction> tiled_root,
    const HloFusionAdaptor& fusion, const TilingSpace& tiling_space,
    absl::flat_hash_map<int64_t,
                        std::pair<const TiledHloInstruction*, Interval>>&
        rt_symbol_to_tiled_hlo) {
  std::vector<TiledHloInstruction*> worklist = {tiled_root.get()};
  OrderedTiledHloPtrSet tiled_hlo_instructions_set;

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

    HloInstructionAdaptor instruction_adaptor(*hlo, &fusion);
    const bool hlo_is_condition = IsControlFlowCondition(*tiled_hlo);
    for (const auto& [operand_id, tile_and_operand] : llvm::enumerate(
             llvm::zip(operands_tiles, instruction_adaptor.GetOperands()))) {
      auto& [tile, operand] = tile_and_operand;
      const HloInstruction* operand_hlo = &operand.instruction();
      auto tiled_operand =
          std::make_unique<TiledHloInstruction>(operand_hlo, tile);
      const bool operand_is_loop = IsControlFlowLoop(*tiled_operand);

      if (hlo_is_condition || operand_is_loop) {
        ASSIGN_OR_RETURN(auto region,
                         CreateHloRegion(std::move(tiled_operand), fusion,
                                         tiling_space, rt_symbol_to_tiled_hlo));

        if (hlo_is_condition) {
          // Case 1: HLO is a condition (e.g., concat).
          // Each operand introduces a new branch/sub-region in `tiled_hlo`.
          CHECK(!region.empty())
              << "CreateHloRegion: returned empty region for "
              << operand_hlo->ToString();
          tiled_hlo->AddOperand(region.back().get());
          tiled_hlo->AddHloRegion(std::move(region));

        } else {
          // Case 2: Operand is a loop (e.g., dot/scaled_dot/reduce).
          // Operand has its loop-body as a region. Operand itself is added as a
          // node to the current flat list.
          CHECK(region.size() == 1)
              << "CreateHloRegion: expected exactly 1 region for "
              << operand_hlo->ToString() << " but got " << region.size();
          tiled_hlo->AddOperand(region.back().get());
          tiled_hlo_instructions_set.Insert(std::move(region.back()));
        }

      } else {
        // Case 3: No new region introduced when processing this operand.
        auto [operand_tiled_hlo, inserted] =
            tiled_hlo_instructions_set.Insert(std::move(tiled_operand));
        if (inserted) {
          worklist.push_back(operand_tiled_hlo);
        }
        tiled_hlo->AddOperand(operand_tiled_hlo);
        // If the operand is a runtime variable, add it to the
        // `rt_symbol_to_tiled_hlo` map.
        std::optional<const TilingSpace::RTVarInfo*> rt_var_info =
            tiling_space.GetRTVarInfo(*hlo, operand_id);
        if (rt_var_info.has_value()) {
          rt_symbol_to_tiled_hlo.insert(std::make_pair(
              rt_var_info.value()->id + tiling_space.num_dimensions(),
              std::make_pair(operand_tiled_hlo, rt_var_info.value()->bounds)));
        }
      }
    }
  }
  TiledHloRegion tiled_hlo_instructions{
      tiled_hlo_instructions_set.ExtractData()};
  SortTiledHloInstructionsInPostOrder(tiled_hlo_instructions, tiled_root.get());
  if (IsControlFlowLoop(*tiled_root)) {
    tiled_root->AddHloRegion(std::move(tiled_hlo_instructions));
    tiled_hlo_instructions.clear();
  }
  tiled_hlo_instructions.push_back(std::move(tiled_root));
  return tiled_hlo_instructions;
}

/*static*/ absl::StatusOr<TiledHloComputation> TiledHloComputation::Tile(
    const HloFusionAdaptor& fusion, std::unique_ptr<TilingSpace> tiling_space) {
  SmallVector<const TiledHloInstruction*> roots;
  SmallVector<const TiledHloInstruction*> roots_with_no_users;
  OrderedTiledHloPtrSet tiled_hlo_instructions_set;

  absl::flat_hash_map<int64_t, std::pair<const TiledHloInstruction*, Interval>>
      rt_symbol_to_tiled_hlo;
  for (const auto& [root, tile] :
       llvm::zip(fusion.GetRoots(), tiling_space->tiled_roots())) {
    auto root_tiled_hlo =
        std::make_unique<TiledHloInstruction>(&root.instruction(), tile);
    roots.push_back(root_tiled_hlo.get());
    if (root.GetUsers().empty()) {
      roots_with_no_users.push_back(root_tiled_hlo.get());
    }

    ASSIGN_OR_RETURN(TiledHloRegion region,
                     CreateHloRegion(std::move(root_tiled_hlo), fusion,
                                     *tiling_space, rt_symbol_to_tiled_hlo));
    for (std::unique_ptr<TiledHloInstruction>& tiled_hlo : region) {
      tiled_hlo_instructions_set.Insert(std::move(tiled_hlo));
    }
  }

  std::vector<std::unique_ptr<TiledHloInstruction>> tiled_hlo_instructions =
      tiled_hlo_instructions_set.ExtractData();

  // Order instructions in def-before-use order.
  SortTiledHloInstructionsInPostOrder(tiled_hlo_instructions,
                                      roots_with_no_users);

  return TiledHloComputation(std::move(tiling_space),
                             TiledHloRegion{std::move(tiled_hlo_instructions)},
                             std::move(roots),
                             std::move(rt_symbol_to_tiled_hlo));
}

std::string TiledHloComputation::ToString() const {
  std::stringstream ss;

  ss << tiling_space_->ToString() << "\n";

  NameUniquer name_uniquer("_");
  absl::flat_hash_map<const TiledHloInstruction*, std::string> tile_names;
  for (const auto& tiled_hlo : tiled_hlo_instructions_) {
    PrepopulateTileNames(tiled_hlo.get(), name_uniquer, tile_names);
  }

  ss << "Tiled HLO:\n";
  for (const auto& tiled_hlo : tiled_hlo_instructions_) {
    PrintTiledHloInstruction(tiled_hlo.get(), tile_names, ss, /*indent=*/2);
  }
  return ss.str();
}

}  // namespace xla::gpu::experimental
