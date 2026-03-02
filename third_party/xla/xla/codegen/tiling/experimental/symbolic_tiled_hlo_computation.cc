/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/codegen/tiling/experimental/symbolic_tiled_hlo_computation.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/experimental/symbolic_tile.h"
#include "xla/codegen/tiling/experimental/symbolic_tile_propagation.h"
#include "xla/codegen/tiling/experimental/symbolic_tiled_hlo.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/instruction_fusion.h"
#include "xla/service/name_uniquer.h"
#include "xla/util.h"

namespace xla::gpu::experimental {
namespace {

using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::mlir::MLIRContext;

// A hash set of unique pointers.
//
// This set add a few key features on top of absl::flat_hash_set<T*>:
// * The set takes ownership of the object and deletes the object if an
//   equivalent element is already in the set.
// * Values are compared by the value behind the pointer, not the pointer
//   itself.
// * This set provides a convenient method to extract the unique pointers into a
//   vector.
// * Values are stored in the order of insertion. This is useful when we have
//   information about the order in which we process elements. For example,
//   during the construction of TiledHloComputation from
//   SymbolicTiledHloInstructions, we know that instruction are already sorted
//   in def-before-use order.
template <typename T>
class OrderedUniquePtrValueHashSet {
 public:
  // Inserts an element into the set.
  // Returns a pair of a non-owning raw pointer to the element that was inserted
  // (or the element that prevented insertion) and a bool indicating whether the
  // element was inserted.
  std::pair<T*, bool> Insert(std::unique_ptr<T> elem) {
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
  std::vector<std::unique_ptr<T>> ExtractData() { return std::move(data_); }

 private:
  struct PtrHash {
    size_t operator()(const T* v) const { return absl::HashOf(*v); }
  };

  struct PtrEqual {
    bool operator()(const T* lhs, const T* rhs) const {
      return lhs == rhs || *lhs == *rhs;
    }
  };

  // Stores non-owning pointers to the elements in the set. Elements are
  // compared by the value behind the pointer, not the pointer itself.
  absl::flat_hash_set<T*, PtrHash, PtrEqual> hash_set_;

  // Stores owning pointers to the elements in the set.
  std::vector<std::unique_ptr<T>> data_;
};

// Sorts tiled hlo instructions in def-before-use order, starting from
// `roots_with_no_users`. If instruction is not reachable from the root then it
// might be put in an arbitrary position.
void SortTiledHloInstructionsInPostOrder(
    std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>&
        tiled_hlo_instructions,
    ArrayRef<const SymbolicTiledHloInstruction*> roots_with_no_users) {
  absl::flat_hash_map<const SymbolicTiledHloInstruction*, int64_t>
      topological_order;

  std::function<void(const SymbolicTiledHloInstruction*)> visit_instruction;
  visit_instruction = [&](const SymbolicTiledHloInstruction* instruction) {
    if (topological_order.contains(instruction)) {
      return;
    }
    for (const SymbolicTiledHloInstruction* operand : instruction->operands()) {
      visit_instruction(operand);
    }
    topological_order[instruction] = topological_order.size();
  };

  for (const SymbolicTiledHloInstruction* root_with_no_user :
       roots_with_no_users) {
    visit_instruction(root_with_no_user);
  }
  absl::c_sort(tiled_hlo_instructions,
               [&](const std::unique_ptr<SymbolicTiledHloInstruction>& t1,
                   const std::unique_ptr<SymbolicTiledHloInstruction>& t2) {
                 return topological_order[t1.get()] <
                        topological_order[t2.get()];
               });
  if (VLOG_IS_ON(4)) {
    VLOG(4)
        << "Sorted symbolic tiled HLO instructions in def-before-use order:\n"
        << absl::StrJoin(tiled_hlo_instructions, "\n",
                         [](std::string* out,
                            const std::unique_ptr<SymbolicTiledHloInstruction>&
                                instruction) {
                           absl::StrAppend(out, instruction->ToString("; "));
                         });
  }
}

bool IsControlFlowLoop(const SymbolicTiledHloInstruction& tiled_hlo) {
  const HloOpcode hlo_opcode = tiled_hlo.hlo()->opcode();
  return hlo_opcode == HloOpcode::kDot || hlo_opcode == HloOpcode::kScaledDot ||
         hlo_opcode == HloOpcode::kReduce;
}

bool IsControlFlowCondition(const SymbolicTiledHloInstruction& tiled_hlo) {
  const HloOpcode hlo_opcode = tiled_hlo.hlo()->opcode();
  return hlo_opcode == HloOpcode::kConcatenate;
}

// Recursively populates `tile_names` with unique names for `tiled_hlo` and
// all instructions within its regions.
void PrepopulateTileNames(
    const SymbolicTiledHloInstruction* tiled_hlo, NameUniquer& name_uniquer,
    absl::flat_hash_map<const SymbolicTiledHloInstruction*, std::string>&
        tile_names) {
  auto [_, inserted] = tile_names.try_emplace(
      tiled_hlo, name_uniquer.GetUniqueName(
                     absl::StrCat(tiled_hlo->hlo()->name(), ".tile_0")));
  if (!inserted) {
    return;
  }
  for (const auto& region : tiled_hlo->regions()) {
    for (const auto& region_instruction : region) {
      PrepopulateTileNames(region_instruction.get(), name_uniquer, tile_names);
    }
  }
}

// Recursively prints `tiled_hlo` and all instructions within its regions.
void PrintTiledHloInstruction(
    const SymbolicTiledHloInstruction* tiled_hlo,
    const absl::flat_hash_map<const SymbolicTiledHloInstruction*, std::string>&
        tile_names,
    std::stringstream& ss, int indent) {
  std::string indentation(indent, ' ');
  absl::InlinedVector<std::string, 4> operand_names;
  for (const auto& operand : tiled_hlo->operands()) {
    CHECK(tile_names.contains(operand)) << operand->hlo()->name();
    operand_names.push_back(tile_names.at(operand));
  }

  ss << indentation << tile_names.at(tiled_hlo) << " = "
     << HloOpcodeString(tiled_hlo->hlo()->opcode()) << "("
     << absl::StrJoin(operand_names, ", ") << ") "
     << tiled_hlo->symbolic_tile().ToString(false) << "\n";

  if (!tiled_hlo->regions().empty()) {
    for (auto const& [i, region] : llvm::enumerate(tiled_hlo->regions())) {
      ss << indentation << "region #" << i << " {\n";
      for (const auto& instruction : region) {
        PrintTiledHloInstruction(instruction.get(), tile_names, ss, indent + 2);
      }
      ss << indentation << "}\n";
    }
  }
}

}  // namespace

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
/*static*/ SymbolicTiledHloRegionOrError SymbolicTiledComputation::CreateRegion(
    std::unique_ptr<SymbolicTiledHloInstruction> tiled_root,
    const HloFusionAdaptor& fusion, const TilingSpace& tiling_space) {
  std::vector<SymbolicTiledHloInstruction*> worklist = {tiled_root.get()};
  OrderedUniquePtrValueHashSet<SymbolicTiledHloInstruction>
      tiled_hlo_instructions_set;

  while (!worklist.empty()) {
    SymbolicTiledHloInstruction* tiled_hlo = worklist.back();
    worklist.pop_back();
    const HloInstruction* hlo = tiled_hlo->hlo();
    if (!fusion.ContainsInstruction(hlo) || hlo->operand_count() == 0) {
      continue;
    }

    auto operands_tiles = PropagateSymbolicTileToInput(
        tiling_space, *hlo, tiled_hlo->symbolic_tile(), 0);
    if (!operands_tiles.has_value()) {
      return FusionDecision::Forbid("Couldn't propagate tile ")
             << tiled_hlo->symbolic_tile().ToString() << " to the input of "
             << hlo->ToString();
    }

    HloInstructionAdaptor instruction_adaptor(*hlo, &fusion);
    auto tiles_and_operands =
        llvm::zip(*operands_tiles, instruction_adaptor.GetOperands());
    const bool hlo_is_condition = IsControlFlowCondition(*tiled_hlo);
    for (const auto& [tile, operand] : tiles_and_operands) {
      const HloInstruction* operand_hlo = &operand.instruction();
      auto tiled_operand =
          std::make_unique<SymbolicTiledHloInstruction>(operand_hlo, tile);
      const bool operand_is_loop = IsControlFlowLoop(*tiled_operand);

      if (hlo_is_condition || operand_is_loop) {
        auto region_or_error =
            CreateRegion(std::move(tiled_operand), fusion, tiling_space);
        if (auto* decision = std::get_if<FusionDecision>(&region_or_error)) {
          return *decision;
        }
        auto region = std::get<SymbolicTiledHloInstruction::Region>(
            std::move(region_or_error));

        if (hlo_is_condition) {
          // Case 1: HLO is a condition (e.g., concat).
          // Each operand introduces a new branch/sub-region in `tiled_hlo`.
          CHECK(!region.empty()) << "CreateRegion: returned empty region for "
                                 << operand_hlo->ToString();
          tiled_hlo->AppendOperand(region.back().get());
          tiled_hlo->AddRegion(std::move(region));

        } else {
          // Case 2: Operand is a loop (e.g., dot/scaled_dot/reduce).
          // Operand has its loop-body as a region. Operand itself is added as a
          // node to the current flat list.
          CHECK(region.size() == 1)
              << "CreateRegion: expected exactly 1 region for "
              << operand_hlo->ToString() << " but got " << region.size();
          tiled_hlo->AppendOperand(region.back().get());
          tiled_hlo_instructions_set.Insert(std::move(region.back()));
        }

      } else {
        // Case 3: No new region introduced when processing this operand.
        auto [operand_tiled_hlo, inserted] =
            tiled_hlo_instructions_set.Insert(std::move(tiled_operand));
        if (inserted) {
          worklist.push_back(operand_tiled_hlo);
        }
        tiled_hlo->AppendOperand(operand_tiled_hlo);
      }
    }
  }

  auto tiled_hlo_instructions = tiled_hlo_instructions_set.ExtractData();
  SortTiledHloInstructionsInPostOrder(tiled_hlo_instructions, tiled_root.get());
  if (IsControlFlowLoop(*tiled_root)) {
    tiled_root->AddRegion(std::move(tiled_hlo_instructions));
    tiled_hlo_instructions.clear();
  }
  tiled_hlo_instructions.push_back(std::move(tiled_root));
  return tiled_hlo_instructions;
}

/*static*/ SymbolicTileAnalysisOrError SymbolicTiledComputation::Tile(
    const HloFusionAdaptor& fusion, MLIRContext* ctx) {
  auto tiling_space = TilingSpace::Create(fusion, ctx);

  SmallVector<const SymbolicTiledHloInstruction*> roots_with_no_users;
  OrderedUniquePtrValueHashSet<SymbolicTiledHloInstruction>
      tiled_hlo_instructions_set;

  for (const auto& [root, tile] :
       llvm::zip(fusion.GetRoots(), tiling_space->tiled_roots())) {
    auto root_tiled_hlo = std::make_unique<SymbolicTiledHloInstruction>(
        &root.instruction(), tile);
    if (root.GetUsers().empty()) {
      roots_with_no_users.push_back(root_tiled_hlo.get());
    }

    auto region_or_error =
        CreateRegion(std::move(root_tiled_hlo), fusion, *tiling_space);
    if (auto* decision = std::get_if<FusionDecision>(&region_or_error)) {
      return *decision;
    }
    auto region = std::get<SymbolicTiledHloInstruction::Region>(
        std::move(region_or_error));
    for (auto& tiled_hlo : region) {
      tiled_hlo_instructions_set.Insert(std::move(tiled_hlo));
    }
  }

  std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>
      tiled_hlo_instructions = tiled_hlo_instructions_set.ExtractData();

  // Order instructions in def-before-use order.
  SortTiledHloInstructionsInPostOrder(tiled_hlo_instructions,
                                      roots_with_no_users);

  return SymbolicTiledComputation(std::move(tiling_space),
                                  std::move(tiled_hlo_instructions));
}

std::string SymbolicTiledComputation::ToString() const {
  std::stringstream ss;

  ss << tiling_space_->ToString() << "\n";

  NameUniquer name_uniquer("_");
  absl::flat_hash_map<const SymbolicTiledHloInstruction*, std::string>
      tile_names;
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
