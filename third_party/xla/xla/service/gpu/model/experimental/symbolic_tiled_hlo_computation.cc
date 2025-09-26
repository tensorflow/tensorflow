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

#include "xla/service/gpu/model/experimental/symbolic_tiled_hlo_computation.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
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
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/model/experimental/symbolic_tile.h"
#include "xla/service/gpu/model/experimental/symbolic_tile_propagation.h"
#include "xla/service/gpu/model/experimental/symbolic_tiled_hlo.h"
#include "xla/service/gpu/model/experimental/tiling_space.h"
#include "xla/service/instruction_fusion.h"
#include "xla/service/name_uniquer.h"
#include "xla/util.h"

namespace xla::gpu::experimental {
namespace {

using ::llvm::ArrayRef;
using ::llvm::SmallVector;
using ::mlir::AffineExpr;
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

/*static*/ SymbolicTileAnalysisOrError SymbolicTiledComputation::Tile(
    const HloFusionAdaptor& fusion, MLIRContext* ctx) {
  auto tiling_space = TilingSpace::Create(fusion, ctx);

  // Add root instructions to the worklist.
  std::vector<SymbolicTiledHloInstruction*> worklist;
  SmallVector<SymbolicTiledHloInstruction*> roots_with_no_users;
  OrderedUniquePtrValueHashSet<SymbolicTiledHloInstruction>
      tiled_hlo_instructions_set;
  for (const auto& [root, tile] :
       llvm::zip(fusion.GetRoots(), tiling_space->tiled_roots())) {
    auto [root_tiled_hlo, _] = tiled_hlo_instructions_set.Insert(
        std::make_unique<SymbolicTiledHloInstruction>(&root.instruction(),
                                                      tile));
    if (root.GetUsers().empty()) {
      roots_with_no_users.push_back(root_tiled_hlo);
    }
    worklist.push_back(root_tiled_hlo);
  }

  while (!worklist.empty()) {
    SymbolicTiledHloInstruction* tiled_hlo_instruction = worklist.back();
    worklist.pop_back();
    const HloInstruction* hlo = tiled_hlo_instruction->hlo();

    if (!fusion.ContainsInstruction(hlo)) {
      continue;
    }
    HloInstructionAdaptor instruction_adaptor(*hlo, &fusion);
    if (hlo->operand_count() == 0) {
      continue;
    }
    auto tiled_operands = PropagateTileToInput(
        *tiling_space, *hlo, tiled_hlo_instruction->symbolic_tile(),
        /*output_id=*/0);

    if (!tiled_operands.has_value()) {
      return FusionDecision::Forbid("Couldn't propagate tile ")
             << tiled_hlo_instruction->symbolic_tile().ToString()
             << " to the input of " << hlo->ToString();
    }
    for (const auto& [tile, operand] :
         llvm::zip(*tiled_operands, instruction_adaptor.GetOperands())) {
      auto tiled_operand = std::make_unique<SymbolicTiledHloInstruction>(
          &operand.instruction(), tile);
      auto [operand_tiled_hlo, inserted] =
          tiled_hlo_instructions_set.Insert(std::move(tiled_operand));
      tiled_hlo_instruction->AppendOperand(operand_tiled_hlo);
      worklist.push_back(operand_tiled_hlo);
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
  absl::flat_hash_map<SymbolicTiledHloInstruction*, std::string> tile_names;

  ss << "Tiled HLO:\n";
  for (const auto& tiled_hlo : tiled_hlo_instructions_) {
    std::string tile_name = name_uniquer.GetUniqueName(
        absl::StrCat(tiled_hlo->hlo()->name(), ".tile_0"));
    tile_names[tiled_hlo.get()] = tile_name;

    absl::InlinedVector<std::string, 4> operand_names;
    for (const auto& operand : tiled_hlo->operands()) {
      operand_names.push_back(tile_names.at(operand));
    }

    ss << tile_name << " = " << HloOpcodeString(tiled_hlo->hlo()->opcode())
       << "(" << absl::StrJoin(operand_names, ", ") << ") ";

    ss << tiled_hlo->symbolic_tile().ToString(false) << "\n";
  }
  return ss.str();
}

}  // namespace xla::gpu::experimental
