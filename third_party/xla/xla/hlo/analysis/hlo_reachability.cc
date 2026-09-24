/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/analysis/hlo_reachability.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/base/prefetch.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/numeric/bits.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

HloReachabilityMap::HloReachabilityMap(
    absl::Span<const HloInstruction* const> instructions)
    : words_per_bitset_((instructions.size() + BitSet::kBits - 1) /
                        BitSet::kBits),
      total_words_((instructions.size() + 1 /*for tmp_bit_set_*/) *
                   words_per_bitset_) {
  if (!instructions.empty()) {
    CHECK(instructions[0]->parent() != nullptr)
        << "Instruction must be in a computation.";
    computation_id_ = instructions[0]->parent()->unique_id();
  } else {
    computation_id_ = kComputationIdAbsent;
  }
  uint64_t row = 0;
  uint64_t total_rows = instructions.size() + 1;  // for tmp_bit_set_
  while (row < total_rows) {
    const int rows_to_allocate = std::min(kRowsPerAllocation, total_rows - row);
    size_t words_to_allocate = rows_to_allocate * words_per_bitset_;
    // make_unique initializes the array of words to 0
    bit_storage_.push_back(std::make_unique<BitSet::Word[]>(words_to_allocate));
    row += rows_to_allocate;
  }

  tmp_bit_set_ = BitSetFromIndex(instructions.size());
  int32_t max_local_id = 0;
  for (const HloInstruction* instruction : instructions) {
    max_local_id = std::max(max_local_id, instruction->local_id());
  }
  indices_.resize(max_local_id + 1, kValueAbsent);
  for (size_t i = 0; i < instructions.size(); ++i) {
    BitSetFromIndex(i).Set(i);  // Instructions are reachable from themselves.
    indices_[GetKey(instructions[i])] = i;
  }
}

bool HloReachabilityMap::SetReachabilityToUnion(
    absl::Span<const HloInstruction* const> inputs,
    const HloInstruction* instruction) {
  Index index = GetIndex(instruction);
  BitSet bit_set = BitSetFromIndex(index);
  tmp_bit_set_.CopyBitSet(bit_set);
  SetReachabilityToUnionHelper(inputs, index);
  return bit_set != tmp_bit_set_;
}

void HloReachabilityMap::FastSetReachabilityToUnion(
    absl::Span<const HloInstruction* const> inputs,
    const HloInstruction* instruction) {
  SetReachabilityToUnionHelper(inputs, GetIndex(instruction));
}

void HloReachabilityMap::FastSetReachabilityToUnion(
    absl::Span<const Index> input_indices, Index index) {
  SetReachabilityToUnionHelper(input_indices, index);
}

void HloReachabilityMap::SetReachabilityToUnionHelper(
    absl::Span<const HloInstruction* const> inputs, Index index) {
  absl::InlinedVector<Index, 16> input_indices;
  input_indices.reserve(inputs.size());
  for (const HloInstruction* input : inputs) {
    input_indices.push_back(GetIndex(input));
  }
  SetReachabilityToUnionHelper(input_indices, index);
}

void HloReachabilityMap::SetReachabilityToUnionHelper(
    absl::Span<const Index> input_indices, Index index) {
  BitSet bit_set = BitSetFromIndex(index);
  // If instruction is part of inputs, don't reset the bit-set.
  if (!absl::c_linear_search(input_indices, index)) {
    bit_set.SetToZero();
  }
  bit_set.Set(index);
  for (Index input_index : input_indices) {
    if (input_index != index) {
      bit_set |= BitSetFromIndex(input_index);
    }
  }
}

void HloReachabilityMap::Replace(const HloInstruction* original,
                                 const HloInstruction* replacement) {
  Key original_key = GetKey(original);
  Key replacement_key = GetKey(replacement);
  if (original_key != replacement_key) {
    DCHECK_LT(original_key, indices_.size());
    if (replacement_key >= indices_.size()) {
      indices_.resize(replacement_key + 1, kValueAbsent);
    }
    indices_[replacement_key] = GetIndex(original);
    indices_[original_key] = kValueAbsent;
  }
}

std::unique_ptr<HloReachabilityMap> HloReachabilityMap::BuildWithRestrictions(
    const HloComputation* computation,
    absl::FunctionRef<void(const HloInstruction*,
                           std::vector<HloInstruction*>*)>
        add_dependencies) {
  const auto& all = computation->MakeInstructionPostOrder();
  auto result = std::make_unique<HloReachabilityMap>(all);

  std::vector<HloInstruction*> inputs;
  for (const HloInstruction* hlo : all) {
    inputs.clear();
    add_dependencies(hlo, &inputs);
    result->FastSetReachabilityToUnion(inputs, hlo);
  }
  return result;
}

std::unique_ptr<HloReachabilityMap> HloReachabilityMap::Build(
    const HloComputation* computation) {
  std::vector<HloInstruction*> instructions =
      computation->MakeInstructionPostOrder();
  auto result = std::make_unique<HloReachabilityMap>(instructions);

  auto get_bit_set = [&](const HloInstruction* instruction) -> BitSet {
    return result->BitSetFromIndex(result->GetIndex(instruction));
  };

  for (const HloInstruction* instruction : instructions) {
    BitSet bit_set = get_bit_set(instruction);

    auto add_dependencies = [&](const HloInstruction* instruction) {
      for (const HloInstruction* operand : instruction->operands()) {
        bit_set |= get_bit_set(operand);
      }
      for (const HloInstruction* predecessor :
           instruction->control_predecessors()) {
        bit_set |= get_bit_set(predecessor);
      }
    };

    add_dependencies(instruction);
  }
  return result;
}

void HloReachabilityMap::UpdateReachabilityThroughInstruction(
    const HloInstruction* instruction) {
  std::queue<const HloInstruction*> worklist;
  worklist.push(instruction);

  std::vector<HloInstruction*> inputs;

  // Keep track of the number of times an instruction is in the worklist and
  // only process it only if it is the last occurrence. Note that this might
  // still mean that an instruction is processed multiple times.
  absl::flat_hash_map<const HloInstruction*, int64_t> in_worklist;

  while (!worklist.empty()) {
    const HloInstruction* item = worklist.front();
    worklist.pop();
    --in_worklist[item];
    if (in_worklist[item] > 0) {
      continue;
    }

    inputs.assign(item->operands().begin(), item->operands().end());
    inputs.insert(inputs.end(), item->control_predecessors().begin(),
                  item->control_predecessors().end());

    if (SetReachabilityToUnion(inputs, item)) {
      // Add immediate successors to worklist.
      for (const HloInstruction* user : item->users()) {
        worklist.push(user);
        ++in_worklist[user];
      }
      for (const HloInstruction* succ : item->control_successors()) {
        worklist.push(succ);
        ++in_worklist[succ];
      }
    }
  }
}

// Use ptr tagging in `worklist` to check if current instruction is successor of
// left or right.
static constexpr uintptr_t FROM_LEFT_FLAG_MASK = 1;
static constexpr uintptr_t PTR_MASK = ~FROM_LEFT_FLAG_MASK;
static_assert(alignof(HloInstruction) >= 2,
              "HloInstruction must be aligned to at least 2 bytes");
void HloReachabilityMap::UpdateReachabilityForMerge(
    const HloInstruction* left, const HloInstruction* right) {
  DCHECK(tmp_worklist_.empty());
  DCHECK(tmp_indices_to_update_.empty());
  DCHECK(IsKeyPresent(GetKey(left)));
  DCHECK(IsKeyPresent(GetKey(right)));

  if (left == right) {
    return;
  }

  Index left_index = GetIndex(left);
  Index right_index = GetIndex(right);
  BitSet left_bit_set = BitSetFromIndex(left_index);
  BitSet right_bit_set = BitSetFromIndex(right_index);

  absl::flat_hash_set<const HloInstruction*> visited;
  auto add_to_worklist = [&](const HloInstruction* instr,
                             bool from_left) -> void {
    if (visited.insert(instr).second) {
      if (IsKeyPresent(GetKey(instr))) {
        BitSet bit_set = BitSetFromIndex(GetIndex(instr));
        // If the node is already reachable from both sides, we can skip it.
        if ((from_left && bit_set.Get(right_index)) ||
            (!from_left && bit_set.Get(left_index))) {
          return;
        }
      }
      uintptr_t raw_addr = reinterpret_cast<uintptr_t>(instr);
      tmp_worklist_.push_back(raw_addr | from_left);
      return;
    }
    return;
  };
  add_to_worklist(left, /*from_left=*/true);
  const bool left_added = !tmp_worklist_.empty();
  add_to_worklist(right, /*from_left=*/false);
  if (tmp_worklist_.empty()) {
    return;
  }
  left_bit_set.GetDifferingWordUnions(right_bit_set, tmp_changed_words_);
  if (tmp_changed_words_.empty()) {
    tmp_worklist_.clear();
    return;
  }
  while (!tmp_worklist_.empty()) {
    const uintptr_t item_and_from_left = tmp_worklist_.back();
    tmp_worklist_.pop_back();

    // Use ptr tagging to show if instruction is successor of left or right.
    const bool from_left = (item_and_from_left & FROM_LEFT_FLAG_MASK);
    const HloInstruction* item =
        reinterpret_cast<const HloInstruction*>(item_and_from_left & PTR_MASK);

    if (IsKeyPresent(GetKey(item))) {
      tmp_indices_to_update_.push_back(GetIndex(item));
    }
    for (const HloInstruction* user : item->users()) {
      add_to_worklist(user, from_left);
    }
    for (const HloInstruction* succ : item->control_successors()) {
      add_to_worklist(succ, from_left);
    }
  }
  DCHECK(tmp_worklist_.empty());
  // Based on the benchmarks, if the number of changed words is more than 25% of
  // the size of the bitset, it is faster do full |= instead of using
  // OrUpdatePartial.
  if (tmp_changed_words_.size() > words_per_bitset_ * 0.25) {
    // Is is guaranteed that either left_bit_set or right_bit_set is in
    // tmp_indices_to_update_ based on the logic above.
    BitSet info = left_added ? left_bit_set : right_bit_set;
    info.OrUpdatePartial(tmp_changed_words_);
    for (Index index : tmp_indices_to_update_) {
      BitSet bit_set = BitSetFromIndex(index);
      bit_set |= info;
    }
  } else {
    for (Index index : tmp_indices_to_update_) {
      BitSet bit_set = BitSetFromIndex(index);
      bit_set.OrUpdatePartial(tmp_changed_words_);
    }
  }
  tmp_changed_words_.clear();
  tmp_indices_to_update_.clear();
}

namespace {

// UpdateMultipleInstructions forwards the dirty words of a row to a successor
// as runs of consecutive words, each run one vectorized union, or the whole
// row as one union when that is cheaper. A run costs about as much as the
// union of this many words on top of its own length: measured with
// BM_HloReachabilityUpdateMultipleInstructionsComb, where ten one word runs
// cost as much as the union of a 257 word row.
constexpr size_t kWordsPerRun = 24;

// Calls fn on instruction if is_present(instruction). Otherwise looks
// through it: the instructions that neighbors yields for it are handled the
// same way, transitively. An instruction absent from the map has no row, but
// paths through it still connect instructions that do.
template <typename IsPresent, typename Neighbors, typename Fn>
void ForEachPresentThrough(const HloInstruction* instruction,
                           const IsPresent& is_present,
                           const Neighbors& neighbors, const Fn& fn) {
  if (is_present(instruction)) {
    fn(instruction);
    return;
  }
  absl::InlinedVector<const HloInstruction*, 4> absent = {instruction};
  for (size_t i = 0; i < absent.size(); ++i) {
    neighbors(absent[i], [&](const HloInstruction* neighbor) {
      if (is_present(neighbor)) {
        fn(neighbor);
      } else if (!absl::c_linear_search(absent, neighbor)) {
        absent.push_back(neighbor);
      }
    });
  }
}

}  // namespace

void HloReachabilityMap::UpdateMultipleInstructions(
    const absl::flat_hash_map<const HloInstruction*,
                              absl::flat_hash_set<const HloInstruction*>>&
        to_update) {
  const size_t num_words = words_per_bitset_;
  const size_t mask_words = (num_words + BitSet::kBits - 1) / BitSet::kBits;
  if (tmp_pending_state_.size() < indices_.size()) {
    tmp_pending_state_.resize(indices_.size(), kPendingNone);
    tmp_dirty_masks_.resize(indices_.size() * mask_words, 0);
  }
  if (tmp_delta_words_.size() < num_words) {
    tmp_delta_words_.resize(num_words);
  }

  const auto is_present = [this](const HloInstruction* instruction) {
    return IsKeyPresent(GetKey(instruction));
  };
  const auto successors = [](const HloInstruction* instruction,
                             const auto& visit) {
    for (const HloInstruction* user : instruction->users()) {
      visit(user);
    }
    for (const HloInstruction* successor : instruction->control_successors()) {
      visit(successor);
    }
  };
  const auto predecessors = [](const HloInstruction* instruction,
                               const auto& visit) {
    for (const HloInstruction* operand : instruction->operands()) {
      visit(operand);
    }
    for (const HloInstruction* predecessor :
         instruction->control_predecessors()) {
      visit(predecessor);
    }
  };
  const auto mask_of = [&](Key key) {
    return tmp_dirty_masks_.data() + key * mask_words;
  };
  // Marks the words [begin, end) in mask.
  const auto mark_dirty = [](BitSet::Word* mask, size_t begin, size_t end) {
    for (size_t word = begin; word < end;) {
      const size_t bit = word % BitSet::kBits;
      const size_t count = std::min(BitSet::kBits - bit, end - word);
      const BitSet::Word bits = count == BitSet::kBits
                                    ? ~BitSet::Word{0}
                                    : ((BitSet::Word{1} << count) - 1) << bit;
      mask[word / BitSet::kBits] |= bits;
      word += count;
    }
  };

  // Min heap by row index. Build assigns indices in post order, so a row is
  // normally popped once, after all of its predecessors. The result does not
  // depend on the order: a change that arrives later queues the row again.
  using Item = std::pair<Index, const HloInstruction*>;
  std::vector<Item>& worklist = tmp_pending_rows_;
  DCHECK(worklist.empty());
  // Sets the pending state of target and queues it if it was not pending.
  // A row that graduates from pending words to a pending row drops its mask.
  const auto set_pending =
      [&](const HloInstruction* target, Index index, PendingState& state,
          PendingState pending) ABSL_ATTRIBUTE_ALWAYS_INLINE {
        if (state == kPendingNone) {
          worklist.emplace_back(index, target);
          // A chain keeps one row queued at a time; a heap of one needs no
          // sift.
          if (worklist.size() > 1) {
            std::push_heap(worklist.begin(), worklist.end(),
                           std::greater<Item>());
          }
        } else if (state == kPendingWords && pending == kPendingRow) {
          BitSet::Word* mask = mask_of(GetKey(target));
          std::fill(mask, mask + mask_words, 0);
        }
        state = pending;
      };

  // Merges the row source of a new predecessor into the row of target
  // and flags the words that changed in the dirty mask of target.
  const auto seed_row = [&](const HloInstruction* target,
                            const BitSet& source) {
    const Key key = GetKey(target);
    const Index index = indices_[key];
    BitSet row = BitSetFromIndex(index);
    PendingState& state = tmp_pending_state_[key];
    if (state == kPendingRow) {
      row |= source;
      return;
    }
    BitSet::Word* const delta = tmp_delta_words_.data();
    row.OrUpdateDelta(source, delta);
    BitSet::Word* mask = mask_of(key);
    BitSet::Word changed = 0;
    for (size_t i = 0; i < num_words; ++i) {
      const BitSet::Word bit = static_cast<BitSet::Word>(delta[i] != 0)
                               << (i % BitSet::kBits);
      mask[i / BitSet::kBits] |= bit;
      changed |= bit;
    }
    if (changed != 0) {
      set_pending(target, index, state, kPendingWords);
    }
  };

  // Merges the row source, forwarded whole, into the row of target, which
  // then forwards its whole row too if it changed: one vectorized union per
  // edge, the old cost of every pop.
  const auto absorb_row =
      [&](const HloInstruction* target, const BitSet& source)
          ABSL_ATTRIBUTE_ALWAYS_INLINE {
            const Key key = GetKey(target);
            const Index index = indices_[key];
            BitSet row = BitSetFromIndex(index);
            PendingState& state = tmp_pending_state_[key];
            if (state == kPendingRow) {
              row |= source;
            } else if (row.OrUpdate(source)) {
              set_pending(target, index, state, kPendingRow);
            }
          };

  // Merges the runs of words runs of the row source into the row of
  // target and flags the runs that changed in the dirty mask of target.
  using Run = std::pair<uint32_t, uint32_t>;
  const auto absorb_runs = [&](const HloInstruction* target,
                               const BitSet& source,
                               absl::Span<const Run> runs) {
    const Key key = GetKey(target);
    const Index index = indices_[key];
    BitSet row = BitSetFromIndex(index);
    PendingState& state = tmp_pending_state_[key];
    if (state == kPendingRow) {
      for (const auto& [begin, end] : runs) {
        row.OrRange(source, begin, end);
      }
      return;
    }
    BitSet::Word* mask = mask_of(key);
    bool changed = false;
    for (const auto& [begin, end] : runs) {
      if (row.OrUpdateRange(source, begin, end)) {
        mark_dirty(mask, begin, end);
        changed = true;
      }
    }
    if (changed) {
      set_pending(target, index, state, kPendingWords);
    }
  };

  // Seed: every updated row absorbs the rows of its new predecessors.
  absl::InlinedVector<const HloInstruction*, 4> targets;
  absl::InlinedVector<const HloInstruction*, 4> sources;
  // NOLINTNEXTLINE the loop aggregation is order independent.
  for (const auto& [instruction, new_predecessors] : to_update) {
    targets.clear();
    ForEachPresentThrough(
        instruction, is_present, successors,
        [&](const HloInstruction* target) { targets.push_back(target); });
    // NOLINTNEXTLINE the loop aggregation is order independent.
    for (const HloInstruction* predecessor : new_predecessors) {
      DCHECK(
          instruction->IsUserOf(predecessor) ||
          absl::c_linear_search(predecessor->control_successors(), instruction))
          << "The new edge from " << predecessor->name() << " to "
          << instruction->name() << " is not in the graph.";
      sources.clear();
      ForEachPresentThrough(
          predecessor, is_present, predecessors,
          [&](const HloInstruction* source) { sources.push_back(source); });
      for (const HloInstruction* source : sources) {
        const BitSet source_row = BitSetFromIndex(GetIndex(source));
        for (const HloInstruction* target : targets) {
          seed_row(target, source_row);
        }
      }
    }
  }

  // Returns the only successor of instruction if it has exactly one, else
  // nullptr. Counts what successors visits.
  const auto only_successor =
      [](const HloInstruction* instruction) -> const HloInstruction* {
    const auto& users = instruction->users();
    const auto& control = instruction->control_successors();
    if (users.size() + control.size() != 1) {
      return nullptr;
    }
    return users.empty() ? control.front() : users.front();
  };

  absl::InlinedVector<Run, 16> runs;
  while (!worklist.empty()) {
    if (worklist.size() > 1) {
      std::pop_heap(worklist.begin(), worklist.end(), std::greater<Item>());
    }
    auto [index, instruction] = worklist.back();
    worklist.pop_back();
    // Set when instruction was reached by tail forwarding below: its whole
    // row is pending, and neither its state nor its mask were touched.
    bool in_hand = false;
    bool whole_row = false;
    for (;;) {
      if (!in_hand) {
        const Key key = GetKey(instruction);
        PendingState& state = tmp_pending_state_[key];
        whole_row = state == kPendingRow;
        if (!whole_row) {
          // The dirty words as runs of consecutive words, clearing the mask.
          BitSet::Word* mask = mask_of(key);
          runs.clear();
          size_t dirty_words = 0;
          for (size_t block = 0; block < mask_words; ++block) {
            BitSet::Word bits = mask[block];
            mask[block] = 0;
            while (bits != 0) {
              const size_t low = absl::countr_zero(bits);
              const size_t count = absl::countr_one(bits >> low);
              const uint32_t begin = block * BitSet::kBits + low;
              const uint32_t end = begin + count;
              if (!runs.empty() && runs.back().second == begin) {
                runs.back().second = end;
              } else {
                runs.emplace_back(begin, end);
              }
              dirty_words += count;
              // Clears bits [low, low + count); every shift stays below kBits.
              bits &= ~BitSet::Word{0} << low << (count - 1) << 1;
            }
          }
          // Forwarding the whole row costs about one union of num_words
          // words, the runs about their length plus kWordsPerRun words each.
          whole_row = dirty_words + runs.size() * kWordsPerRun >= num_words;
        }
        state = kPendingNone;
      }
      const BitSet row = BitSetFromIndex(index);
      // Tail forwarding: a whole row that goes to a single present successor
      // while nothing else is queued makes that successor the next pop, with
      // its whole row pending and its mask untouched. Handling it in place
      // skips the queue and the state round trip; the pops happen in the same
      // order, since that successor would have been the only queued row.
      if (whole_row && worklist.empty()) {
        const HloInstruction* next = only_successor(instruction);
        if (next != nullptr && ABSL_PREDICT_TRUE(is_present(next))) {
          const Key next_key = GetKey(next);
          DCHECK_EQ(tmp_pending_state_[next_key], kPendingNone);
          const Index next_index = indices_[next_key];
          BitSet next_row = BitSetFromIndex(next_index);
          // On a chain the key of the row after next is read right after
          // this union; fetched now so that the union hides the miss on it.
          if (const auto& users = next->users(); users.size() == 1) {
            absl::PrefetchToLocalCache(users.front());
          }
          if (!next_row.OrUpdate(row)) {
            break;
          }
          index = next_index;
          instruction = next;
          in_hand = true;
          continue;
        }
      }
      const auto forward = [&](const HloInstruction* target)
                               ABSL_ATTRIBUTE_ALWAYS_INLINE {
                                 if (whole_row) {
                                   absorb_row(target, row);
                                 } else {
                                   absorb_runs(target, row, runs);
                                 }
                               };
      successors(
          instruction,
          [&](const HloInstruction* successor) ABSL_ATTRIBUTE_ALWAYS_INLINE {
            if (ABSL_PREDICT_TRUE(is_present(successor))) {
              forward(successor);
            } else {
              ForEachPresentThrough(successor, is_present, successors, forward);
            }
          });
      break;
    }
  }
}

}  // namespace xla
