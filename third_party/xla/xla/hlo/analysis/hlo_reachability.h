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

#ifndef XLA_HLO_ANALYSIS_HLO_REACHABILITY_H_
#define XLA_HLO_ANALYSIS_HLO_REACHABILITY_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/types.h"

namespace xla {

// A class for representing reachability between HloInstructions.
//
// It has an adjacency matrix and it is up to the user of the class to set the
// adjacency matrix such that it represents reachability, i.e. such that it is
// transitive. That the graph be transitive is thus not an invariant of this
// class, but it is required for the name of the class and its methods to make
// sense.
class HloReachabilityMap {
 public:
  using Index = size_t;

  // Sets up a graph with no edges and where the nodes correspond to the given
  // instructions.
  explicit HloReachabilityMap(
      absl::Span<const HloInstruction* const> instructions);

  // Computes and returns the reachability between HLO instructions in the
  // computation. The returned HloReachabilityMap is constructed such that
  // HloReachabilityMap::IsReachable(a, b) returns true iff there exists a
  // directed path (from producer to consumer) from 'a' to 'b'. Both data
  // dependencies (operands) and control dependencies are considered for
  // reachability. Trivially an instruction is reachable from itself.
  static std::unique_ptr<HloReachabilityMap> Build(
      const HloComputation* computation);

  // Similar to the above Build operation except that it tries to identify
  // paths between instructions that do not contain control instructions
  // and multiple operands, i.e., b is_reachable a == true iff
  // b = f(f(f(f(f(a), constant), constant), constant).
  // Further, the only ops allowed in a path are basic math operations such
  // as add, sub, mul, div.
  static std::unique_ptr<HloReachabilityMap> BuildWithRestrictions(
      const HloComputation* computation,
      absl::FunctionRef<void(const HloInstruction*,
                             std::vector<HloInstruction*>*)>
          add_dependencies);

  // Set the reachability set of 'instruction' to the union of the reachability
  // sets of 'inputs'. Upon return, IsReachable(x, instruction) where
  // 'x' is not 'instruction' will return true iff IsReachable(x, input) is true
  // for some 'input' in 'inputs'. Also sets 'instruction' to be reachable from
  // itself. Returns whether the reachability set of 'instruction' changed.
  //
  // !!! THIS FUNCTION DOES NOT COMPUTE REACHABILITY !!! It sets the adjacency
  // vector in the internal graph of this HloReachabilityMap for the given
  // instruction and does not transitively update any other part of the
  // adjacency matrix.
  bool SetReachabilityToUnion(absl::Span<const HloInstruction* const> inputs,
                              const HloInstruction* instruction);

  // As above, but faster because it does not check if the reachability changed.
  void FastSetReachabilityToUnion(
      absl::Span<const HloInstruction* const> inputs,
      const HloInstruction* instruction);
  // As above, but use Index instead if it's already looked up which is even
  // faster since no hash map lookup will occur.
  void FastSetReachabilityToUnion(absl::Span<const Index> input_indices,
                                  Index index);

  Index GetIndex(const HloInstruction* instruction) const {
    return indices_.at(GetKey(instruction));
  }

  // Sets entry so that IsReachable(a, b) will return true
  //
  // !!! THIS FUNCTION DOES NOT COMPUTE REACHABILITY !!! It sets the adjacency
  // matrix in the internal graph of this HloReachabilityMap to have an edge
  // from a to b and does not transitively update any other part of the
  // adjacency matrix.
  void SetReachable(const HloInstruction* a, const HloInstruction* b) {
    SetReachable(GetIndex(a), GetIndex(b));
  }
  void SetReachable(Index a, Index b) { bit_sets_[b].Set(a); }

  // Updates the given reachability map after the immediate predecessor set
  // (operands and control predecessors) of 'instruction' has changed.
  void UpdateReachabilityThroughInstruction(const HloInstruction* instruction);

  // Returns true if "b" is reachable from "a"
  //
  // Note that this function only correctly answers queries about reachability
  // if the set of edges that have been provided to this class are transitive.
  bool IsReachable(const HloInstruction* a, const HloInstruction* b) const {
    return IsReachable(GetIndex(a), GetIndex(b));
  }
  bool IsReachable(Index a, Index b) const { return bit_sets_[b].Get(a); }

  // Returns true if "b" is reachable from "a" or "a" is reachable from "b"
  //
  // Note that this function only correctly answers queries about reachability
  // if the set of edges that have been provided to this class are transitive.
  bool IsConnected(const HloInstruction* a, const HloInstruction* b) const {
    return IsConnected(GetIndex(a), GetIndex(b));
  }
  bool IsConnected(Index a, Index b) const {
    return IsReachable(a, b) || IsReachable(b, a);
  }

  // Checks if an instruction is in the Reachability map.
  bool IsPresent(const HloInstruction* instruction) const {
    return indices_.contains(GetKey(instruction));
  }

  // Replace the instruction "original" with "replacement" in the reachability
  // map.
  void Replace(const HloInstruction* original,
               const HloInstruction* replacement);

 private:
  // A dynamically sized bit-set implementation specialized for this use case
  // providing fast bitwise OR (not available in tsl::gtl::BitMap).
  class BitSet {
   public:
    BitSet() = default;
    explicit BitSet(size_t size)
        : size_(size), vector_((size + kBits - 1) / kBits, 0) {}

    // Returns the bit at the given index.
    bool Get(Index index) const {
      DCHECK(index >= 0 && index < size_);
      return vector_[index / kBits] & (1ull << (index % kBits));
    }

    // Sets the bit at the given index.
    void Set(Index index) {
      DCHECK(index >= 0 && index < size_);
      vector_[index / kBits] |= 1ull << (index % kBits);
    }

    // Sets this bit-set to union of this bit-set and `other`.
    void operator|=(const BitSet& other) {
      if (this == &other) return;
      DCHECK(size_ == other.size_);

      // Ease the work of the auto-vectorizer.
      const Word* a = vector_.data();
      const Word* b = other.vector_.data();
      Word* __restrict out = vector_.data();
      size_t num_words = vector_.size();
      for (size_t i = 0; i < num_words; ++i) {
        out[i] = a[i] | b[i];
      }
    }

    // Sets the bitvector to all zeros.
    void SetToZero() { absl::c_fill(vector_, 0); }

    bool operator==(const BitSet& other) const {
      return vector_ == other.vector_;
    }
    bool operator!=(const BitSet& other) const { return !(*this == other); }

   private:
    using Word = uint64_t;
    static constexpr size_t kBits = 64;

    size_t size_;  // Number of bits in the set.
    std::vector<Word> vector_;
  };

  friend class HloReachabilityMapBitSetBenchmark;

  using Key = std::pair<int, int>;  // module ID, instruction ID.
  static Key GetKey(const HloInstruction* instruction) {
    return {instruction->GetModule()->unique_id(), instruction->unique_id()};
  }

  // Helper for SetReachabilityToUnion/FastSetReachabilityToUnion.
  void SetReachabilityToUnionHelper(
      absl::Span<const HloInstruction* const> inputs, Index index);
  void SetReachabilityToUnionHelper(absl::Span<const Index> input_indices,
                                    Index index);

  // Map from instruction to index. The index is used for bit_set_ and the bits
  // within a BitSet.
  absl::flat_hash_map<Key, Index> indices_;

  // Bit-sets holding the reachability to each instruction. The bit-set for
  // instruction X includes ones for each instruction which X is reachable from.
  std::vector<BitSet> bit_sets_;

  // A temporary used by SetReachabilityToUnion to avoid an allocation with each
  // call to the method.
  BitSet tmp_bit_set_;
};

}  // namespace xla

#endif  // XLA_HLO_ANALYSIS_HLO_REACHABILITY_H_
