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

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
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

  HloReachabilityMap(const HloReachabilityMap& b) noexcept = delete;
  HloReachabilityMap& operator=(HloReachabilityMap& b) noexcept = delete;

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
  // b = f(f(f(f(f(a), constant), constant), constant), constant).
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
  void SetReachable(Index a, Index b) { BitSetFromIndex(b).Set(a); }

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
  bool IsReachable(Index a, Index b) const { return BitSetFromIndex(b).Get(a); }

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
    // If we cannot construct the key, then the instruction is not in the
    // reachability map.
    return (instruction == nullptr
                ? false
                : (instruction->GetModule() != nullptr
                       ? indices_.contains(GetKey(instruction))
                       : false));
  }

  // Replace the instruction "original" with "replacement" in the reachability
  // map.
  void Replace(const HloInstruction* original,
               const HloInstruction* replacement);

 private:
  // A BitSet is a view over a contiguous region of member that holds a bitset
  // as an array of words.
  class BitSet {
   public:
    using Word = uint64_t;
    static constexpr size_t kBits = 64;

    BitSet() : ptr_(nullptr), bits_(0) {}
    // Create a BitSet view of "num_bits" starting at "ptr".  The memory backing
    // the bit set must be rounded up to the nearest word boundary (for
    // efficiency, we sometimes write full words at the ragged edges of bitsets
    // that are not exactly multiples of kBits in size).
    explicit BitSet(Word* ptr, size_t num_bits) : ptr_(ptr), bits_(num_bits) {}

    // Returns the bit at the given index.
    bool Get(Index index) const {
      DCHECK(index >= 0 && index < bits_);
      return ptr_[index / kBits] & (1ull << (index % kBits));
    }

    // Sets the bit at the given index.
    void Set(Index index) {
      DCHECK(index >= 0 && index < bits_);
      ptr_[index / kBits] |= 1ull << (index % kBits);
    }

    // Sets this bit-set to union of this bit-set and `other`.
    void operator|=(const BitSet& other) {
      DCHECK(bits_ == other.bits_);
      if (ptr_ == other.ptr_) {
        return;
      }

      // Ease the work of the auto-vectorizer.
      const Word* a = ptr_;
      const Word* b = other.ptr_;
      Word* __restrict out = ptr_;
      size_t num_words = NumWords();
      for (size_t i = 0; i < num_words; ++i) {
        out[i] = a[i] | b[i];
      }
    }
    // Copy the bitset contents of "other" into "this".
    void CopyBitSet(const BitSet& other) {
      DCHECK(bits_ == other.bits_);
      if (ptr_ == other.ptr_) {
        return;
      }

      // Ease the work of the auto-vectorizer.
      const Word* b = other.ptr_;
      Word* __restrict out = ptr_;
      size_t num_words = NumWords();
      for (size_t i = 0; i < num_words; ++i) {
        out[i] = b[i];
      }
    }

    size_t NumWords() const { return (bits_ + kBits - 1) / kBits; }
    size_t NumBytes() const {
      return NumWords() * sizeof(Word) / sizeof(uint8_t);
    }

    // Sets the bitvector to all zeros.
    void SetToZero() { memset(ptr_, 0, NumBytes()); }

    bool operator==(const BitSet& other) const {
      if (bits_ != other.bits_) {
        return false;
      }
      absl::Span<Word> aspan(ptr_, NumWords());
      absl::Span<Word> bspan(other.ptr_, other.NumWords());
      return aspan == bspan;
    }
    bool operator!=(const BitSet& other) const { return !(*this == other); }

   private:
    Word* ptr_;
    size_t bits_;  // Number of bits in the set.
  };

  BitSet BitSetFromIndex(Index i) const {
    const int block = i / kRowsPerAllocation;
    const int row_within_block = i % kRowsPerAllocation;
    return BitSet(
        bit_storage_[block].get() + row_within_block * words_per_bitset_,
        bits_per_bitset_);
  }

  friend class HloReachabilityMapBitSetBenchmark;

  using Key = std::pair<int64_t, int64_t>;  // module ID, instruction ID.
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

  // We allocate an (instructions.size() + 1) * (roundup(instructions.size(),
  // 64) bit matrix to hold the adjacency information.  We round up one
  // dimension so that each bit matrix starts on its own word boundary.  We
  // allocate one extra so that we have one bit vector worth of temporary
  // storage for use during the reachability computation.

  // To avoid allocating a single giant block of memory in one allocation, we
  // allocate groups of up to kRowsPerAllocation bitsets at a time.  These
  // groups of rows are stored in the elements of "bit_storage_".
  //
  // BitSetFromIndex(i) abstracts away this representation to give the proper
  // pointer to the "i"th row.
  static constexpr int kRowsPerAllocation = 1024;

  size_t bits_per_bitset_;
  size_t words_per_bitset_;
  size_t total_words_;  // Total allocated words in bit_storage_
  std::vector<std::unique_ptr<BitSet::Word[]>> bit_storage_;

  // A temporary used by SetReachabilityToUnion to avoid an allocation with each
  // call to the method.
  BitSet tmp_bit_set_;
};

}  // namespace xla

#endif  // XLA_HLO_ANALYSIS_HLO_REACHABILITY_H_
