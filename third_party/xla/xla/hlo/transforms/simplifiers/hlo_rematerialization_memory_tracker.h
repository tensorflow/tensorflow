/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_REMATERIALIZATION_MEMORY_TRACKER_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_REMATERIALIZATION_MEMORY_TRACKER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/tuple_points_to_analysis.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/transforms/simplifiers/hlo_rematerialization_data_structures.h"
#include "xla/hlo/transforms/simplifiers/hlo_rematerialization_options.h"
#include "xla/service/call_graph.h"

namespace xla {

// Return if this is an instruction that relays the buffers it uses to its own
// users and if this is one of these instructions we support the
// rematerialization of.
bool IsSupportedIndirectUser(const HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kBitcast ||
         instruction->opcode() == HloOpcode::kGetTupleElement;
}

class HloRematerializationBufferAnalyzer {
 public:
  // A Buffer represents a single LogicalBuffer in the computation including
  // various metadata useful for tracking liveness of the value. A LogicalBuffer
  // is not used directly because the HLO graph is transformed and
  // TuplePointsToAnalysis which owns all LogicalBuffers cannot be updated after
  // HLO graph transformations.
  struct Buffer {
    // The unique id of this Buffer. This value is equal to the buffer's index
    // in the vector buffers_.
    const BufferId id;

    // The instruction which defines this buffer.
    HloRematItem* defining_instruction;

    // The materialized size of the buffer in bytes.
    const int64_t size;

    // Shape of the buffer.
    Shape shape;

    // Whether this buffer is live-out of the computation.
    bool live_out;

    // Whether this buffer has indirect uses. Ie, an instruction which is not a
    // user of defining_instruction uses this buffer. This can occur due to
    // buffer aliasing (eg, tuples).
    bool has_indirect_uses;

    // Position in the tuple this buffer definition lives in.
    ShapeIndex index;

    // The instructions which use this buffer.
    UsesList users;

    // The number of users (HloInstructions) of this buffer which have not yet
    // been placed in the sequence.
    int64_t unfinished_user_count;

    std::string ToString() const {
      return absl::StrCat("Buffer ", id, " (defined by ",
                          defining_instruction->instruction->name(), ", size ",
                          size, " bytes)");
    }
  };

  static HloRematerializationBufferAnalyzer CreateAnalyzer(
      const HloRematerializationOptions& options,
      const HloComputation* computation,
      const TuplePointsToAnalysis& points_to_analysis,
      const HloRematInstructionList& instruction_list);

  // Return the items which use the given LogicalBuffer. Sets
  // has_indirect_users to whether any of the uses is indirect. A use is
  // indirect if the instruction defining logical_buffer is not an operand of
  // the use. This can happen via buffer aliasing (eg, tuples).
  static UsesList GetUsers(const HloRematInstructionList& instruction_list,
                           const LogicalBuffer* logical_buffer,
                           const TuplePointsToAnalysis& points_to_analysis,
                           bool* has_indirect_users);

  // Returns the number of bytes allocated for the buffer with the given id.
  // Buffers allocated by the calling computation (eg, parameter and output
  // buffers) are considered to have zero bytes because the memory is accounted
  // for in a different computation.
  int64_t AllocatedSize(BufferId buffer_id) const {
    const Buffer& buffer = buffers_.at(buffer_id);
    HloInstruction* inst = buffer.defining_instruction->instruction;
    HloOpcode def_opcode = inst->opcode();
    if (buffer.live_out || def_opcode == HloOpcode::kParameter) {
      return 0;
    } else {
      if (options_.host_memory_offload_config && buffer.shape.has_layout() &&
          buffer.shape.layout().memory_space() ==
              options_.host_memory_offload_config->host_memory_space) {
        // Host memory counts for nothing.
        return 0;
      }
      return buffer.size;
    }
  }

  // Buffers have users and users have buffers used. This function resolves
  // outstanding issues in that bidirectional dependency.
  void ReplaceUsesInUsersOfBuffer(Buffer& buffer, BufferId old_id) const;

  // Gets the compact shape of given hlo instruction. An internal cache is used
  // to avoid computing the shape multiple times.
  absl::StatusOr<const Shape*> GetCompactShape(const HloInstruction* hlo);

  // Creates a Buffer representing the given logical buffer. The buffer is added
  // to buffers_ and a reference is returned.
  Buffer& CreateBufferFromLogicalBuffer(
      const LogicalBuffer* logical_buffer,
      const TuplePointsToAnalysis& points_to_analysis, bool live_out) {
    bool has_indirect_uses = false;
    UsesList users = GetUsers(instruction_list_, logical_buffer,
                              points_to_analysis, &has_indirect_uses);
    return NewBuffer(instruction_list_.GetItem(logical_buffer->instruction()),
                     logical_buffer->shape(), logical_buffer->index(),
                     std::move(users), live_out, has_indirect_uses);
  }

  // Creates a new buffer representing a rematerialization of given buffer for
  // the given uses.
  Buffer& RematerializeBuffer(const Buffer& original_buffer,
                              HloRematItem* remat_item,
                              UsesList&& rematerialized_uses,
                              bool check_placement) {
    if (check_placement) {
      CHECK(original_buffer.defining_instruction->placed)
          << original_buffer.defining_instruction->instruction->name();
    }
    CHECK(!original_buffer.has_indirect_uses) << original_buffer.ToString();
    CHECK(!original_buffer.live_out) << original_buffer.ToString();
    if (check_placement) {
      for (HloRematItemUse& use : rematerialized_uses) {
        CHECK(!use.user->placed) << use.user->instruction->name();
      }
    }
    return NewBuffer(remat_item, original_buffer.shape, original_buffer.index,
                     std::move(rematerialized_uses), /*live_out=*/false,
                     /*has_indirect_uses=*/false);
  }

  // Creates a new buffer, adds it to buffers_, and returns a reference.
  Buffer& NewBuffer(HloRematItem* defining_instruction, const Shape& shape,
                    const ShapeIndex& index, UsesList&& uses, bool live_out,
                    bool has_indirect_uses) {
    int buffer_id = buffers_.size();
    auto get_num_of_unique_users = [](const UsesList& uses) -> int64_t {
      absl::flat_hash_set<HloRematItem*> users_set;
      for (const HloRematItemUse& use : uses) {
        users_set.insert(use.user);
      }
      return users_set.size();
    };
    buffers_.push_back(Buffer{buffer_id, defining_instruction,
                              options_.hlo_cost_analysis.GetShapeSize(shape),
                              shape, live_out, has_indirect_uses, index, uses,
                              get_num_of_unique_users(uses)});
    return buffers_.back();
  }

  const HloRematerializationOptions& options_;
  const HloRematInstructionList& instruction_list_;
  std::vector<Buffer> buffers_;

  // A map that caches existing known compact shape for each instruction.
  absl::flat_hash_map<const HloInstruction*, Shape> compact_shape_;

 private:
  HloRematerializationBufferAnalyzer(
      const HloRematerializationOptions& options,
      const HloRematInstructionList& instruction_list)
      : options_(options), instruction_list_(instruction_list) {}
};

struct RematStrategy {
  enum {
    // Recompute the node at a later program point.
    kRecompute,
    // Change the layout into a compact form and uncompress it back at a later
    // program point.
    kCompress,
    // Copy the data off the device to the host to be copied back later.
    kHostOffload,
  } kind;
  Shape compact_shape;
};

// Interface for memory trackers, which track the memory usage of a computation.
// Memory usage is the sum of the sizes of live values (LogicalBuffers) at the
// current point in the instruction sequence. Different memory trackers offer
// different ways to adjust the current point in the instruction sequence, and
// the shared interface methods are relative to this point.
class HloRematerializationMemoryTracker {
 public:
  // Denotes that `new_item` should be inserted after `predecessor`.
  struct NewItemAndPredecessor {
    HloRematItem* new_item;
    HloRematItem* predecessor;
  };
  // Denotes that `new_item` should be inserted before `successor`.
  struct NewItemAndSuccessor {
    HloRematItem* new_item;
    HloRematItem* successor;
  };

  HloRematerializationMemoryTracker(const HloRematerializationOptions& options,
                                    const HloComputation* computation)
      : options_(options), computation_(computation) {}
  virtual ~HloRematerializationMemoryTracker() = default;

  bool IsInstructionPlaced(const HloInstruction* instruction) const;

  // Returns whether 'item' is currently in progress.
  bool IsInProgressItem(HloRematItem* item) const;

  // Counts the bytes that this item occupies by summing up the buffers defined
  // by this item. If only_count_unplaced_users is true, only count users of
  // buffers which are not yet placed. This will represent the current memory
  // usage of the item. Otherwise, count all buffers. This will represent the
  // peak memory usage of the item.
  int64_t BytesUsedByBuffers(const HloRematItem* item,
                             bool only_count_unplaced_users) const;

  // Given a list of uses return two lists where one is the ones which are
  // placed and the other is ones which are not yet placed.
  std::tuple<UsesList, UsesList> GetPlacedAndUnplacedUsers(
      const UsesList& uses) const;

  // The next six methods are used to update the memory tracker to account for
  // the recomputation, compression, or offloading of `original_item`. The
  // methods are split up into pairs: an
  // Add*Instruction(s)ToBufferCalculations() method which should be called
  // first and updates buffer relationships maintained by the tracker. These
  // relationships may be useful for a caller that needs to determine exactly
  // where to insert the new instructions so that in can appropriately invoke
  // Add*Instruction(s)ToInstructionOrdering() determining exactly where to
  // insert the new instructions.

  // See above comment about how to call this pair of methods. For the first
  // method, all remaining unplaced uses of `original_item` are relegated to
  // `recompute_item`. This method should be called after the HLO graph has
  // been transformed (recomputation instruction created and connected
  // to uses).
  virtual absl::Status AddRecomputeInstructionToBufferCalculations(
      HloRematItem* original_item, HloRematItem* recompute_item,
      absl::Span<HloRematItem*> indirect_users) = 0;

  virtual absl::Status AddRecomputeInstructionToInstructionOrdering(
      HloRematItem* original_item, NewItemAndSuccessor recompute,
      absl::Span<HloRematItem*> indirect_users) = 0;

  // All remaining unplaced uses of `original_item` are relegated to
  // `uncompressed_item`. The caller is responsible for marking the compressed
  // instruction as placed and leaving the uncompressed instruction as unplaced.
  virtual absl::Status AddCompressInstructionsToBufferCalculations(
      HloRematItem* original_item, HloRematItem* compressed_item,
      HloRematItem* uncompressed) = 0;

  virtual absl::Status AddCompressInstructionsToInstructionOrdering(
      HloRematItem* original_item, NewItemAndPredecessor compressed,
      NewItemAndSuccessor uncompressed) = 0;

  // Given the newly created instructions for host memory offload, create new
  // buffers, link their uses to their users, and update the current memory
  // usage.
  virtual absl::Status AddOffloadInstructionsToBufferCalculations(
      HloRematItem* original_item, HloRematItem* copy_start_to_host_item,
      HloRematItem* copy_done_to_host_item,
      HloRematItem* copy_start_to_device_item,
      HloRematItem* copy_done_to_device_item) = 0;

  virtual absl::Status AddOffloadInstructionsToInstructionOrdering(
      HloRematItem* original_item, NewItemAndPredecessor copy_start_to_host,
      NewItemAndPredecessor copy_done_to_host,
      NewItemAndSuccessor copy_start_to_device,
      NewItemAndSuccessor copy_done_to_device) = 0;

  // Returns the list of uses for a specific 'item'.
  UsesList GetItemUses(HloRematItem* item) const;

  // Selects and returns the best candidate instructions for rematerialization.
  // A sequence of candidate instructions of length between min_block_size and
  // max_block_size (both inclusive) with the lowest rematerialization cost is
  // selected among those candidates which reduce memory use at the program
  // point of the current instruction as indicated by memory_tracker. Returns an
  // empty vector if no candidates are found. Also returns an integer that
  // represents the amount of "effort" expended to find the candidate
  // instructions.
  std::tuple<std::vector<HloRematItem*>, RematStrategy, int>
  PickRematerializationCandidates(
      const HloRematInstructionList& instruction_list,
      int64_t memory_limit_bytes,
      absl::flat_hash_map<const HloInstruction*, bool>* rematerializable_map,
      int min_block_size, int max_block_size, int64_t peak_memory_bytes);

  // Returns a block of up to min_block_size consecutive candidate instructions
  // from instruction_list starting from start_item. Returns fewer than
  // min_block_size instructions if the block of unplaced instructions starting
  // from start_item is smaller than min_block_size.
  std::vector<HloRematItem*> GetInitialBlock(
      const HloRematInstructionList& instruction_list, HloRematItem* start_item,
      int min_block_size) const;

  // Returns whether 'item' has any unplaced users.
  bool HasUnplacedUsers(HloRematItem* item) const;

  // Whether it is possible to recompute/compress/offload the given sequence of
  // instructions or given instruction at the current point.
  bool IsCurrentlyRecomputable(
      absl::Span<const HloRematItem* const> items) const;
  bool IsCurrentlyCompressible(const HloRematItem* item) const;
  bool IsCurrentlyOffloadable(const HloRematItem* item) const;

  // The number of bytes that the current memory usage will be reduced by if the
  // given sequence of instructions or given instruction is
  // recomputed/compressed/offloaded.
  int64_t MemoryReducedIfRecomputed(
      absl::Span<const HloRematItem* const> items) const;
  int64_t MemoryReducedIfCompressed(const HloRematItem* item,
                                    const Shape& compact_shape) const;
  int64_t MemoryReducedIfOffloaded(const HloRematItem* item) const;

  // Calculates the cost of recomputing/compressing/offloading the
  // candidate_item(s).
  std::optional<int64_t> GetCostOfRecompute(
      absl::Span<const HloRematItem* const> candidate_items,
      int64_t memory_limit_bytes) const;
  std::optional<int64_t> GetCostOfCompression(
      const HloRematItem* candidate_item, int64_t memory_limit_bytes,
      int64_t peak_memory_bytes);
  std::optional<int64_t> GetCostOfHostOffload(
      const HloRematItem* candidate_item, int64_t memory_limit_bytes) const;

  // Returns the total allocated size of all buffers defined by `item`.
  int64_t AllocatedSize(HloRematItem* item) const;

  const HloRematerializationOptions& options() const { return options_; }
  const HloComputation* computation() const { return computation_; }

  virtual std::string ToString() const = 0;

 protected:
  // Subclasses hold shared functionality using a private
  // HloRematerializationBufferAnalyzer member variable. These protected
  // accessors enable some public functions to be implemented directly.
  virtual HloRematerializationBufferAnalyzer* buffer_analyzer() = 0;
  virtual const HloRematerializationBufferAnalyzer* buffer_analyzer() const = 0;

  // Forwarding this from private HloRematInstructionList member variable.
  virtual HloRematItem* GetItem(HloInstruction* instruction) = 0;
  virtual const HloRematItem* GetItem(
      const HloInstruction* instruction) const = 0;

  virtual bool IsItemPlaced(const HloRematItem* item) const = 0;
  virtual bool IsItemFinished(const HloRematItem* item) const = 0;
  bool IsInstructionCurrentlyLive(const HloRematItem* item) const;

  virtual bool IsBufferLive(BufferId buffer_id) const = 0;

  virtual HloRematItem* in_progress_item() const = 0;

  // Returns whether the given buffer is being used by the in-progress
  // instruction.
  virtual bool IsBufferInUse(BufferId buffer_id) const;

 private:
  using Buffer = HloRematerializationBufferAnalyzer::Buffer;

  // Whether any user of any item in `items` has been placed at the current
  // point.
  bool NoUserPlaced(absl::Span<const HloRematItem* const> items) const;

  const HloRematerializationOptions& options_;
  const HloComputation* computation_;
};

// Memory tracker which allows for the sequential placement of instructions. The
// state of the tracker can be advanced through the instruction list via
// BeginInstruction() and EndInstruction().
class HloRematerializationSweepMemoryTracker
    : public HloRematerializationMemoryTracker {
 public:
  static std::unique_ptr<HloRematerializationSweepMemoryTracker> CreateTracker(
      const HloRematerializationOptions& options,
      const HloComputation* computation,
      const TuplePointsToAnalysis& points_to_analysis,
      const HloRematInstructionList& instruction_list);

  virtual ~HloRematerializationSweepMemoryTracker() = default;

  // Starts the placement of the given instruction. This adds the sizes of the
  // LogicalBuffers defined by the instruction to the current memory
  // usage. Placement is broken into two steps (BeginInstruction and
  // EndInstruction) to accurately model memory usage. At BeginInstruction the
  // memory for the output value(s) of the current instruction is allocated. At
  // EndInstruction memory for dead operand(s) is freed.
  absl::Status BeginInstruction(HloRematItem* item);

  // Finishes the placement of the current instruction. This frees any dead
  // operands or dead result of the instruction. This must be called after
  // each call to BeginInstruction.
  absl::Status EndInstruction();

  absl::Status AddRecomputeInstructionToBufferCalculations(
      HloRematItem* original_item, HloRematItem* recompute_item,
      absl::Span<HloRematItem*> indirect_users) final;

  // Sweep memory tracker does not maintain its own instruction ordering so this
  // does nothing.
  absl::Status AddRecomputeInstructionToInstructionOrdering(
      HloRematItem* original_item, NewItemAndSuccessor recompute,
      absl::Span<HloRematItem*> indirect_users) final {
    return absl::OkStatus();
  }

  absl::Status AddCompressInstructionsToBufferCalculations(
      HloRematItem* original_item, HloRematItem* compressed_item,
      HloRematItem* uncompressed) final;

  // Sweep memory tracker does not maintain its own instruction ordering so this
  // does nothing.
  absl::Status AddCompressInstructionsToInstructionOrdering(
      HloRematItem* original_item, NewItemAndPredecessor compressed,
      NewItemAndSuccessor uncompressed) final {
    return absl::OkStatus();
  }

  absl::Status AddOffloadInstructionsToBufferCalculations(
      HloRematItem* original_item, HloRematItem* copy_start_to_host_item,
      HloRematItem* copy_done_to_host_item,
      HloRematItem* copy_start_to_device_item,
      HloRematItem* copy_done_to_device_item) final;

  // Sweep memory tracker does not maintain its own instruction ordering so this
  // does nothing.
  absl::Status AddOffloadInstructionsToInstructionOrdering(
      HloRematItem* original_item, NewItemAndPredecessor copy_start_to_host,
      NewItemAndPredecessor copy_done_to_host,
      NewItemAndSuccessor copy_start_to_device,
      NewItemAndSuccessor copy_done_to_device) final {
    return absl::OkStatus();
  }

  // Returns the current memory usage. This is the sum of sizes of all live
  // values.
  int64_t memory_usage() const { return memory_usage_; }

  // Checks invariants of the data structure. This is expensive to call.
  bool Check() const;

  std::string ToString() const final;

 protected:
  HloRematerializationBufferAnalyzer* buffer_analyzer() final {
    return buffer_analyzer_.get();
  }
  const HloRematerializationBufferAnalyzer* buffer_analyzer() const final {
    return buffer_analyzer_.get();
  }

  HloRematItem* GetItem(HloInstruction* instruction) final {
    return instruction_list_.GetItem(instruction);
  }
  const HloRematItem* GetItem(const HloInstruction* instruction) const final {
    return instruction_list_.GetItem(instruction);
  }

  bool IsItemPlaced(const HloRematItem* item) const final;
  bool IsItemFinished(const HloRematItem* item) const final;

  bool IsBufferLive(BufferId buffer_id) const final;

  HloRematItem* in_progress_item() const final { return in_progress_item_; }

 private:
  using Buffer = HloRematerializationBufferAnalyzer::Buffer;

  HloRematerializationSweepMemoryTracker(
      const HloRematerializationOptions& options,
      const HloComputation* computation,
      const HloRematInstructionList& instruction_list,
      HloRematerializationBufferAnalyzer buffer_analyzer)
      : HloRematerializationMemoryTracker(options, computation),
        instruction_list_(instruction_list),
        buffer_analyzer_(std::make_unique<HloRematerializationBufferAnalyzer>(
            std::move(buffer_analyzer))) {}

  // Adjusts our tracked memory usage as a result of this new item coming into
  // scope.
  void CountAllocatedMemory(HloRematItem* item);

  // Adjusts our tracked memory usage as a result of this item going out of
  // scope.
  absl::Status CountFreedMemory(HloRematItem* item);

  // Instruction list containing the ordering of instructions in
  // computation_. This is the order in which instructions are placed
  // (BeginInstruction/EndInstruction calls).
  const HloRematInstructionList& instruction_list_;

  // Memory usage at the currently placed instruction.
  int64_t memory_usage_ = 0;

  // The instruction currently being placed. This value is non-null only
  // between the calling of BeginInstruction and EndInstruction.
  HloRematItem* in_progress_item_ = nullptr;

  std::unique_ptr<HloRematerializationBufferAnalyzer> buffer_analyzer_;
};

// Memory tracker which uses balanced tree data structures to allow for
// efficient random access to instructions. The state of the tracker can be
// moved around in the instruction list via JumpToInstruction(). Also offers
// additional methods to query for the peak memory usage and the instruction
// during which it occurs. Slightly more expensive to call than
// HloRematerializationSweepMemoryTracker (extra factor is logarithmic in the
// total number of instructions).
class HloRematerializationPeakMemoryTracker
    : public HloRematerializationMemoryTracker {
 public:
  static absl::StatusOr<std::unique_ptr<HloRematerializationPeakMemoryTracker>>
  CreateTracker(const HloRematerializationOptions& options,
                const HloComputation* computation,
                const TuplePointsToAnalysis& points_to_analysis,
                const HloRematInstructionList& instruction_list,
                const CallGraph* call_graph,
                const absl::flat_hash_set<absl::string_view>& execution_threads,
                const absl::flat_hash_map<
                    const HloComputation*,
                    std::unique_ptr<HloRematerializationPeakMemoryTracker>>&
                    computation_peak_memory_tracker);

  // Jumps to during the placement of the given instruction, when the memory for
  // the output value(s) of the instruction have been allocated but the memory
  // for dead operand(s) has not yet been freed.
  absl::Status JumpToInstruction(const HloInstruction* instruction);

  absl::StatusOr<MemoryUsageAndInstruction>
  ComputePeakMemoryUsageAndInstruction() const;

  absl::Status AddRecomputeInstructionToBufferCalculations(
      HloRematItem* original_item, HloRematItem* recompute_item,
      absl::Span<HloRematItem*> indirect_users) final;

  absl::Status AddRecomputeInstructionToInstructionOrdering(
      HloRematItem* original_item, NewItemAndSuccessor recompute,
      absl::Span<HloRematItem*> indirect_users) final;

  absl::Status AddCompressInstructionsToBufferCalculations(
      HloRematItem* original_item, HloRematItem* compressed_item,
      HloRematItem* uncompressed) final;

  absl::Status AddCompressInstructionsToInstructionOrdering(
      HloRematItem* original_item, NewItemAndPredecessor compressed,
      NewItemAndSuccessor uncompressed) final;

  absl::Status AddOffloadInstructionsToBufferCalculations(
      HloRematItem* original_item, HloRematItem* copy_start_to_host_item,
      HloRematItem* copy_done_to_host_item,
      HloRematItem* copy_start_to_device_item,
      HloRematItem* copy_done_to_device_item) final;

  absl::Status AddOffloadInstructionsToInstructionOrdering(
      HloRematItem* original_item, NewItemAndPredecessor copy_start_to_host,
      NewItemAndPredecessor copy_done_to_host,
      NewItemAndSuccessor copy_start_to_device,
      NewItemAndSuccessor copy_done_to_device) final;

  // Returns the current memory usage. This is the sum of sizes of all live
  // values. This method is not const because it depends on querying a lazy data
  // structure.
  absl::StatusOr<int64_t> GetMemoryUsage();

  // Returns the current memory usage but excludes called computations (this is
  // what SweepMemoryTracker would report since it does not keep track of called
  // computations).
  absl::StatusOr<int64_t> GetMemoryUsageWithoutCalledComputations();

  absl::Status CalleeComputationWasUpdated(
      const HloComputation* callee_computation);

  // Checks invariants of the data structure. This is expensive to call.
  bool Check() const;

  std::string ToString() const final;

 protected:
  HloRematerializationBufferAnalyzer* buffer_analyzer() final {
    return buffer_analyzer_.get();
  }
  const HloRematerializationBufferAnalyzer* buffer_analyzer() const final {
    return buffer_analyzer_.get();
  }

  HloRematItem* GetItem(HloInstruction* instruction) final {
    return instruction_list_.GetItem(instruction);
  }
  const HloRematItem* GetItem(const HloInstruction* instruction) const final {
    return instruction_list_.GetItem(instruction);
  }

  bool IsItemPlaced(const HloRematItem* item) const final;
  bool IsItemFinished(const HloRematItem* item) const final;

  bool IsBufferLive(BufferId buffer_id) const final;

  HloRematItem* in_progress_item() const final { return in_progress_item_; }

 private:
  using Buffer = HloRematerializationBufferAnalyzer::Buffer;

  HloRematerializationPeakMemoryTracker(
      const HloRematerializationOptions& options,
      const HloComputation* computation,
      const HloRematInstructionList& instruction_list,
      const CallGraph* call_graph,
      const absl::flat_hash_set<absl::string_view>& execution_threads,
      const absl::flat_hash_map<
          const HloComputation*,
          std::unique_ptr<HloRematerializationPeakMemoryTracker>>&
          computation_peak_memory_tracker,
      HloRematerializationBufferAnalyzer buffer_analyzer)
      : HloRematerializationMemoryTracker(options, computation),
        instruction_list_(instruction_list),
        call_graph_(call_graph),
        execution_threads_(execution_threads),
        computation_peak_memory_tracker_(computation_peak_memory_tracker),
        buffer_analyzer_(std::make_unique<HloRematerializationBufferAnalyzer>(
            std::move(buffer_analyzer))) {}

  // A set of helper initialization methods for CreateTracker() factory
  // function.
  // Must be called after `instruction_list_` and `buffer_analyzer_` are
  // initialized.
  absl::Status InitializeSegmentTree();
  // Must be called after `instruction_list_` is initialized.
  absl::Status InitializeSleatorDietzOrderMaintenance();
  // Must be called after `buffer_analyzer_` and `instruction_ordering_` are
  // initialized.
  absl::Status InitializeFinalUser();

  absl::Status UpdateFinalUser(BufferId buffer_id, const UsesList& uses);
  // These two methods insert into `instruction_ordering_` and `segtree_` at the
  // same time. The memory usage for `segtree_` is determined by
  // GetMemoryUsageAfter()/GetMemoryUsageBefore() applied to the
  // predecessor/successor, respectively. This is a useful starting amount which
  // would be correct if the new item was a no-op.
  absl::Status InsertInstructionAfterPredecessor(
      NewItemAndPredecessor new_item_and_predecessor);
  absl::Status InsertInstructionBeforeSuccessor(
      NewItemAndSuccessor new_item_and_successor);

  absl::StatusOr<int64_t> CalledComputationsMemoryUsage(
      const HloInstruction* instruction);

  // The memory usage during a particular instruction, as if BeginInstruction()
  // was called but EndInstruction() was not.
  absl::StatusOr<int64_t> GetMemoryUsageDuring(
      const HloInstruction* instruction);
  // The memory usage right after a particular instruction, as if both
  // BeginInstruction() and EndInstruction() were called.
  absl::StatusOr<int64_t> GetMemoryUsageAfter(
      const HloInstruction* instruction);
  // The memory usage right before a particular instruction, as if neither
  // BeginInstruction() nor EndInstruction() were called.
  absl::StatusOr<int64_t> GetMemoryUsageBefore(
      const HloInstruction* instruction);

  // Instruction list containing the ordering of instructions in
  // computation_. This is the order the instructions are placed (used by
  // segtree_).
  const HloRematInstructionList& instruction_list_;

  const CallGraph* call_graph_;
  const absl::flat_hash_set<absl::string_view>& execution_threads_;

  // A map that caches peak memory usage of called computations. Updated via
  // calls to CalleeComputationWasUpdated().

  // In order to correctly account for callees when computing the peak memory
  // usage, we are given reference access to a map from other computations to
  // their own memory trackers. We are informed about changes to this map via
  // CalleeComputationWasUpdated().
  const absl::flat_hash_map<
      const HloComputation*,
      std::unique_ptr<HloRematerializationPeakMemoryTracker>>&
      computation_peak_memory_tracker_;
  // The peak memory usage of each computation according to the last time we
  // queried its memory tracker (so we can compute deltas).
  absl::flat_hash_map<const HloComputation*, int64_t>
      last_computation_peak_memory_usage_;
  // The calling instruction of each computation so we can quickly update.
  absl::flat_hash_map<const HloComputation*, const HloInstruction*>
      computation_to_calling_instruction_;

  std::unique_ptr<HloRematerializationBufferAnalyzer> buffer_analyzer_;
  std::unique_ptr<AVLLazySegmentTree> segtree_;
  SleatorDietzOrderMaintenance instruction_ordering_;
  // PeakMemoryTracker maintains this map in lieu of
  // `Buffer.unfinished_user_count`.
  absl::flat_hash_map<BufferId, HloRematItem*> final_user_;

  // The current instruction (i.e. the last instruction jumped to by
  // JumpToInstruction()). Initially null, then nonnull after the first call to
  // JumpToInstruction().
  HloRematItem* in_progress_item_ = nullptr;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_REMATERIALIZATION_MEMORY_TRACKER_H_
