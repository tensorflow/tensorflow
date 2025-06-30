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

#ifndef XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_OPTIONS_H_
#define XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_OPTIONS_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/layout.h"
#include "xla/service/buffer_value.h"
#include "xla/service/hlo_value.h"
#include "xla/service/memory_space_assignment/allocation_value.h"
#include "xla/service/memory_space_assignment/buffer_interval_comparator.h"
#include "xla/service/memory_space_assignment/cost_analysis.h"
#include "xla/service/memory_space_assignment/memory_space_assignment.pb.h"
#include "xla/service/memory_space_assignment/prefetch_interval_picker.h"
#include "xla/service/memory_space_assignment/repacking.h"
#include "xla/service/memory_space_assignment/slice.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla {
namespace memory_space_assignment {

using IsAllowedInAlternateMemoryFunction = std::function<bool(const HloValue&)>;
using IsUseAllowedInAlternateMemoryFunction =
    std::function<bool(const HloUse&)>;
using IsPositionAllowedInAlternateMemoryFunction =
    std::function<bool(const HloPosition&)>;
using ReservedScopedMemoryFunction = std::function<int64_t(
    const HloInstruction*,
    const absl::flat_hash_set<
        std::pair<int, ShapeIndex>>& /*operands_in_alternate_memory*/,
    const absl::flat_hash_set<ShapeIndex>& /*outputs_in_alternate_memory*/)>;
using PositionRequiresContiguousAllocationFunction =
    std::function<bool(const HloPosition&)>;
using WindowPrefetchDetailFunction =
    std::function<WindowPrefetchDetail(const HloInstruction*)>;
using WindowPrefetchNotifyOperandAppendedFunction =
    std::function<void(HloInstruction*, int64_t, int64_t)>;
using IsAsyncSliceImplementedFunction =
    std::function<bool(const HloInstruction*)>;
using InitSplitTreeFn = std::function<ShapeTree<int64_t>(
    const HloInstruction*,
    absl::flat_hash_map<const HloInstruction*, ShapeTree<int64_t>>*)>;
using DetermineSplitDimensionFunction =
    std::function<std::optional<SplitConfig>(
        const HloValue&,
        absl::flat_hash_map<const HloInstruction*, ShapeTree<int64_t>>*)>;
using BitcastSplitFn = std::function<absl::StatusOr<int64_t>(
    const HloInstruction* instruction, int64_t split_dim)>;
using ShapeSizeFn = std::function<int64_t(const Shape&)>;
using HloPositionOrUse = std::variant<HloPosition, HloUse>;

// MSA allows for custom post-allocation transformations. When a post-allocation
// transformation is performed on an instruction, this result is returned. It
// tells MSA:
//  1. A list of instructions that MSA should delete.
//  2. A list of HloUses that the transformation replaced.
//
// This information is then processed via
// FixAllocationSequenceAfterPostAllocationTransformation call.
struct PostAllocationTransformationUpdate {
  std::vector<HloInstruction*> to_be_removed;
  absl::flat_hash_map<HloUse, HloUse> update_use_map;

  std::string ToString() const;
};

// The different modes for window prefetch. kWindowExposure is currently the
// default mode, where the window buffer is exposed from the reserved scoped
// memory. kWindowPrefetch is a mode where the window buffer is not only exposed
// from the reserved scoped memory, but also has the content prefetched into
// alternate memory.
enum class WindowPrefetchMode {
  kWindowExposure,
  kWindowPrefetch,
};

// A struct to specify the memory space coloring of a buffer position or use.
struct BufferColoring {
  HloPositionOrUse buffer_position_or_use;  // Buffer position or use to color.
  int64_t memory_space;                     // How to color the buffer.
};

// The different options to be passed to the Run() API.
struct Options {
  // The backend-specific integer value that describes the default memory.
  int64_t default_memory_space = 0;

  // Backend-specific integer value that describes the alternate memory.
  int64_t alternate_memory_space = 0;

  // Maximum size of the alternate memory space.
  int64_t max_size_in_bytes = 0;

  // Memory alignment of the alternate memory space.
  int64_t alignment_in_bytes = 1;

  // If provided, we sort the buffers using this comparator. Otherwise, we use
  // GlobalDecreasingSizeBestFitHeap::kSpatial.
  BufferIntervalComparator* buffer_interval_comparator = nullptr;

  // This object determines how early and how late prefetches can occur.
  PrefetchIntervalPicker* prefetch_interval_picker = nullptr;

  // This object is used to determine the benefit of a particular allocation.
  CostAnalysis* cost_analysis = nullptr;

  // Size function for buffer values.
  BufferValue::SizeFunction size_fn;

  ShapeSizeFn shape_size_fn;

  std::function<Shape(const Shape&)> get_equivalent_s8_shape_fn;

  // This function can be used to prevent certain HloValues (e.g., based on
  // the opcode) to be placed on the alternate memory.
  IsAllowedInAlternateMemoryFunction is_allowed_in_alternate_mem_fn;

  // This function can be used to prevent certain HloUses (e.g., based on
  // the opcode) to be placed on the alternate memory.
  IsUseAllowedInAlternateMemoryFunction is_use_allowed_in_alternate_mem_fn =
      [](const HloUse&) { return true; };

  // Specifies if the given position is allowed in the alternate memory.
  IsPositionAllowedInAlternateMemoryFunction
      is_position_allowed_in_alternate_mem_fn =
          [](const HloPosition&) { return true; };

  // This function returns the amount of scoped memory in bytes that should be
  // reserved during the execution of this instruction. Note that the
  // `operands_in_alternate_memory` also includes the window prefetched
  // operands.
  ReservedScopedMemoryFunction reserved_scoped_memory_fn =
      [](const HloInstruction*,
         const absl::flat_hash_set<
             std::pair<int, ShapeIndex>>& /*operands_in_alternate_memory*/,
         const absl::flat_hash_set<
             ShapeIndex>& /*outputs_in_alternate_memory*/) { return 0; };

  PositionRequiresContiguousAllocationFunction
      position_requires_contiguous_allocation_fn =
          [](const HloPosition&) { return false; };

  // This function is called to get details about window prefetches.
  WindowPrefetchDetailFunction window_prefetch_detail_fn =
      [](const HloInstruction*) { return WindowPrefetchDetail(); };

  // This function is called to notify that an operand has been appended as a
  // window prefetch buffer.
  WindowPrefetchNotifyOperandAppendedFunction notify_operand_appended_fn =
      [](HloInstruction*, int64_t, int64_t) {};

  // This function can be used to check if an equivalent asynchronous slice
  // lowering is implemented for a given  synchronous slice instruction.
  IsAsyncSliceImplementedFunction is_async_slice_implemented_fn =
      [](const HloInstruction*) { return false; };

  // Should only be used for testing purposes. This function allows us to
  // modify the AllocationResult after the AllocationRequest has been processed
  // by AllocateSegment().
  std::function<void(const AllocationRequest&, AllocationResult&)>
      allocation_result_modifier_testing_fn = nullptr;

  // Should only be used for testing purposes. This function allows us to
  // modify the AllocationRequest before the AllocationRequest is passed to
  // AllocateSegment().
  std::function<void(AllocationRequest&)>
      allocation_request_modifier_testing_fn = nullptr;

  // This function chooses a dimension to split the given HloValue on. Splitting
  // will be disabled if this function is not provided.
  DetermineSplitDimensionFunction determine_split_dimension_fn = nullptr;

  // This function sets up a split tree, based on an instruction's shape, with
  // kAny as the default. Splitting will be disabled if this function is not
  // provided.
  InitSplitTreeFn init_split_tree_fn = nullptr;

  // Determines the appropriate output split for a bitcast given an input split.
  // Splitting will be disabled if this function is not provided.
  BitcastSplitFn bitcast_split_fn = nullptr;

  // Dimension number indicating no split is present.
  int64_t replicated_split_dimension = -1;

  // Dimension number indicating any split is allowable.
  int64_t any_split_dimension = -2;

  // Applies post-allocation transformations to the given instruction. This
  // function is called after the allocations are found in the MsaAlgorithm. It
  // is called on each instruction I that meets the following conditions:
  // 1. I is called from a non-fusion computation
  // 2. I's operands are not in alternate memory
  // 3. I is not successfully converted to async instruction.
  // 4. I's operands don't have in-place users, e.g., a dynamic-update-slice.
  //
  // The transformation function is allowed to do the following:
  //  1. Mark instructions for removal.
  //  2. Modify existing instructions.
  //
  // This transformation is NOT allowed to:
  //  1. Directly remove instructions (or nullify them).
  //  2. Add new instructions.
  //
  // Note that it is up to the transformation function to ensure that the
  // changes to the module preserves the semantics of the original program.
  std::function<absl::StatusOr<PostAllocationTransformationUpdate>(
      HloInstruction*)>
      post_allocation_transformation_fn;

  // If true, we will try to reduce scoped allocation buffer size for all
  // instructions if their operand/output has been allocated in alternate
  // memory.
  bool reduce_scoped_memory_limit = false;

  // If true, we allocate the reserved scoped memory at the same offset. This
  // is useful to enable more deduplication between HLOs that have reserved
  // scoped memories, but may result in less efficient memory packing.
  bool allocate_reserved_scoped_memory_at_same_offset = true;

  // Specifies the upper bound for number of outstanding prefetches and
  // evictions, -1 for unlimited.
  int64_t max_outstanding_prefetches = -1;
  int64_t max_outstanding_evictions = -1;

  // Extra outstanding prefetch limit for while uses (in addition to
  // max_outstanding_prefetches).
  int64_t while_use_extra_outstanding_prefetch_limit = 0;

  // Specifies the maximum number of retries that will be performed for each
  // value in case prefetching failed due to running out of asynchronous
  // copies or asynchronous copy resource.
  int64_t max_retries = 1;

  // The maximum number of repacks that we are willing to perform in case we
  // can't allocate a buffer due to running out of memory. If this value is
  // greater than 0, repacker must be non-nullptr.
  int64_t max_repacks = 0;

  // The repacking algorithm to reduce fragmentation. Must be non-null if
  // max_repacks is greater than 0.
  MemorySpaceAssignmentRepacker* repacker = nullptr;

  // This is only useful for testing, repack after every allocation.
  bool repack_after_every_allocation = false;

  // If true, verifies the memory space assignment against overlapping
  // buffers.
  bool verify = false;

  // If not nullptr, this function is called to dump debugging information.
  // The first argument is appended to the file name and the second argument
  // is the contents of the file.
  std::function<void(absl::string_view, absl::string_view)> dump_fn = nullptr;

  // Enable prefetching buffers into preferred memory across program
  // boundaries
  bool enable_cross_program_prefetch = true;

  // If true, use buffer_interval_compare to determine which buffers to
  // prefetch across program boundaries.
  bool default_cross_program_prefetch_heuristic = false;

  // Enable cross-program prefetch freeing optimization where the
  // cross-program-prefetched buffer can be reused.
  bool enable_cross_program_prefetch_freeing = true;

  // The maximum number of cross program prefetches.
  // TODO(tjablin): Use a heuristic to determine this automatically.
  int max_cross_program_prefetches = 1;

  // If false, we assume tensors that we couldn't explicitly determine to be
  // activations are activations. If true, we assume these aren't activations,
  // so they may be cross-program-prefetch candidates.
  bool cross_program_prefetch_permissive_mode = false;

  // Enable redundant eviction optimization in/around while loops. If enabled,
  // this optimization would keep a copy of the buffer in the default memory in
  // addition to alternate memory to eliminate redundant evictions.
  bool enable_while_redundant_eviction_elimination = true;

  // An optional memory space assignment autotuning config, which is used
  // to sort allocated buffers.
  std::optional<std::vector<uint64_t>> autotuning_config = std::nullopt;

  // If true, uses the earlier instance of the same instruction to use as
  // preferred prefetch start time.
  bool use_repeated_instance_for_preferred_prefetch_time = false;

  // If true, enforces the FIFO order for prefetches.
  bool enforce_prefetch_fifo_order = false;

  // If true, tries to replace synchronous copy instructions with asynchronous
  // ones. If it fails to replace the copy, it keeps the sync version.
  bool enable_sync_copy_replacement = false;

  // If true, tries to replace synchronous slice instructions with asynchronous
  // ones. If it fails to replace the slice, it keeps the sync version.
  bool enable_sync_slice_replacement = false;

  // If non-zero, this is the number of extra outstanding async copies that we
  // allow for each sync mem op that is converted to an async mem op.
  int extend_async_copies_limit_for_sync_mem_op_conversion = 0;

  // The ratio of use bytes to copy bytes for a given allocation site below
  // which we consider the site to be inefficient. A value of 0 would treat all
  // sites as efficient and a value of 1 would require the amount of bytes used
  // at the site to be at least as much as the async copy bytes. There are two
  // factors that determine the copy and use bytes:
  //   - Some uses don't actually access the entire tensor, e.g. in
  //     dynamic-update-slice.
  //   - copy_bytes may be larger than the size of the tensor as well. An
  //     example is a tensor may be prefetched, used, and then evicted. In that
  //     case copy_bytes would be twice the size of the tensor.
  float inefficient_use_to_copy_ratio = 0.0;

  // This is mostly used for testing, it allows a test case to inject its own
  // logic for MsaAlgorithm::GetInefficientAllocationSites.
  std::function<std::vector<std::variant<HloPosition, HloUse>>(
      absl::Span<HloPosition>)>
      get_inefficient_allocation_sites_fn = nullptr;

  // Config to filter prefetches and update preferred prefetch times for the
  // filtered prefetches.
  PreferredPrefetchOverrides preferred_prefetch_overrides;

  // Options for slicing prefetches into smaller asynchronously copied pieces.
  SlicedPrefetchOptions sliced_prefetch_options;

  // Options for the memory-bound loop optimizer feature.
  MemoryBoundLoopOptimizerOptions memory_bound_loop_optimizer_options;

  SliceProposalFunction propose_slice_fn = [](const Shape&,
                                              const SlicedPrefetchOptions&)
      -> absl::StatusOr<SliceProposalCollection> {
    return UnimplementedStrCat("Generation of SliceProposals unimplemented");
  };

  // Option to always spill buffers from alternate memory to default memory
  // and prefetching back to alternate memory(if needed) just in time for use.
  bool always_spill_to_default_memory = false;

  // If true, enables window prefetching. Window prefetching is a mechanism
  // where we prefetch windows of data into the alternate memory before the
  // first use of the buffer. This allows large tensors to be prefetched as well
  // and gives MSA more flexibility in choosing the prefetch time and how much
  // data to prefetch.
  bool enable_window_prefetch = false;

  // The mode to use for window prefetching.
  WindowPrefetchMode window_prefetch_mode = WindowPrefetchMode::kWindowExposure;

  MsaSortOrderOverrides msa_sort_order_overrides;

  // A mode that enables expanding scoped alternate memory allocations to the
  // largest contiguous open space available.
  ExpandedScopedAlternateMemoryMode::Value
      expanded_scoped_alternate_memory_mode =
          ExpandedScopedAlternateMemoryMode::DISABLED;

  std::vector<BufferColoring> buffer_colorings;

  // If set, this is the size of scoped alternate memory that we require MSA to
  // allocate for post-module operations.
  uint64_t post_module_scoped_alternate_memory_size_in_bytes = 0;

  std::string ToString() const;
};

}  // namespace memory_space_assignment
}  // namespace xla

#endif  // XLA_SERVICE_MEMORY_SPACE_ASSIGNMENT_OPTIONS_H_
