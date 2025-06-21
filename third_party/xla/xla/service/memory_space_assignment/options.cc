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

#include "xla/service/memory_space_assignment/options.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace xla {
namespace memory_space_assignment {

std::string PostAllocationTransformationUpdate::ToString() const {
  return absl::StrCat("to_be_removed: ",
                      absl::StrJoin(to_be_removed, ", ",
                                    [](std::string* out, const auto& entry) {
                                      absl::StrAppend(out, entry->name());
                                    }),
                      "\n", "update_use_map: ",
                      absl::StrJoin(update_use_map, ", ",
                                    [](std::string* out, const auto& entry) {
                                      absl::StrAppend(
                                          out, "<", entry.first.ToString(),
                                          " -> ", entry.second.ToString(), ">");
                                    }),
                      "\n");
}

std::string Options::ToString() const {
  return absl::StrJoin(
      {absl::StrCat("default_memory_space: ", default_memory_space),
       absl::StrCat("alternate_memory_space: ", alternate_memory_space),
       absl::StrCat("max_size_in_bytes: ", max_size_in_bytes),
       absl::StrCat("alignment_in_bytes: ", alignment_in_bytes),
       absl::StrCat("replicated_split_dimension: ", replicated_split_dimension),
       absl::StrCat("any_split_dimension: ", any_split_dimension),
       absl::StrCat("reduce_scoped_memory_limit: ", reduce_scoped_memory_limit),
       absl::StrCat("allocate_reserved_scoped_memory_at_same_offset: ",
                    allocate_reserved_scoped_memory_at_same_offset),
       absl::StrCat("max_outstanding_prefetches: ", max_outstanding_prefetches),
       absl::StrCat("max_outstanding_evictions: ", max_outstanding_evictions),
       absl::StrCat("while_use_extra_outstanding_prefetch_limit: ",
                    while_use_extra_outstanding_prefetch_limit),
       absl::StrCat("max_retries: ", max_retries),
       absl::StrCat("max_repacks: ", max_repacks),
       absl::StrCat("repack_after_every_allocation: ",
                    repack_after_every_allocation),
       absl::StrCat("verify: ", verify),
       absl::StrCat("enable_cross_program_prefetch: ",
                    enable_cross_program_prefetch),
       absl::StrCat("default_cross_program_prefetch_heuristic: ",
                    default_cross_program_prefetch_heuristic),
       absl::StrCat("enable_cross_program_prefetch_freeing: ",
                    enable_cross_program_prefetch_freeing),
       absl::StrCat("max_cross_program_prefetches: ",
                    max_cross_program_prefetches),
       absl::StrCat("cross_program_prefetch_permissive_mode: ",
                    cross_program_prefetch_permissive_mode),
       absl::StrCat("enable_while_redundant_eviction_elimination: ",
                    enable_while_redundant_eviction_elimination),
       absl::StrCat("use_repeated_instance_for_preferred_prefetch_time: ",
                    use_repeated_instance_for_preferred_prefetch_time),
       absl::StrCat("enforce_prefetch_fifo_order: ",
                    enforce_prefetch_fifo_order),
       absl::StrCat("enable_sync_copy_replacement: ",
                    enable_sync_copy_replacement),
       absl::StrCat("enable_sync_slice_replacement: ",
                    enable_sync_slice_replacement),
       absl::StrCat("extend_async_copies_limit_for_sync_mem_op_conversion: ",
                    extend_async_copies_limit_for_sync_mem_op_conversion),
       absl::StrCat("inefficient_use_to_copy_ratio: ",
                    inefficient_use_to_copy_ratio),
       absl::StrCat("always_spill_to_default_memory: ",
                    always_spill_to_default_memory),
       absl::StrCat("enable_window_prefetch: ", enable_window_prefetch),
       absl::StrCat("window_prefetch_mode: ",
                    window_prefetch_mode == WindowPrefetchMode::kWindowExposure
                        ? "kWindowExposure"
                        : "kWindowPrefetch"),
       absl::StrCat("expanded_scoped_alternate_memory_mode: ",
                    ExpandedScopedAlternateMemoryMode::Value_Name(
                        expanded_scoped_alternate_memory_mode)),
       absl::StrCat("buffer_colorings: ", buffer_colorings.size(), " elements"),
       absl::StrCat("post_module_scoped_alternate_memory_size_in_bytes: ",
                    post_module_scoped_alternate_memory_size_in_bytes),
       absl::StrCat("buffer_interval_comparator: ",
                    buffer_interval_comparator ? "present" : "nullptr"),
       absl::StrCat("prefetch_interval_picker: ",
                    prefetch_interval_picker ? "present" : "nullptr"),
       absl::StrCat("cost_analysis: ", cost_analysis ? "present" : "nullptr"),
       absl::StrCat("repacker: ", repacker ? "present" : "nullptr"),
       absl::StrCat("autotuning_config: ",
                    autotuning_config.has_value() ? "present" : "nullopt"),
       absl::StrCat("preferred_prefetch_overrides: ",
                    preferred_prefetch_overrides.DebugString()),
       absl::StrCat("sliced_prefetch_options: ",
                    sliced_prefetch_options.DebugString()),
       absl::StrCat("memory_bound_loop_optimizer_options: ",
                    memory_bound_loop_optimizer_options.DebugString()),
       absl::StrCat("msa_sort_order_overrides: ",
                    msa_sort_order_overrides.DebugString())},
      "\n");
}

}  // namespace memory_space_assignment
}  // namespace xla
