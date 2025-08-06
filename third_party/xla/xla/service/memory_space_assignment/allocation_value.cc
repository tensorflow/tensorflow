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

#include "xla/service/memory_space_assignment/allocation_value.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"

namespace xla::memory_space_assignment {

std::string AllocationRequest::ToString() const {
  return absl::StrJoin(
      {absl::StrCat("size: ", size),
       absl::StrCat("inclusive_start_time: ", inclusive_start_time,
                    "; end_time: ", end_time,
                    "; latest_prefetch_time: ", latest_prefetch_time,
                    "; required_copy_allocation_latest_time: ",
                    required_copy_allocation_latest_time),
       absl::StrCat("prefer_no_copy_alternate_mem_allocation: ",
                    prefer_no_copy_alternate_mem_allocation,
                    "; allow_no_copy_alternate_mem_allocation: ",
                    allow_no_copy_alternate_mem_allocation,
                    "; require_no_copy_alternate_mem_allocation: ",
                    require_no_copy_alternate_mem_allocation,
                    "; require_copy_allocation: ", require_copy_allocation),
       absl::StrCat("allow_prefetch: ", allow_prefetch),
       absl::StrCat("earliest_prefetch_time: ",
                    earliest_prefetch_time.has_value()
                        ? absl::StrCat(*earliest_prefetch_time)
                        : "nullopt"),
       absl::StrCat("preferred_prefetch_time: ",
                    preferred_prefetch_time.has_value()
                        ? absl::StrCat(*preferred_prefetch_time)
                        : "nullopt"),
       absl::StrCat(
           "preferred_offset: ",
           preferred_offset
               ? absl::StrCat(preferred_offset->offset, "; allocations: ",
                              preferred_offset->allocations.size())
               : "nullptr"),
       absl::StrCat("use: ", use ? use->hlo_use.ToString() : "nullptr"),
       absl::StrCat("allocation_value: ", allocation_value
                                              ? allocation_value->ToString()
                                              : "nullptr"),
       absl::StrCat("allocation_value_to_update: ",
                    allocation_value_to_update
                        ? allocation_value_to_update->ToString()
                        : "nullptr"),
       absl::StrCat("all_use_times: ", all_use_times.size(), " elements"),
       absl::StrCat("required_copy_allocation_for: ",
                    required_copy_allocation_for
                        ? required_copy_allocation_for->ToString()
                        : "nullptr"),
       absl::StrCat("required_copy_for_slice: ", required_copy_for_slice),
       absl::StrCat("only_extend_existing_allocation: ",
                    only_extend_existing_allocation),
       absl::StrCat("processed_allocation_values: ",
                    processed_allocation_values.size(), " elements"),
       absl::StrCat("no_copy_chunk_inclusive_start_time: ",
                    no_copy_chunk_inclusive_start_time.has_value()
                        ? absl::StrCat(*no_copy_chunk_inclusive_start_time)
                        : "nullopt"),
       absl::StrCat("require_start_colored_in_alternate_memmory: ",
                    require_start_colored_in_alternate_memory,
                    "; require_end_colored_in_alternate_memory: ",
                    require_end_colored_in_alternate_memory,
                    "; require_start_colored_in_default_memory: ",
                    require_start_colored_in_default_memory,
                    "; require_end_colored_in_default_memory: ",
                    require_end_colored_in_default_memory)},
      "\n");
}

}  // namespace xla::memory_space_assignment
