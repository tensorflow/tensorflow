/* Copyright 2019 The OpenXLA Authors.

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

#include "xla/stream_executor/allocator_stats.h"

#include <string>

#include "absl/strings/str_format.h"
#include "tsl/platform/numbers.h"

namespace stream_executor {

std::string AllocatorStats::DebugString() const {
  return absl::StrFormat(
      "Limit:            %20s\n"
      "InUse:            %20s\n"
      "MaxInUse:         %20s\n"
      "NumAllocs:        %20lld\n"
      "MaxAllocSize:     %20s\n"
      "Reserved:         %20s\n"
      "PeakReserved:     %20s\n"
      "LargestFreeBlock: %20s\n",
      tsl::strings::HumanReadableNumBytes(this->bytes_limit ? *this->bytes_limit
                                                            : 0),
      tsl::strings::HumanReadableNumBytes(this->bytes_in_use),
      tsl::strings::HumanReadableNumBytes(this->peak_bytes_in_use),
      this->num_allocs,
      tsl::strings::HumanReadableNumBytes(this->largest_alloc_size),
      tsl::strings::HumanReadableNumBytes(this->bytes_reserved),
      tsl::strings::HumanReadableNumBytes(this->peak_bytes_reserved),
      tsl::strings::HumanReadableNumBytes(this->largest_free_block_bytes));
}

}  // namespace stream_executor
