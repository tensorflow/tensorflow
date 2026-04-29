/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_RUNTIME_CHIP_ID_H_
#define XLA_RUNTIME_CHIP_ID_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/lib/gtl/int_type.h"

namespace xla {

// Some of the accelerator devices consist of multiple chips, and XLA might need
// to address them separately. For example GB200 (1) from NVIDIA can be viewed
// as a single device that consists of one Grace CPU and two Blackwell GPUs (one
// `LocalDeviceId` with two `LocalChipId`s). Some TPU chips from Google have two
// tensor cores (2) that appear as a single device to JAX/XLA users, and XLA
// (including PjRt runtime) need to address these chips separately.
//
// (1) https://www.nvidia.com/en-us/data-center/gb200-nvl72/
// (2)
// https://docs.jax.dev/en/latest/pallas/tpu/pipelining.html#tpus-in-megacore-configuration

// Strongly-typed integer type for identifying local chips that belong to one of
// the local devices.
TSL_LIB_GTL_DEFINE_INT_TYPE(LocalChipId, int32_t);

// Strongly-typed integer type for identifying global chips in a distributed
// execution.
TSL_LIB_GTL_DEFINE_INT_TYPE(GlobalChipId, int32_t);

template <typename Sink>
void AbslStringify(Sink& sink, LocalChipId id) {
  absl::Format(&sink, "%d", id.value());
}

template <typename Sink>
void AbslStringify(Sink& sink, GlobalChipId id) {
  absl::Format(&sink, "%d", id.value());
}

// StrJoin for global chip ids that shortens long list of ids for readability.
//
// It is not uncommon to see in XLA a list of global chips with more than 1k
// of entries. We don't need to print them all to get a human readable list
// of chips for logging and debugging.
inline std::string HumanReadableChips(absl::Span<const GlobalChipId> chips,
                                      absl::string_view separator = ",",
                                      size_t first = 8, size_t last = 2) {
  if (chips.size() > first + last) {
    return absl::StrCat(absl::StrJoin(chips.first(first), separator), "...",
                        absl::StrJoin(chips.last(last), separator));
  }
  return absl::StrJoin(chips, separator);
}

}  // namespace xla

#endif  // XLA_RUNTIME_CHIP_ID_H_
