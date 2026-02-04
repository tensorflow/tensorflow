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

#ifndef XLA_RUNTIME_DEVICE_ID_H_
#define XLA_RUNTIME_DEVICE_ID_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service.h"
#include "xla/tsl/lib/gtl/int_type.h"

namespace xla {

// Strongly-typed integer type for naming a device globally within a distributed
// system. XLA doesn't have a strong opinion about what global numbering scheme
// is applied to GPUs; the user must provide a local -> global mapping via
// GpuExecutableRunOptions for the local GPUs.
TSL_LIB_GTL_DEFINE_INT_TYPE(GlobalDeviceId, int64_t);
TSL_LIB_GTL_DEFINE_INT_TYPE(LocalDeviceId, int64_t);

using ::tsl::IncarnationId;  // NOLINT(misc-unused-using-decls)

template <typename Sink>
void AbslStringify(Sink& sink, GlobalDeviceId id) {
  absl::Format(&sink, "%d", id.value());
}

template <typename Sink>
void AbslStringify(Sink& sink, LocalDeviceId id) {
  absl::Format(&sink, "%d", id.value());
}

// StrJoin for global devices that shortens long list of devices for readbility.
//
// It is not uncommon to see in XLA a list of global devices with more than 1k
// of entries. We don't need to print them all to get a human readable list
// of devices for logging and debugging.
inline std::string HumanReadableDevices(
    absl::Span<const GlobalDeviceId> devices, absl::string_view separator = ",",
    size_t first = 8, size_t last = 2) {
  if (devices.size() > first + last) {
    return absl::StrCat(absl::StrJoin(devices.first(first), separator), "...",
                        absl::StrJoin(devices.last(last), separator));
  }
  return absl::StrJoin(devices, separator);
}

}  // namespace xla

#endif  // XLA_RUNTIME_DEVICE_ID_H_
