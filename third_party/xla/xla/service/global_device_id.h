/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GLOBAL_DEVICE_ID_H_
#define XLA_SERVICE_GLOBAL_DEVICE_ID_H_

#include <string>

#include "absl/types/span.h"
#include "xla/runtime/device_id.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service.h"

namespace xla {

// DEPRECATED: Use GlobalDeviceId from device_id.h instead.
using GlobalDeviceId = GlobalDeviceId;

// Returns a comma-separated string of global device IDs.
std::string GlobalDeviceIdsToString(absl::Span<GlobalDeviceId const> ids);

using ::tsl::IncarnationId;  // NOLINT(misc-unused-using-decls)

}  // namespace xla

#endif  // XLA_SERVICE_GLOBAL_DEVICE_ID_H_
