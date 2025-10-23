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

#ifndef XLA_TOOLS_PLATFORM_XPROF_LOADER_H_
#define XLA_TOOLS_PLATFORM_XPROF_LOADER_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/statusor.h"
#include "xla/service/hlo.pb.h"

namespace xla {
absl::StatusOr<HloModuleProto> LoadHloModuleFromXprof(
    std::string xprof_session_id, std::optional<uint64_t> xprof_hlo_program_id);
}  // namespace xla

#endif  // XLA_TOOLS_PLATFORM_XPROF_LOADER_H_
