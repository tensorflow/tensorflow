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

#include <memory>

#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/proto/pjrt_abi_version.pb.h"
#include "xla/pjrt/se/stream_executor_pjrt_abi_version.h"

namespace xla {
namespace gpu_xla_executable_abi_version_serdes {

absl::StatusOr<std::unique_ptr<xla::PjRtExecutableAbiVersion>>
PjRtExecutableAbiVersionFromProto(
    const xla::PjRtExecutableAbiVersionProto& proto) {
  return StreamExecutorPjRtExecutableAbiVersion::FromProto(proto);
}

}  // namespace gpu_xla_executable_abi_version_serdes
}  // namespace xla
