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

#include "xla/pjrt/gpu/abi_helpers.h"

#include <memory>

#include "absl/status/statusor.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_abi_version_helpers.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/plugin/plugin_names.h"
#include "xla/pjrt/proto/pjrt_abi_version.pb.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {

absl::StatusOr<std::unique_ptr<xla::PjRtRuntimeAbiVersion>>
PjRtRuntimeAbiVersionFromProto(const xla::PjRtRuntimeAbiVersionProto& proto) {
  ASSIGN_OR_RETURN(const PJRT_Api* c_api, pjrt::PjrtApi(kGpuPjrtName));
  return pjrt::CApiRuntimeAbiVersionFromProto(proto, c_api);
}

absl::StatusOr<std::unique_ptr<xla::PjRtExecutableAbiVersion>>
PjRtExecutableAbiVersionFromProto(
    const xla::PjRtExecutableAbiVersionProto& proto) {
  ASSIGN_OR_RETURN(const PJRT_Api* c_api, pjrt::PjrtApi(kGpuPjrtName));
  return pjrt::CApiExecutableAbiVersionFromProto(proto, c_api);
}

}  // namespace xla::gpu
