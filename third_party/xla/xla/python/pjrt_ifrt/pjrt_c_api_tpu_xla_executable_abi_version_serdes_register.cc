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

#include "absl/base/attributes.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_abi_version_helpers.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/plugin/plugin_names.h"
#include "xla/pjrt/proto/pjrt_abi_version.pb.h"

namespace xla {

namespace tpu_xla_executable_abi_version_serdes {

ABSL_ATTRIBUTE_WEAK
absl::StatusOr<std::unique_ptr<xla::PjRtExecutableAbiVersion>>
PjRtExecutableAbiVersionFromProto(
    const xla::PjRtExecutableAbiVersionProto& proto) {
  ASSIGN_OR_RETURN(const PJRT_Api* c_api, pjrt::PjrtApi(kTpuPjrtName));
  return pjrt::CApiExecutableAbiVersionFromProto(proto, c_api);
}

}  // namespace tpu_xla_executable_abi_version_serdes

}  // namespace xla
