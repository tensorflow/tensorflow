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

#ifndef XLA_PJRT_C_PJRT_C_API_ABI_VERSION_HELPERS_H_
#define XLA_PJRT_C_PJRT_C_API_ABI_VERSION_HELPERS_H_

#include <memory>

#include "absl/status/statusor.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/pjrt_abi_version.h"
#include "xla/pjrt/proto/pjrt_abi_version.pb.h"

namespace pjrt {

absl::StatusOr<std::unique_ptr<xla::PjRtRuntimeAbiVersion>>
CApiRuntimeAbiVersionFromProto(const xla::PjRtRuntimeAbiVersionProto& proto,
                               const PJRT_Api* c_api);

absl::StatusOr<std::unique_ptr<xla::PjRtExecutableAbiVersion>>
CApiExecutableAbiVersionFromProto(
    const xla::PjRtExecutableAbiVersionProto& proto, const PJRT_Api* c_api);

}  // namespace pjrt

#endif  // XLA_PJRT_C_PJRT_C_API_ABI_VERSION_HELPERS_H_
