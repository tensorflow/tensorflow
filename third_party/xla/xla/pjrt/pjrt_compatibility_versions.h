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

#ifndef XLA_PJRT_PJRT_COMPATIBILITY_VERSIONS_H_
#define XLA_PJRT_PJRT_COMPATIBILITY_VERSIONS_H_

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_build_stamp.h"
#include "xla/pjrt/proto/pjrt_build_stamp.pb.h"

namespace xla {

// Well-known `PjRtBuildStampProto.compatibility_versions` names and typed
// getters for them. Setters live in `pjrt_build_stamp_writer.h`,
// which have build visibility restricted to compiler build stamp factories.

// Name of the Mosaic IR compatibility version in
// `PjRtBuildStampProto.compatibility_versions`.
inline constexpr absl::string_view kMosaicCompatibilityVersionName = "mosaic";

absl::StatusOr<PjRtCompatibilityVersionProto> GetMosaicVersion(
    const PjRtBuildStamp& stamp);

}  // namespace xla

#endif  // XLA_PJRT_PJRT_COMPATIBILITY_VERSIONS_H_
