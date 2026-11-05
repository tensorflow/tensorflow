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

#ifndef XLA_PJRT_PJRT_BUILD_STAMP_WRITER_H_
#define XLA_PJRT_PJRT_BUILD_STAMP_WRITER_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/proto/pjrt_build_stamp.pb.h"

// Helpers for compiler build stamp factories to populate a
// `PjRtBuildStampProto` before calling `PjRtBuildStamp::FromProto()` and
// registering the result via `PjRtRegisterBuildStampFactory()`.
//
// Visibility of this target is restricted to build stamp factories. Consumers
// should read build stamps via `PjRtGetBuildStamp()` and the getters in
// `pjrt_compatibility_versions.h`.

namespace xla {

// Returns a proto populated with the build provenance of the current binary
// (build timestamp, VCS revision, and build label).
PjRtBuildStampProto PjRtBuildStampProtoFromBuildData();

// Sets the attribute `key` to `value` in `proto`.
// Each attribute key may be set at most once.
absl::Status SetAttribute(PjRtBuildStampProto& proto, absl::string_view key,
                          const PjRtValueType& value);

// Typed setters for well-known `PjRtBuildStampProto.compatibility_versions`.
// Version names and getters live in `pjrt_compatibility_versions.h`.
//
// Each version may be set at most once.

// Sets the Mosaic IR compatibility version supported by the compiler.
absl::Status SetMosaicVersion(PjRtBuildStampProto& proto, int64_t major_version,
                              int64_t minor_version = 0);

}  // namespace xla

#endif  // XLA_PJRT_PJRT_BUILD_STAMP_WRITER_H_
