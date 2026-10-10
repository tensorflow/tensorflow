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

#ifndef XLA_PJRT_PJRT_BUILD_STAMP_INTERNAL_H_
#define XLA_PJRT_PJRT_BUILD_STAMP_INTERNAL_H_

#include <cstdint>

#include "xla/pjrt/proto/pjrt_build_stamp.pb.h"

namespace xla {

// Sets the Mosaic IR compatibility version in `proto.mutable_versions()`.
// This is an internal helper for compiler factories / plugin builders and
// should not be called by general PJRT clients.
void SetMosaicVersion(PjRtBuildStampProto& proto, int64_t major_version,
                      int64_t minor_version = 0);

}  // namespace xla

#endif  // XLA_PJRT_PJRT_BUILD_STAMP_INTERNAL_H_
