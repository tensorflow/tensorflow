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

#ifndef XLA_PJRT_PJRT_ABI_VERSION_H_
#define XLA_PJRT_PJRT_ABI_VERSION_H_

#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/proto/pjrt_abi_version.pb.h"

namespace xla {

// Abstract base class for representing the ABI version of a PJRT platform.
class PjRtAbiVersion {
 public:
  virtual ~PjRtAbiVersion() = default;

  virtual absl::StatusOr<PjRtAbiVersionProto> ToProto() const = 0;
  virtual PjRtPlatformId platform_id() const = 0;

  // Comparator
  virtual bool IsCompatible(const PjRtAbiVersion& other) const = 0;
};

}  // namespace xla

#endif  // XLA_PJRT_PJRT_ABI_VERSION_H_
