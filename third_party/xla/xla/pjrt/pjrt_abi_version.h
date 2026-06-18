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

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/proto/pjrt_abi_version.pb.h"

namespace xla {
class PjRtExecutableAbiVersion;

// Abstract base class for representing the ABI version of a PJRT platform.
class PjRtRuntimeAbiVersion {
 public:
  virtual ~PjRtRuntimeAbiVersion() = default;

  // Returns OK if compatible with other runtime ABI version.
  virtual absl::Status IsCompatibleWith(
      const PjRtRuntimeAbiVersion& runtime_abi_version) const = 0;
  // Returns OK if compatible with the executable ABI.
  virtual absl::Status IsCompatibleWith(
      const PjRtExecutableAbiVersion& executable_abi_version) const = 0;

  virtual absl::StatusOr<PjRtRuntimeAbiVersionProto> ToProto() const = 0;
  virtual PjRtPlatformId platform_id() const = 0;
};

// Abstract base class for representing a PjRtExecutable's ABI version.
class PjRtExecutableAbiVersion {
 public:
  virtual ~PjRtExecutableAbiVersion() = default;

  virtual absl::StatusOr<PjRtExecutableAbiVersionProto> ToProto() const = 0;
  virtual PjRtPlatformId platform_id() const = 0;
};

}  // namespace xla

#endif  // XLA_PJRT_PJRT_ABI_VERSION_H_
