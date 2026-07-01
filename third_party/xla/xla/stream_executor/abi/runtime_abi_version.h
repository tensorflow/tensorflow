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

#ifndef XLA_STREAM_EXECUTOR_ABI_RUNTIME_ABI_VERSION_H_
#define XLA_STREAM_EXECUTOR_ABI_RUNTIME_ABI_VERSION_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/abi/runtime_abi_version.pb.h"
#include "xla/stream_executor/platform_id.h"

namespace stream_executor {

class RuntimeAbiVersion {
 public:
  // Returns OK if the runtime ABI is compatible with the executable ABI.
  virtual absl::Status IsCompatibleWith(
      const ExecutableAbiVersion& executable_abi_version) const = 0;

  virtual absl::StatusOr<RuntimeAbiVersionProto> ToProto() const = 0;
  virtual absl::StatusOr<PlatformId> platform_id() const = 0;

  virtual ~RuntimeAbiVersion() = default;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ABI_RUNTIME_ABI_VERSION_H_
