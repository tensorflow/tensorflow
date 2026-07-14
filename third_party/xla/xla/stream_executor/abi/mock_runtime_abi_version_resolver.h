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

#ifndef XLA_STREAM_EXECUTOR_ABI_MOCK_RUNTIME_ABI_VERSION_RESOLVER_H_
#define XLA_STREAM_EXECUTOR_ABI_MOCK_RUNTIME_ABI_VERSION_RESOLVER_H_

#include <memory>

#include <gmock/gmock.h>
#include "absl/status/statusor.h"
#include "xla/stream_executor/abi/runtime_abi_version.h"
#include "xla/stream_executor/abi/runtime_abi_version_resolver.h"

namespace stream_executor {

class MockRuntimeAbiVersionResolver : public RuntimeAbiVersionResolver {
 public:
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<RuntimeAbiVersion>>,
              GetRuntimeAbiVersion, (const RuntimeAbiVersionProto& proto),
              (const, override));
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ABI_MOCK_RUNTIME_ABI_VERSION_RESOLVER_H_
