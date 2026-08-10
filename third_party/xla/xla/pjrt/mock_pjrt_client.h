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

#ifndef XLA_PJRT_MOCK_PJRT_CLIENT_H_
#define XLA_PJRT_MOCK_PJRT_CLIENT_H_

#include <memory>

#include <gmock/gmock.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_executable.h"

namespace xla {

class MockPjRtClient : public PjRtClient {
 public:
  MOCK_METHOD(int, process_index, (), (const, override));
  MOCK_METHOD(int, device_count, (), (const, override));
  MOCK_METHOD(int, addressable_device_count, (), (const, override));
  MOCK_METHOD(absl::Span<PjRtDevice* const>, devices, (), (const, override));
  MOCK_METHOD(absl::Span<PjRtDevice* const>, addressable_devices, (),
              (const, override));
  MOCK_METHOD(absl::Span<PjRtMemorySpace* const>, memory_spaces, (),
              (const, override));
  MOCK_METHOD(PjRtPlatformId, platform_id, (), (const, override));
  MOCK_METHOD(absl::string_view, platform_name, (), (const, override));
  MOCK_METHOD(absl::string_view, platform_version, (), (const, override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<PjRtExecutable>>, Compile,
              (const XlaComputation& computation, CompileOptions options),
              (override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>,
              CompileAndLoad,
              (const XlaComputation& computation, CompileOptions options),
              (override));
};
}  // namespace xla

#endif  // XLA_PJRT_MOCK_PJRT_CLIENT_H_
