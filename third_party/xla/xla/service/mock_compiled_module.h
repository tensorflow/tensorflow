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

#ifndef XLA_SERVICE_MOCK_COMPILED_MODULE_H_
#define XLA_SERVICE_MOCK_COMPILED_MODULE_H_

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/compiled_module.h"
#include "xla/service/executable.h"
#include "xla/stream_executor/abi/executable_abi_version.h"
#include "xla/stream_executor/platform_id.h"

namespace xla {

using ::testing::Ref;

class MockCompiledModule : public CompiledModule {
 public:
  MOCK_METHOD(absl::StatusOr<std::string>, SerializeAsString, (),
              (const, override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Executable>>, LoadExecutable, (),
              (ref(&&), override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Executable>>, LoadExecutable,
              (stream_executor::PlatformId platform_id,
               const stream_executor::DeviceDescription& device_description),
              (ref(&&), override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<BufferAssignment>>,
              buffer_assignment, (), (const, override));
  MOCK_METHOD(const HloModule*, optimized_module, (), (const, override));
  MOCK_METHOD(std::shared_ptr<HloModule>, shared_optimized_module, (),
              (override));
  MOCK_METHOD(absl::StatusOr<stream_executor::ExecutableAbiVersion>,
              GetExecutableAbiVersion, (), (const, override));
};

}  // namespace xla

#endif  // XLA_SERVICE_MOCK_COMPILED_MODULE_H_
