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

#include "xla/pjrt/raw_buffer.h"

#include <optional>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/test.h"

namespace xla {

std::optional<absl::StatusOr<tsl::RCReference<PjRtRawBuffer>>> MockFactory(
    PjRtBuffer* buffer) {
  if (buffer == nullptr) {
    return std::nullopt;
  }
  return absl::InvalidArgumentError("MockFactory");
}

REGISTER_PJRT_RAW_BUFFER_FACTORY(MockFactory);

TEST(RawBufferTest, FactoryFallback) {
  auto status = PjRtRawBuffer::CreateRawAliasOfBuffer(nullptr).status();
  ASSERT_THAT(status,
              tsl::testing::StatusIs(tsl::error::INVALID_ARGUMENT,
                                     testing::HasSubstr("null buffer.")));
}

TEST(RawBufferTest, FactoryError) {
  int dummy;
  auto status = PjRtRawBuffer::CreateRawAliasOfBuffer(
                    reinterpret_cast<PjRtBuffer*>(&dummy))
                    .status();
  ASSERT_THAT(status,
              tsl::testing::StatusIs(tsl::error::INVALID_ARGUMENT,
                                     testing::HasSubstr("MockFactory")));
}

}  // namespace xla
