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

#include "xla/pjrt/plugin/testing/testing_cpp_pjrt.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "tsl/platform/fingerprint.h"

namespace testing {

class TestingPjrtClient : public xla::PjRtClient {
 public:
  TestingPjrtClient() = default;
  ~TestingPjrtClient() override = default;
  absl::string_view platform_name() const override;
  int process_index() const override;
  int device_count() const override;
  int addressable_device_count() const override;
  absl::Span<xla::PjRtDevice* const> devices() const override;
  absl::Span<xla::PjRtDevice* const> addressable_devices() const override;
  absl::Span<xla::PjRtMemorySpace* const> memory_spaces() const override;
  xla::PjRtPlatformId platform_id() const override;
  absl::string_view platform_version() const override;

 private:
  std::vector<xla::PjRtDevice*> devices_;
  std::vector<xla::PjRtMemorySpace*> memory_spaces_;
};  // end class

absl::string_view TestingPjrtClient::platform_name() const {
  return "testing_pjrt_client";
}

int TestingPjrtClient::process_index() const { return 0; }

xla::PjRtPlatformId TestingPjrtClient::platform_id() const {
  constexpr char kTestingBackendName[] = "testing_backend";
  static const uint64_t kTestingBackendId =
      tsl::Fingerprint64(kTestingBackendName);
  return kTestingBackendId;
}

int TestingPjrtClient::device_count() const { return 0; }

int TestingPjrtClient::addressable_device_count() const { return 0; }

absl::Span<xla::PjRtDevice* const> TestingPjrtClient::addressable_devices()
    const {
  return devices_;
}

absl::Span<xla::PjRtDevice* const> TestingPjrtClient::devices() const {
  return devices_;
}

absl::Span<xla::PjRtMemorySpace* const> TestingPjrtClient::memory_spaces()
    const {
  return memory_spaces_;
}

absl::string_view TestingPjrtClient::platform_version() const {
  return "Testing platform version";
}

std::unique_ptr<xla::PjRtClient> CreateTestingPjrtClient() {
  return std::make_unique<testing::TestingPjrtClient>();
}

}  // namespace testing
