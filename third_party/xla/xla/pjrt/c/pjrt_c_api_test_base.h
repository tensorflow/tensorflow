/* Copyright 2023 The OpenXLA Authors.

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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/shape.h"

#ifndef XLA_PJRT_C_PJRT_C_API_TEST_BASE_H_
#define XLA_PJRT_C_PJRT_C_API_TEST_BASE_H_

namespace pjrt {

class PjrtCApiTestBase : public ::testing::Test {
 public:
  explicit PjrtCApiTestBase(const PJRT_Api* api);
  ~PjrtCApiTestBase() override;

 protected:
  const PJRT_Api* api_;
  PJRT_Client* client_;
  void destroy_client(PJRT_Client* client);

  int GetDeviceId(PJRT_DeviceDescription* device_desc) const;

  int GetDeviceId(PJRT_Device* device) const;

  bool IsValidDeviceId(PJRT_Device* device) const;

  int GetLocalHardwareId(PJRT_Device* device) const;

  absl::Span<PJRT_Device* const> GetClientDevices() const;

  int GetNumDevices() const;

  std::string BuildSingleDeviceCompileOptionStr();

  absl::Span<PJRT_Device* const> GetClientAddressableDevices() const;

  PJRT_Client_BufferFromHostBuffer_Args CreateBufferFromHostBufferArgs(
      const std::vector<float>& data, const xla::Shape& shape,
      xla::PjRtClient::HostBufferSemantics host_buffer_semantics,
      PJRT_Device* device = nullptr);

  std::pair<std::unique_ptr<PJRT_Buffer, ::pjrt::PJRT_BufferDeleter>,
            xla::PjRtFuture<>>
  create_buffer(PJRT_Device* device = nullptr);

  std::unique_ptr<PJRT_Error, ::pjrt::PJRT_ErrorDeleter> ToUniquePtr(
      PJRT_Error* error);

 private:
  PjrtCApiTestBase(const PjrtCApiTestBase&) = delete;
  void operator=(const PjrtCApiTestBase&) = delete;
};

}  // namespace pjrt

#endif  // XLA_PJRT_C_PJRT_C_API_TEST_BASE_H_
