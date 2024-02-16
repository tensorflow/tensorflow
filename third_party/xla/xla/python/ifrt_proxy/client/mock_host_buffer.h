/*
 * Copyright 2023 The OpenXLA Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XLA_PYTHON_IFRT_PROXY_CLIENT_MOCK_HOST_BUFFER_H_
#define XLA_PYTHON_IFRT_PROXY_CLIENT_MOCK_HOST_BUFFER_H_

#include <cstdint>

#include <gmock/gmock.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt_proxy/client/host_buffer.h"

namespace xla {
namespace ifrt {
namespace proxy {

class MockClientHostBufferStore final : public ClientHostBufferStore {
 public:
  MOCK_METHOD(uint64_t, NextHandle, (), (override));
  MOCK_METHOD(Future<absl::Status>, Store,
              (uint64_t handle, absl::string_view data), (override));
  MOCK_METHOD(Future<absl::Status>, Store,
              (uint64_t handle, const absl::Cord& data), (override));
  MOCK_METHOD(Future<absl::StatusOr<absl::Cord>>, Lookup, (uint64_t handle),
              (override));
  MOCK_METHOD(Future<absl::Status>, Delete, (uint64_t handle), (override));
};

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_CLIENT_MOCK_HOST_BUFFER_H_
