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

#ifndef XLA_PYTHON_IFRT_PROXY_SERVER_MOCK_IFRT_BACKEND_H_
#define XLA_PYTHON_IFRT_PROXY_SERVER_MOCK_IFRT_BACKEND_H_

#include <memory>

#include <gmock/gmock.h>
#include "absl/status/status.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/server/ifrt_backend.h"

namespace xla {
namespace ifrt {
namespace proxy {

class MockIfrtBackend final : public BackendInterface {
 public:
  MOCK_METHOD(Future<Response>, Process, (std::unique_ptr<IfrtRequest> request),
              (final));
};

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_SERVER_MOCK_IFRT_BACKEND_H_
