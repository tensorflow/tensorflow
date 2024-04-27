// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/python/ifrt_proxy/server/ifrt_session_handler.h"

#include <cstdint>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt_proxy/server/ifrt_backend.h"
#include "tsl/platform/status_matchers.h"

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

using ::testing::Not;
using ::tsl::testing::IsOk;

// FakeBackend. Currently: Fails or returns illegal values where possible.
// All other methods return dummy strings or empty vectors. Individual tests
// can make derived classes that override specific methods as needed.
class FakeBackend : public BackendInterface {
 public:
  FakeBackend() = default;
  ~FakeBackend() override = default;

  Future<BackendInterface::Response> Process(
      std::unique_ptr<IfrtRequest> request) override {
    return Future<BackendInterface::Response>(std::make_unique<IfrtResponse>());
  }
};

TEST(IfrtSessionHandlerTest, NullptrForBackendMakerFails) {
  EXPECT_THAT(IfrtSessionHandler::Create(1234, nullptr), Not(IsOk()));
}

TEST(IfrtSessionHandlerTest, SuccessfulCreation) {
  std::unique_ptr<BackendInterface> backend = std::make_unique<FakeBackend>();
  EXPECT_THAT(
      IfrtSessionHandler::Create(
          1234, [&](uint64_t session_id) { return std::move(backend); }),
      IsOk());
}

// TODO(b/282757875) Add "end-to-end" tests that cover the entire path from the
// Server/BidiReactor to the backend.  Since IfrtSessionHandler writes the
// responses (IfrtResponse messages) directly to the Bidi Reactor, tests for the
// actual processing of requests need a full server and a fake client that
// allows us retrieve and examine the responses.

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
