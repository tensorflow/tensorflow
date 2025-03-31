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

#include "xla/python/ifrt_proxy/client/compiler.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/program.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt_proxy/client/client_session.h"
#include "xla/python/ifrt_proxy/client/host_buffer.h"
#include "xla/python/ifrt_proxy/client/mock_client_session.h"
#include "xla/python/ifrt_proxy/client/mock_host_buffer.h"
#include "xla/python/ifrt_proxy/client/rpc_helper.h"
#include "xla/python/ifrt_proxy/client/version.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/test_utils.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Invoke;
using ::testing::Optional;
using ::testing::Return;
using ::tsl::protobuf::TextFormat;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

struct TestProgram : llvm::RTTIExtends<TestProgram, Program> {
  static char ID;  // NOLINT
};

[[maybe_unused]] char TestProgram::ID = 0;  // NOLINT

class TestProgramSerDes : public llvm::RTTIExtends<TestProgramSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::proxy::TestProgram";
  }

  absl::StatusOr<std::string> Serialize(
      Serializable& serializable, std::unique_ptr<SerializeOptions>) override {
    CHECK(llvm::isa<TestProgram>(serializable));
    return "";
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    return std::make_unique<TestProgram>();
  }

  static char ID;  // NOLINT
};

[[maybe_unused]] char TestProgramSerDes::ID = 0;  // NOLINT

struct TestCompileOptions
    : llvm::RTTIExtends<TestCompileOptions, CompileOptions> {
  static char ID;  // NOLINT
};

[[maybe_unused]] char TestCompileOptions::ID = 0;  // NOLINT

class TestCompileOptionsSerDes
    : public llvm::RTTIExtends<TestCompileOptionsSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::proxy::TestCompileOptions";
  }

  absl::StatusOr<std::string> Serialize(
      Serializable& serializable, std::unique_ptr<SerializeOptions>) override {
    CHECK(llvm::isa<TestCompileOptions>(serializable));
    return "";
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const std::string& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    return std::make_unique<TestCompileOptions>();
  }

  static char ID;  // NOLINT
};

[[maybe_unused]] char TestCompileOptionsSerDes::ID = 0;  // NOLINT

IfrtProxyVersion Version() {
  IfrtProxyVersion version;
  version.set_protocol_version(kClientMinVersion);
  return version;
}

class CompilerTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    RegisterSerDes<TestProgram>(std::make_unique<TestProgramSerDes>());
    RegisterSerDes<TestCompileOptions>(
        std::make_unique<TestCompileOptionsSerDes>());
  }

  void SetUp() override {
    session_ = std::make_shared<MockClientSession>();
    rpc_helper_ = std::make_shared<RpcHelper>(Version(), session_);

    host_buffer_store_ = std::make_shared<MockClientHostBufferStore>();
    rpc_helper_->set_host_buffer_store(host_buffer_store_);

    // Default handler that ignores all uninteresting requests but still
    // invokes the callback in order to avoid hanging the caller forever.
    EXPECT_CALL(*session_, Enqueue(_))
        .WillRepeatedly(Return(Future<ClientSession::Response>(
            absl::InternalError("Request has no mock handlers"))));
  }

  std::shared_ptr<MockClientSession> session_;
  std::shared_ptr<RpcHelper> rpc_helper_;
  std::shared_ptr<ClientHostBufferStore> host_buffer_store_;
};

TEST_F(CompilerTest, Compile) {
  std::vector<MockDevice> devices(2);
  TestQueue<IfrtRequest> requests_queue(/*pop_timeout=*/absl::Minutes(1));

  MockClient client;
  ON_CALL(client, LookupDevice(_)).WillByDefault(Invoke([&](DeviceId id) {
    return &devices[id.value()];
  }));

  Compiler compiler(&client, rpc_helper_);

  IfrtResponse response;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(compile_response {
             loaded_executable_handle: 1234
             name: "foo-executable"
             num_devices: 2
             addressable_device_ids: [ 0, 1 ]
             fingerprint_value: "fingerprint"
             ready_future_handle: 5678
           })pb",
      &response));
  EXPECT_CALL(*session_,
              Enqueue(IfrtRequestOfType(IfrtRequest::kCompileRequest)))
      .WillOnce(MockClientCaptureAndReturn(&requests_queue, response));

  ASSERT_TRUE(TextFormat::ParseFromString(R"pb(
                                            response_metadata {
                                              status {
                                                code: 2  # UNKNOWN
                                                message: "injected error"
                                              }
                                            }
                                          )pb",
                                          &response));
  EXPECT_CALL(*session_,
              Enqueue(IfrtRequestOfType(IfrtRequest::kCheckFutureRequest)))
      .WillOnce(MockClientCaptureAndReturn(&requests_queue, response));

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      compiler.Compile(std::make_unique<TestProgram>(),
                       std::make_unique<TestCompileOptions>()));

  EXPECT_EQ(requests_queue.Pop().compile_request().program().type_name(),
            "xla::ifrt::proxy::TestProgram");

  EXPECT_EQ(executable->name(), "foo-executable");
  EXPECT_EQ(executable->num_devices(), 2);
  EXPECT_THAT(executable->addressable_devices(),
              ElementsAre(&devices[0], &devices[1]));
  EXPECT_THAT(executable->Fingerprint(),
              IsOkAndHolds(Optional(std::string("fingerprint"))));
  EXPECT_THAT(executable->GetReadyFuture().Await(),
              StatusIs(absl::StatusCode::kUnknown, "injected error"));

  EXPECT_EQ(requests_queue.Pop().check_future_request().future_handle(), 5678);
}

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
