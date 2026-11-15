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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/call_once.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "google/protobuf/text_format.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/basic_device_list.h"
#include "xla/python/ifrt/compiler.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/program.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt_proxy/client/client_session.h"
#include "xla/python/ifrt_proxy/client/host_buffer.h"
#include "xla/python/ifrt_proxy/client/mock_client_session.h"
#include "xla/python/ifrt_proxy/client/mock_host_buffer.h"
#include "xla/python/ifrt_proxy/client/rpc_helper.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/test_utils.h"
#include "xla/python/ifrt_proxy/common/versions.h"
#include "xla/python/pjrt_ifrt/xla_compiler.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/protobuf.h"

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

struct TestProgram : RTTIExtends<TestProgram, Program> {
  static char ID;  // NOLINT
};

[[maybe_unused]] char TestProgram::ID = 0;  // NOLINT

class TestProgramSerDes : public RTTIExtends<TestProgramSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::proxy::TestProgram";
  }

  absl::StatusOr<absl::Cord> Serialize(
      const Serializable& serializable,
      std::unique_ptr<SerializeOptions>) override {
    CHECK(isa<TestProgram>(serializable));
    return absl::Cord();
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const absl::Cord& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    return std::make_unique<TestProgram>();
  }

  static char ID;  // NOLINT
};

[[maybe_unused]] char TestProgramSerDes::ID = 0;  // NOLINT

struct TestCompileOptions : RTTIExtends<TestCompileOptions, CompileOptions> {
  static char ID;  // NOLINT
};

[[maybe_unused]] char TestCompileOptions::ID = 0;  // NOLINT

class TestCompileOptionsSerDes
    : public RTTIExtends<TestCompileOptionsSerDes, SerDes> {
 public:
  absl::string_view type_name() const override {
    return "xla::ifrt::proxy::TestCompileOptions";
  }

  absl::StatusOr<absl::Cord> Serialize(
      const Serializable& serializable,
      std::unique_ptr<SerializeOptions>) override {
    CHECK(isa<TestCompileOptions>(serializable));
    return absl::Cord();
  }

  absl::StatusOr<std::unique_ptr<Serializable>> Deserialize(
      const absl::Cord& serialized,
      std::unique_ptr<DeserializeOptions> options) override {
    return std::make_unique<TestCompileOptions>();
  }

  static char ID;  // NOLINT
};

[[maybe_unused]] char TestCompileOptionsSerDes::ID = 0;  // NOLINT

IfrtProxyVersion Version(int protocol = protocol_version::kClientMin) {
  IfrtProxyVersion version;
  version.set_protocol_version(protocol);
  version.set_ifrt_serdes_version_number(
      SerDesAnyVersionAccessor::GetMinimum().version_number().value());
  return version;
}

class CompilerTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    // Derived fixtures run this once per suite; register the SerDes once per
    // process.
    static absl::once_flag once;
    absl::call_once(once, [] {
      RegisterSerDes<TestProgram>(std::make_unique<TestProgramSerDes>());
      RegisterSerDes<TestCompileOptions>(
          std::make_unique<TestCompileOptionsSerDes>());
    });
  }

  void SetUp() override {
    session_ = std::make_shared<MockClientSession>();
    rpc_helper_ = std::make_shared<RpcHelper>(Version(), session_);

    host_buffer_store_ = std::make_shared<MockClientHostBufferStore>();
    rpc_helper_->set_host_buffer_store(host_buffer_store_);

    // Default handler that ignores all uninteresting requests but still
    // invokes the callback in order to avoid hanging the caller forever.
    EXPECT_CALL(*session_, Enqueue(_))
        .WillRepeatedly(Return(tsl::Future<ClientSession::Response>(
            absl::InternalError("Request has no mock handlers"))));
  }

  std::shared_ptr<MockClientSession> session_;
  std::shared_ptr<RpcHelper> rpc_helper_;
  std::shared_ptr<MockClientHostBufferStore> host_buffer_store_;
};

TEST_F(CompilerTest, Compile) {
  std::vector<MockDevice> devices(2);
  TestQueue<IfrtRequest> requests_queue(
      /*default_pop_timeout=*/absl::Minutes(1));
  auto device_list = BasicDeviceList::Create({&devices[0], &devices[1]});

  MockClient client;
  ON_CALL(client, LookupDevice(_)).WillByDefault([&](DeviceId id) {
    return &devices[id.value()];
  });
  ON_CALL(client, MakeDeviceList(_)).WillByDefault(Return(device_list));

  Compiler compiler(&client, rpc_helper_);

  IfrtResponse response;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(compile_response {
             loaded_executable_handle: 1234
             name: "foo-executable"
             num_devices: 2
             device_ids: [ 0, 1 ]
             addressable_device_ids: [ 0, 1 ]
             fingerprint_value: "fingerprint"
           })pb",
      &response));
  EXPECT_CALL(*session_,
              Enqueue(IfrtRequestOfType(IfrtRequest::kCompileRequest)))
      .WillOnce(MockClientCaptureAndReturn(&requests_queue, response));

  ASSERT_TRUE(TextFormat::ParseFromString(R"pb(
                                            response_metadata {
                                              status {
                                                code: 0  # OK
                                              }
                                            }
                                          )pb",
                                          &response));

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      compiler
          .CompileAndLoad(std::make_unique<TestProgram>(),
                          std::make_unique<TestCompileOptions>())
          .Await());

  EXPECT_EQ(requests_queue.Pop().compile_request().program().type_name(),
            "xla::ifrt::proxy::TestProgram");

  EXPECT_EQ(executable->name(), "foo-executable");
  EXPECT_THAT(executable->devices(), Optional(device_list));
  EXPECT_THAT(executable->Fingerprint(),
              absl_testing::IsOkAndHolds(Optional(std::string("fingerprint"))));
}

// Tests for `DeserializeLoadedExecutable`.
class DeserializeLoadedExecutableTest : public CompilerTest {
 protected:
  void SetUp() override {
    CompilerTest::SetUp();
    rpc_helper_ = std::make_shared<RpcHelper>(
        Version(protocol_version::kClientMax), session_);
    rpc_helper_->set_host_buffer_store(host_buffer_store_);

    for (int i = 0; i < devices_.size(); ++i) {
      ON_CALL(devices_[i], Id()).WillByDefault(Return(DeviceId(i)));
    }
    device_list_ = BasicDeviceList::Create({&devices_[0], &devices_[1]});
    ON_CALL(client_, LookupDevice(_)).WillByDefault([this](DeviceId id) {
      return &devices_[id.value()];
    });
    ON_CALL(client_, MakeDeviceList(_)).WillByDefault(Return(device_list_));
  }

  std::unique_ptr<DeserializeExecutableOptions> MakeOptions(
      std::optional<xla::CompileOptions> compile_options = std::nullopt) {
    return std::make_unique<XlaDeserializeExecutableOptions>(
        std::move(compile_options), device_list_);
  }

  std::vector<MockDevice> devices_ = std::vector<MockDevice>(2);
  DeviceListRef device_list_;
  MockClient client_;
};

TEST_F(DeserializeLoadedExecutableTest, UploadsBytesAndCreatesExecutable) {
  Compiler compiler(&client_, rpc_helper_);

  uint64_t stored_handle = 0;
  absl::Cord stored_data;
  EXPECT_CALL(*host_buffer_store_, Store(_, testing::An<const absl::Cord&>()))
      .WillOnce([&](uint64_t handle, const absl::Cord& data) {
        stored_handle = handle;
        stored_data = data;
        return tsl::Future<>(absl::OkStatus());
      });

  TestQueue<IfrtRequest> requests_queue(
      /*default_pop_timeout=*/absl::Minutes(1));
  IfrtResponse response;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(deserialize_loaded_executable_response {
             compile_response {
               loaded_executable_handle: 1234
               name: "deserialized-executable"
               num_devices: 2
               device_ids: [ 0, 1 ]
               addressable_device_ids: [ 0, 1 ]
               fingerprint_value: "fingerprint"
             }
           })pb",
      &response));
  EXPECT_CALL(*session_, Enqueue(IfrtRequestOfType(
                             IfrtRequest::kDeserializeLoadedExecutableRequest)))
      .WillOnce(MockClientCaptureAndReturn(&requests_queue, response));

  const absl::Cord serialized("serialized executable bytes");
  xla::CompileOptions compile_options;
  compile_options.executable_build_options.set_num_replicas(3);
  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      compiler
          .DeserializeLoadedExecutable(serialized, MakeOptions(compile_options))
          .Await());

  const IfrtRequest request = requests_queue.Pop();
  const DeserializeLoadedExecutableRequest& deserialize_request =
      request.deserialize_loaded_executable_request();
  EXPECT_EQ(deserialize_request.serialized_executable_handle(), stored_handle);
  EXPECT_NE(stored_handle, 0);
  EXPECT_EQ(stored_data, serialized);
  EXPECT_THAT(deserialize_request.device_ids(), ElementsAre(0, 1));
  ASSERT_TRUE(deserialize_request.has_compile_options());
  TF_ASSERT_OK_AND_ASSIGN(
      auto deserialized_options,
      Deserialize<CompileOptions>(deserialize_request.compile_options(),
                                  /*options=*/nullptr));
  auto* xla_options = dyn_cast<XlaCompileOptions>(deserialized_options.get());
  ASSERT_NE(xla_options, nullptr);
  EXPECT_EQ(
      xla_options->compile_options.executable_build_options.num_replicas(), 3);

  EXPECT_EQ(executable->name(), "deserialized-executable");
  EXPECT_THAT(executable->devices(), Optional(device_list_));
  EXPECT_THAT(executable->Fingerprint(),
              absl_testing::IsOkAndHolds(Optional(std::string("fingerprint"))));
}

TEST_F(DeserializeLoadedExecutableTest, PropagatesServerError) {
  Compiler compiler(&client_, rpc_helper_);
  EXPECT_CALL(*host_buffer_store_, Store(_, testing::An<const absl::Cord&>()))
      .WillOnce(Return(tsl::Future<>(absl::OkStatus())));

  // The server reports application errors in the response metadata.
  IfrtResponse response;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(response_metadata {
             status { code: 9 message: "ABI mismatch" }  # FAILED_PRECONDITION
           })pb",
      &response));
  EXPECT_CALL(*session_, Enqueue(IfrtRequestOfType(
                             IfrtRequest::kDeserializeLoadedExecutableRequest)))
      .WillOnce(Return(tsl::Future<ClientSession::Response>(
          std::make_shared<IfrtResponse>(response))));

  EXPECT_THAT(
      compiler.DeserializeLoadedExecutable(absl::Cord("bytes"), MakeOptions())
          .Await()
          .status(),
      absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                             testing::HasSubstr("ABI mismatch")));
}

TEST_F(DeserializeLoadedExecutableTest, RejectsUnsupportedOptionsType) {
  Compiler compiler(&client_, rpc_helper_);
  EXPECT_CALL(*host_buffer_store_, Store(_, testing::An<const absl::Cord&>()))
      .Times(0);
  EXPECT_CALL(*session_, Enqueue(_)).Times(0);

  // The base class is not `XlaDeserializeExecutableOptions`, so the client
  // cannot convert it into the request.
  EXPECT_THAT(
      compiler
          .DeserializeLoadedExecutable(
              absl::Cord("bytes"),
              std::make_unique<DeserializeExecutableOptions>(device_list_))
          .Await()
          .status(),
      absl_testing::StatusIs(
          absl::StatusCode::kUnimplemented,
          testing::HasSubstr(
              "supports only `XlaDeserializeExecutableOptions`")));
}

TEST_F(DeserializeLoadedExecutableTest, ReturnsUnimplementedOnOldServer) {
  rpc_helper_ = std::make_shared<RpcHelper>(
      Version(protocol_version::kDeserializeLoadedExecutable - 1), session_);
  rpc_helper_->set_host_buffer_store(host_buffer_store_);
  Compiler compiler(&client_, rpc_helper_);

  // An older server does not delete the host buffer, so the client must not
  // upload it.
  EXPECT_CALL(*host_buffer_store_, Store(_, testing::An<const absl::Cord&>()))
      .Times(0);
  EXPECT_CALL(*session_, Enqueue(_)).Times(0);

  EXPECT_THAT(
      compiler.DeserializeLoadedExecutable(absl::Cord("bytes"), MakeOptions())
          .Await()
          .status(),
      absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                             testing::HasSubstr("requires protocol version")));
}

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
