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

#include "xla/python/ifrt_proxy/client/mpmd_executable.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "google/protobuf/text_format.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt_proxy/client/client_session.h"
#include "xla/python/ifrt_proxy/client/host_buffer.h"
#include "xla/python/ifrt_proxy/client/mock_client_session.h"
#include "xla/python/ifrt_proxy/client/mock_host_buffer.h"
#include "xla/python/ifrt_proxy/client/rpc_helper.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/versions.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "tsl/platform/protobuf.h"

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Pair;
using ::testing::Pointee;
using ::testing::Return;
using ::testing::UnorderedElementsAre;
using ::tsl::proto_testing::EquivToProto;
using ::tsl::proto_testing::Partially;
using ::tsl::protobuf::TextFormat;

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

IfrtProxyVersion Version(int protocol_version = protocol_version::kClientMax) {
  IfrtProxyVersion version;
  version.set_protocol_version(protocol_version);
  version.set_ifrt_serdes_version_number(
      SerDesVersion::current().version_number().value());
  return version;
}

class MpmdLoadedExecutableTest : public ::testing::Test {
 protected:
  void SetUp() override {
    session_ = std::make_shared<MockClientSession>();
    rpc_helper_ = std::make_shared<RpcHelper>(Version(), session_);

    host_buffer_store_ = std::make_shared<MockClientHostBufferStore>();
    rpc_helper_->set_host_buffer_store(host_buffer_store_);

    ON_CALL(*session_, Enqueue(_))
        .WillByDefault(Return(tsl::Future<ClientSession::Response>(
            absl::InternalError("Request has no mock handlers"))));

    ON_CALL(device0_, Id()).WillByDefault(Return(DeviceId(0)));
    ON_CALL(device1_, Id()).WillByDefault(Return(DeviceId(1)));
  }

  void ExpectBaseMetadataCall(uint64_t handle) {
    IfrtResponse default_response;
    EXPECT_CALL(*session_,
                Enqueue(Pointee(Partially(EquivToProto(absl::Substitute(
                    R"pb(loaded_executable_metadata_request {
                           loaded_executable_handle: $0
                         })pb",
                    handle))))))
        .WillOnce(MockClientSessionReturnResponse(default_response));
  }

  void ExpectMpmdMetadataCall(uint64_t handle,
                              const IfrtResponse& mpmd_response) {
    IfrtResponse response = mpmd_response;
    response.mutable_response_metadata();
    if (!response.has_loaded_executable_mpmd_metadata_response()) {
      response.mutable_loaded_executable_mpmd_metadata_response();
    }
    EXPECT_CALL(*session_,
                Enqueue(Pointee(Partially(EquivToProto(absl::Substitute(
                    R"pb(loaded_executable_mpmd_metadata_request {
                           mpmd_loaded_executable_handle: $0
                         })pb",
                    handle))))))
        .WillOnce(MockClientSessionReturnResponse(response));
  }

  void ExpectMpmdMetadataCall(uint64_t handle) {
    IfrtResponse default_response;
    default_response.mutable_response_metadata();
    default_response.mutable_loaded_executable_mpmd_metadata_response();
    ExpectMpmdMetadataCall(handle, default_response);
  }

  void ExpectDestructorCall(uint64_t handle) {
    IfrtResponse destruct_response;
    EXPECT_CALL(*session_,
                Enqueue(Pointee(Partially(EquivToProto(absl::Substitute(
                    R"pb(loaded_executable_destruct_request {
                           loaded_executable_handle: $0
                         })pb",
                    handle))))))
        .WillOnce(MockClientSessionReturnResponse(destruct_response));
  }

  void SetupConstructorExpectations(uint64_t handle,
                                    const IfrtResponse& mpmd_response) {
    ExpectBaseMetadataCall(handle);
    ExpectMpmdMetadataCall(handle, mpmd_response);
    ExpectDestructorCall(handle);
  }

  void SetupConstructorExpectations(uint64_t handle) {
    IfrtResponse default_mpmd_response;
    default_mpmd_response.mutable_loaded_executable_mpmd_metadata_response();
    SetupConstructorExpectations(handle, default_mpmd_response);
  }

  std::shared_ptr<MockClientSession> session_;
  std::shared_ptr<RpcHelper> rpc_helper_;
  std::shared_ptr<ClientHostBufferStore> host_buffer_store_;
  MockClient client_;
  MockDevice device0_;
  MockDevice device1_;
};

TEST_F(MpmdLoadedExecutableTest, GetMpmdAddressableDevicesSuccess) {
  SetupConstructorExpectations(1234);

  absl::flat_hash_map<std::string, std::vector<xla::ifrt::Device*>> devices_map;
  devices_map["mesh0"] = {&device0_};
  devices_map["mesh1"] = {&device1_};

  MpmdLoadedExecutable executable(
      &client_, rpc_helper_, /*handle=*/1234, /*name=*/"mpmd_foo",
      /*num_devices=*/2, /*devices=*/{},
      /*addressable_devices=*/{}, /*mpmd_addressable_devices=*/devices_map,
      /*fingerprint=*/"fingerprint",
      /*loaded_host_callbacks=*/{}, /*loaded_host_callback_handles=*/{});

  TF_ASSERT_OK_AND_ASSIGN(auto result, executable.GetMpmdAddressableDevices());
  EXPECT_THAT(result,
              UnorderedElementsAre(Pair("mesh0", ElementsAre(&device0_)),
                                   Pair("mesh1", ElementsAre(&device1_))));
}

TEST_F(MpmdLoadedExecutableTest, GetMpmdAddressableDevicesError) {
  SetupConstructorExpectations(1234);

  MpmdLoadedExecutable executable(
      &client_, rpc_helper_, /*handle=*/1234, /*name=*/"mpmd_foo",
      /*num_devices=*/2, /*devices=*/{},
      /*addressable_devices=*/{},
      /*mpmd_addressable_devices=*/absl::InternalError("injected error"),
      /*fingerprint=*/"fingerprint",
      /*loaded_host_callbacks=*/{}, /*loaded_host_callback_handles=*/{});

  EXPECT_THAT(
      executable.GetMpmdAddressableDevices(),
      absl_testing::StatusIs(absl::StatusCode::kInternal, "injected error"));
}

TEST_F(MpmdLoadedExecutableTest, GetMpmdAddressableDevicesVersionCheck) {
  rpc_helper_ = std::make_shared<RpcHelper>(
      Version(protocol_version::kMpmdLoadedExecutableMethods - 1), session_);
  SetupConstructorExpectations(1234);

  MpmdLoadedExecutable executable(
      &client_, rpc_helper_, /*handle=*/1234, /*name=*/"mpmd_foo",
      /*num_devices=*/2, /*devices=*/{},
      /*addressable_devices=*/{},
      /*mpmd_addressable_devices=*/absl::InternalError("injected error"),
      /*fingerprint=*/"fingerprint",
      /*loaded_host_callbacks=*/{}, /*loaded_host_callback_handles=*/{});

  EXPECT_THAT(executable.GetMpmdAddressableDevices(),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                     HasSubstr("GetMpmdAddressableDevices")));
}

TEST_F(MpmdLoadedExecutableTest, GetMpmdCompiledMemoryStatsSuccess) {
  IfrtResponse response;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(
        loaded_executable_mpmd_metadata_response {
          mpmd_compiled_memory_stats {
            compiled_memory_stats {
              key: "mesh0"
              value { generated_code_size_in_bytes: 1024 }
            }
          }
        }
      )pb",
      &response));
  SetupConstructorExpectations(1234, response);

  MpmdLoadedExecutable executable(
      &client_, rpc_helper_, /*handle=*/1234, /*name=*/"mpmd_foo",
      /*num_devices=*/1, /*devices=*/{},
      /*addressable_devices=*/{}, /*mpmd_addressable_devices=*/
      absl::flat_hash_map<std::string, std::vector<xla::ifrt::Device*>>(),
      /*fingerprint=*/"fingerprint",
      /*loaded_host_callbacks=*/{}, /*loaded_host_callback_handles=*/{});

  TF_ASSERT_OK_AND_ASSIGN(auto stats, executable.GetMpmdCompiledMemoryStats());
  EXPECT_THAT(stats, UnorderedElementsAre(Pair("mesh0", _)));
  EXPECT_EQ(stats["mesh0"].generated_code_size_in_bytes, 1024);
}

TEST_F(MpmdLoadedExecutableTest, GetMpmdCompiledMemoryStatsRpcError) {
  ExpectBaseMetadataCall(1234);
  EXPECT_CALL(*session_, Enqueue(Pointee(Partially(EquivToProto(
                             R"pb(loaded_executable_mpmd_metadata_request {
                                    mpmd_loaded_executable_handle: 1234
                                  })pb")))))
      .WillOnce(Return(tsl::Future<ClientSession::Response>(
          absl::UnavailableError("RPC failed"))));
  ExpectDestructorCall(1234);

  MpmdLoadedExecutable executable(
      &client_, rpc_helper_, /*handle=*/1234, /*name=*/"mpmd_foo",
      /*num_devices=*/1, /*devices=*/{},
      /*addressable_devices=*/{}, /*mpmd_addressable_devices=*/
      absl::flat_hash_map<std::string, std::vector<xla::ifrt::Device*>>(),
      /*fingerprint=*/"fingerprint",
      /*loaded_host_callbacks=*/{}, /*loaded_host_callback_handles=*/{});

  EXPECT_THAT(executable.GetMpmdCompiledMemoryStats(),
              absl_testing::StatusIs(absl::StatusCode::kUnavailable,
                                     HasSubstr("RPC failed")));
}

TEST_F(MpmdLoadedExecutableTest, GetMpmdCompiledMemoryStatsVersionCheck) {
  rpc_helper_ = std::make_shared<RpcHelper>(
      Version(protocol_version::kMpmdLoadedExecutableMethods - 1), session_);
  ExpectBaseMetadataCall(1234);
  EXPECT_CALL(*session_, Enqueue(Pointee(Partially(EquivToProto(
                             R"pb(loaded_executable_mpmd_metadata_request {
                                    mpmd_loaded_executable_handle: 1234
                                  })pb")))))
      .WillOnce(Return(tsl::Future<ClientSession::Response>(
          absl::UnimplementedError("Should not be called"))));
  ExpectDestructorCall(1234);

  MpmdLoadedExecutable executable(
      &client_, rpc_helper_, /*handle=*/1234, /*name=*/"mpmd_foo",
      /*num_devices=*/1, /*devices=*/{},
      /*addressable_devices=*/{}, /*mpmd_addressable_devices=*/
      absl::flat_hash_map<std::string, std::vector<xla::ifrt::Device*>>(),
      /*fingerprint=*/"fingerprint",
      /*loaded_host_callbacks=*/{}, /*loaded_host_callback_handles=*/{});

  EXPECT_THAT(executable.GetMpmdCompiledMemoryStats(),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                     HasSubstr("GetMpmdCompiledMemoryStats")));
}

TEST_F(MpmdLoadedExecutableTest, GetMpmdHloModules) {
  SetupConstructorExpectations(1234);

  MpmdLoadedExecutable executable(
      &client_, rpc_helper_, /*handle=*/1234, /*name=*/"mpmd_foo",
      /*num_devices=*/1, /*devices=*/{},
      /*addressable_devices=*/{}, /*mpmd_addressable_devices=*/
      absl::flat_hash_map<std::string, std::vector<xla::ifrt::Device*>>(),
      /*fingerprint=*/"fingerprint",
      /*loaded_host_callbacks=*/{}, /*loaded_host_callback_handles=*/{});

  EXPECT_THAT(executable.GetMpmdHloModules(),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(MpmdLoadedExecutableTest, GetMpmdCostAnalysisSuccess) {
  SetupConstructorExpectations(1234);

  IfrtResponse response;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(
        loaded_executable_mpmd_cost_analysis_response {
          attributes {
            attributes {
              key: "mesh0"
              value {
                attributes {
                  key: "cost"
                  value { float_value: 1.0 }
                }
              }
            }
          }
        }
      )pb",
      &response));
  EXPECT_CALL(*session_, Enqueue(Pointee(Partially(EquivToProto(
                             R"pb(loaded_executable_mpmd_cost_analysis_request {
                                    loaded_executable_handle: 1234
                                  })pb")))))
      .WillOnce(MockClientSessionReturnResponse(response));

  MpmdLoadedExecutable executable(
      &client_, rpc_helper_, /*handle=*/1234, /*name=*/"mpmd_foo",
      /*num_devices=*/1, /*devices=*/{},
      /*addressable_devices=*/{}, /*mpmd_addressable_devices=*/
      absl::flat_hash_map<std::string, std::vector<xla::ifrt::Device*>>(),
      /*fingerprint=*/"fingerprint",
      /*loaded_host_callbacks=*/{}, /*loaded_host_callback_handles=*/{});

  TF_ASSERT_OK_AND_ASSIGN(auto result, executable.GetMpmdCostAnalysis());
  EXPECT_THAT(result, UnorderedElementsAre(Pair("mesh0", _)));
  EXPECT_THAT(result.at("mesh0").Get<float>("cost"),
              absl_testing::IsOkAndHolds(Eq(1.0f)));

  TF_ASSERT_OK_AND_ASSIGN(auto result2, executable.GetMpmdCostAnalysis());
  EXPECT_THAT(result2, UnorderedElementsAre(Pair("mesh0", _)));
}

TEST_F(MpmdLoadedExecutableTest, GetMpmdCostAnalysisRpcError) {
  SetupConstructorExpectations(1234);

  EXPECT_CALL(*session_, Enqueue(Pointee(Partially(EquivToProto(
                             R"pb(loaded_executable_mpmd_cost_analysis_request {
                                    loaded_executable_handle: 1234
                                  })pb")))))
      .WillOnce(Return(tsl::Future<ClientSession::Response>(
          absl::UnavailableError("RPC failed"))));

  MpmdLoadedExecutable executable(
      &client_, rpc_helper_, /*handle=*/1234, /*name=*/"mpmd_foo",
      /*num_devices=*/1, /*devices=*/{},
      /*addressable_devices=*/{}, /*mpmd_addressable_devices=*/
      absl::flat_hash_map<std::string, std::vector<xla::ifrt::Device*>>(),
      /*fingerprint=*/"fingerprint",
      /*loaded_host_callbacks=*/{}, /*loaded_host_callback_handles=*/{});

  EXPECT_THAT(executable.GetMpmdCostAnalysis(),
              absl_testing::StatusIs(absl::StatusCode::kUnavailable,
                                     HasSubstr("RPC failed")));
}

TEST_F(MpmdLoadedExecutableTest, GetMpmdCostAnalysisVersionCheck) {
  rpc_helper_ = std::make_shared<RpcHelper>(
      Version(protocol_version::kMpmdLoadedExecutableMethods - 1), session_);
  SetupConstructorExpectations(1234);

  EXPECT_CALL(*session_, Enqueue(Pointee(Partially(EquivToProto(
                             R"pb(loaded_executable_mpmd_cost_analysis_request {
                                    loaded_executable_handle: 1234
                                  })pb")))))
      .Times(0);

  MpmdLoadedExecutable executable(
      &client_, rpc_helper_, /*handle=*/1234, /*name=*/"mpmd_foo",
      /*num_devices=*/1, /*devices=*/{},
      /*addressable_devices=*/{}, /*mpmd_addressable_devices=*/
      absl::flat_hash_map<std::string, std::vector<xla::ifrt::Device*>>(),
      /*fingerprint=*/"fingerprint",
      /*loaded_host_callbacks=*/{}, /*loaded_host_callback_handles=*/{});

  EXPECT_THAT(executable.GetMpmdCostAnalysis(),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                     HasSubstr("GetMpmdCostAnalysis")));
}

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
