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

#include "xla/python/ifrt_proxy/client/executable.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "google/protobuf/text_format.h"
#include "xla/layout_util.h"
#include "xla/pjrt/profiling/device_time_measurement.h"
#include "xla/pjrt/profiling/test_util/mock_device_time_measurement.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/basic_device_list.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt_proxy/client/array.h"
#include "xla/python/ifrt_proxy/client/client_session.h"
#include "xla/python/ifrt_proxy/client/host_buffer.h"
#include "xla/python/ifrt_proxy/client/mock_client_session.h"
#include "xla/python/ifrt_proxy/client/mock_host_buffer.h"
#include "xla/python/ifrt_proxy/client/rpc_helper.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/test_utils.h"
#include "xla/python/ifrt_proxy/common/types.h"
#include "xla/python/ifrt_proxy/common/versions.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/platform.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Pointee;
using ::testing::Return;
using ::testing::SizeIs;
using ::tsl::proto_testing::EquivToProto;
using ::tsl::proto_testing::Partially;
using ::tsl::protobuf::TextFormat;

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

IfrtProxyVersion Version() {
  IfrtProxyVersion version;
  version.set_protocol_version(protocol_version::kClientMax);
  version.set_ifrt_serdes_version_number(
      SerDesVersion::current().version_number().value());
  return version;
}

class LoadedExecutableTest : public ::testing::Test {
 protected:
  void SetUp() override {
    session_ = std::make_shared<MockClientSession>();
    rpc_helper_ = std::make_shared<RpcHelper>(Version(), session_);

    host_buffer_store_ = std::make_shared<MockClientHostBufferStore>();
    rpc_helper_->set_host_buffer_store(host_buffer_store_);

    // Default handler that ignores all uninteresting requests, but still
    // invokes the callback in order to avoid hanging the caller forever.
    EXPECT_CALL(*session_, Enqueue(_))
        .WillRepeatedly(Return(tsl::Future<ClientSession::Response>(
            absl::InternalError("Request has no mock handlers"))));
  }

  std::shared_ptr<MockClientSession> session_;
  std::shared_ptr<RpcHelper> rpc_helper_;
  std::shared_ptr<ClientHostBufferStore> host_buffer_store_;
};

TEST_F(LoadedExecutableTest, Metadata) {
  TestQueue<IfrtRequest> requests_queue(/*pop_timeout=*/absl::Minutes(1));
  IfrtResponse response;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(
        loaded_executable_metadata_response {
          parameter_shardings {
            shardings { type: REPLICATED }
            shardings {
              type: OTHER
              tile_shape {
                element_type: BF16
                dimensions: [ 2, 2 ]
              }
              tile_assignment_dimensions: [ 0, 1 ]
            }
          }
          output_shardings { shardings { type: REPLICATED } }
          parameter_layouts_list {
            layouts { minor_to_major: 0 }
            layouts { minor_to_major: [ 1, 0 ] }
          }
          output_layouts_list { layouts { minor_to_major: [ 1, 0 ] } }
          output_memory_kinds { memory_kind_lists { memory_kinds: [ "foo" ] } }
        }
      )pb",
      &response));

  EXPECT_CALL(
      *session_,
      Enqueue(IfrtRequestOfType(IfrtRequest::kLoadedExecutableMetadataRequest)))
      .WillOnce(MockClientCaptureAndReturn(&requests_queue, response));

  MockClient client;
  MockDevice device1;
  ON_CALL(device1, Id()).WillByDefault(Return(DeviceId(1)));
  MockDevice device2;
  ON_CALL(device2, Id()).WillByDefault(Return(DeviceId(2)));
  LoadedExecutable executable(
      &client, rpc_helper_, /*handle=*/1234, /*name=*/"foo",
      /*num_devices=*/2, /*devices=*/{},
      /*addressable_devices=*/{},
      /*fingerprint=*/"fingerprint",
      /*loaded_host_callbacks=*/{}, /*loaded_host_callback_handles=*/{});

  EXPECT_EQ(requests_queue.Pop()
                .loaded_executable_metadata_request()
                .loaded_executable_handle(),
            1234);
  if (executable.GetParameterShardings().has_value()) {
    std::vector<OpSharding> param_shardings =
        *std::move(executable.GetParameterShardings());
    ASSERT_EQ(param_shardings.size(), 2);
    EXPECT_EQ(param_shardings[0].type(), OpSharding::REPLICATED);
    ASSERT_EQ(param_shardings[1].type(), OpSharding::OTHER);
    EXPECT_EQ(param_shardings[1].tile_shape().element_type(), xla::BF16);
    EXPECT_THAT(param_shardings[1].tile_shape().dimensions(),
                ElementsAre(2, 2));
    EXPECT_THAT(param_shardings[1].tile_assignment_dimensions(),
                ElementsAre(0, 1));
  }
  if (executable.GetOutputShardings().has_value()) {
    std::vector<OpSharding> output_shardings =
        *std::move(executable.GetOutputShardings());
    ASSERT_EQ(output_shardings.size(), 1);
    EXPECT_EQ(output_shardings[0].type(), OpSharding::REPLICATED);
  }
  TF_ASSERT_OK_AND_ASSIGN(auto parameter_layouts,
                          executable.GetParameterLayouts());
  ASSERT_EQ(parameter_layouts.size(), 2);
  EXPECT_EQ(parameter_layouts[0]->xla_layout(),
            xla::LayoutUtil::MakeDescendingLayout(/*num_dims=*/1));
  EXPECT_EQ(parameter_layouts[1]->xla_layout(),
            xla::LayoutUtil::MakeDescendingLayout(/*num_dims=*/2));
  TF_ASSERT_OK_AND_ASSIGN(auto output_layouts, executable.GetOutputLayouts());
  ASSERT_EQ(output_layouts.size(), 1);
  EXPECT_EQ(output_layouts[0]->xla_layout(),
            xla::LayoutUtil::MakeDescendingLayout(/*num_dims=*/2));
  EXPECT_THAT(executable.GetOutputMemoryKinds(),
              absl_testing::IsOkAndHolds(ElementsAre(ElementsAre("foo"))));
}

TEST_F(LoadedExecutableTest, Execute) {
  MockClient client;

  IfrtResponse response;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(
        loaded_executable_metadata_response {
          parameter_shardings {
            shardings { type: REPLICATED }
            shardings { type: REPLICATED }
          }
          output_shardings {
            shardings { type: REPLICATED }
            shardings { type: REPLICATED }
          }
          output_layouts_list {
            layouts { minor_to_major: [ 1, 0 ] }
            layouts { minor_to_major: 0 }
          }
        }
      )pb",
      &response));
  EXPECT_CALL(*session_, Enqueue(Pointee(Partially(EquivToProto(
                             R"pb(loaded_executable_metadata_request {
                                    loaded_executable_handle: 1234
                                  })pb")))))
      .WillOnce(MockClientSessionReturnResponse(response));

  ON_CALL(client, MakeDeviceList(_))
      .WillByDefault([](absl::Span<xla::ifrt::Device* const> devices) {
        return xla::ifrt::BasicDeviceList::Create(devices);
      });

  MockDevice device;
  ON_CALL(device, client()).WillByDefault(Return(&client));
  ON_CALL(device, Id()).WillByDefault(Return(DeviceId(1)));
  ON_CALL(client, LookupDevice(DeviceId(1))).WillByDefault(Return(&device));

  LoadedExecutable executable(
      &client, rpc_helper_, /*handle=*/1234, /*name=*/"foo",
      /*num_devices=*/2, /*devices=*/{}, /*addressable_devices=*/{},
      /*fingerprint=*/"fingerprint",
      /*loaded_host_callbacks=*/{}, /*loaded_host_callback_handles=*/{});

  xla::ifrt::LoadedExecutable::ExecuteOptions exec_options;
  exec_options.fill_status = true;

  IfrtResponse execute_response;
  IfrtResponse fetch_execute_result_response;

  ASSERT_TRUE(TextFormat::ParseFromString(R"pb(
                                            loaded_executable_execute_response {
                                              status_handle: 2000
                                              outputs {
                                                dtype { kind: KIND_F32 }
                                                shape { dims: [ 4, 4 ] }
                                                array_handle: 3000
                                              }
                                              outputs {
                                                dtype { kind: KIND_F16 }
                                                shape { dims: [ 8 ] }
                                                array_handle: 3001
                                              }
                                            }
                                          )pb",
                                          &execute_response));
  {
    auto* outputs =
        execute_response.mutable_loaded_executable_execute_response()
            ->mutable_outputs();
    TF_ASSERT_OK(SingleDeviceSharding::Create(&device, MemoryKind())
                     ->ToProto(*(*outputs)[0].mutable_sharding(),
                               rpc_helper_->ifrt_serdes_version()));
    TF_ASSERT_OK(SingleDeviceSharding::Create(&device, MemoryKind())
                     ->ToProto(*(*outputs)[1].mutable_sharding(),
                               rpc_helper_->ifrt_serdes_version()));
  }
  EXPECT_CALL(*session_, Enqueue(Pointee(Partially(EquivToProto(
                             R"pb(loaded_executable_execute_request {
                                    loaded_executable_handle: 1234
                                    args_handles: [ 1000, 1001 ]
                                    device_ids: [ 1 ]
                                  })pb")))))
      .WillOnce(MockClientSessionReturnResponse(execute_response));

  ASSERT_TRUE(TextFormat::ParseFromString(R"pb(
                                            response_metadata {
                                              status {
                                                code: 2  # UNKNOWN
                                                message: "injected error"
                                              }
                                            }
                                          )pb",
                                          &fetch_execute_result_response));
  EXPECT_CALL(*session_,
              Enqueue(Pointee(Partially(EquivToProto(
                  R"pb(loaded_executable_fetch_execute_result_request {
                         result_status_handle: 2000
                       })pb")))))
      .WillOnce(MockClientSessionReturnResponse(fetch_execute_result_response));

  DeviceListRef devices = BasicDeviceList::Create({&device});

  std::vector<xla::ifrt::ArrayRef> args;
  for (const uint64_t handle : {1000, 1001}) {
    args.push_back(tsl::MakeRef<Array>(
        &client, rpc_helper_, DType(DType::kF32), Shape({2, 2}),
        OpaqueSharding::Create(devices, MemoryKind()), ArrayHandle{handle},
        /*layout=*/nullptr));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      executable.Execute(absl::MakeSpan(args), exec_options, devices));

  EXPECT_THAT(
      result.status.Await(),
      absl_testing::StatusIs(absl::StatusCode::kUnknown, "injected error"));

  ASSERT_THAT(result.outputs, SizeIs(2));

  const auto output0 = result.outputs[0];
  EXPECT_EQ(output0->dtype(), DType(DType::kF32));
  EXPECT_EQ(output0->shape(), Shape({4, 4}));
  EXPECT_EQ(llvm::cast<Array>(output0.get())
                ->GetHandleUnknownIfBeingDonated()
                ->handle,
            3000);

  const auto output1 = result.outputs[1];
  EXPECT_EQ(output1->dtype(), DType(DType::kF16));
  EXPECT_EQ(output1->shape(), Shape({8}));
  EXPECT_EQ(llvm::cast<Array>(output1.get())
                ->GetHandleUnknownIfBeingDonated()
                ->handle,
            3001);

  // Execute again. This time, the client already knows the output spec and so
  // will supply client-generated handles.
  execute_response.mutable_loaded_executable_execute_response()
      ->clear_outputs();
  execute_response.mutable_loaded_executable_execute_response()
      ->set_status_handle(0);
  TestQueue<IfrtRequest> requests_queue(/*pop_timeout=*/absl::Minutes(1));

  EXPECT_CALL(
      *session_,
      Enqueue(IfrtRequestOfType(IfrtRequest::kLoadedExecutableExecuteRequest)))
      .WillOnce(MockClientCaptureAndReturn(&requests_queue, execute_response));
  EXPECT_CALL(*session_,
              Enqueue(IfrtRequestOfType(
                  IfrtRequest::kLoadedExecutableFetchExecuteResultRequest)))
      .WillOnce(MockClientCaptureAndReturn(&requests_queue,
                                           fetch_execute_result_response));

  TF_ASSERT_OK_AND_ASSIGN(
      result, executable.Execute(absl::MakeSpan(args), exec_options, devices));

  auto execute_req = requests_queue.Pop().loaded_executable_execute_request();
  auto fetch_execute_result_req =
      requests_queue.Pop().loaded_executable_fetch_execute_result_request();

  EXPECT_THAT(
      result.status.Await(),
      absl_testing::StatusIs(absl::StatusCode::kUnknown, "injected error"));
  EXPECT_EQ(execute_req.result_status_handle(),
            fetch_execute_result_req.result_status_handle());

  ASSERT_THAT(result.outputs, SizeIs(2));
  ASSERT_THAT(execute_req.result_array_handle(), SizeIs(2));
  EXPECT_EQ(llvm::cast<Array>(result.outputs[0].get())
                ->GetHandleUnknownIfBeingDonated()
                ->handle,
            execute_req.result_array_handle()[0]);
  EXPECT_EQ(llvm::cast<Array>(result.outputs[1].get())
                ->GetHandleUnknownIfBeingDonated()
                ->handle,
            execute_req.result_array_handle()[1]);
}

TEST_F(LoadedExecutableTest, DeviceTime) {
  if (tsl::kIsOpenSource) {
    GTEST_SKIP()
        << "DeviceTimeMeasurement implementation isn't available in OSS.";
  }

  MockClient client;

  IfrtResponse response;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(
        loaded_executable_metadata_response {
          parameter_shardings {}
          output_shardings {}
          output_layouts_list {}
        }
      )pb",
      &response));
  EXPECT_CALL(*session_, Enqueue(Pointee(Partially(EquivToProto(
                             R"pb(loaded_executable_metadata_request {
                                    loaded_executable_handle: 1234
                                  })pb")))))
      .WillOnce(MockClientSessionReturnResponse(response));

  LoadedExecutable executable(
      &client, rpc_helper_, /*handle=*/1234, /*name=*/"foo",
      /*num_devices=*/1, /*devices=*/{}, /*addressable_devices=*/{},
      /*fingerprint=*/"fingerprint",
      /*loaded_host_callbacks=*/{}, /*loaded_host_callback_handles=*/{});

  xla::ifrt::LoadedExecutable::ExecuteOptions exec_options;
  exec_options.fill_status = true;

  IfrtResponse execute_response;
  IfrtResponse fetch_execute_result_response;

  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(
        loaded_executable_execute_response { status_handle: 2000 }
      )pb",
      &execute_response));
  EXPECT_CALL(*session_, Enqueue(Pointee(Partially(EquivToProto(
                             R"pb(loaded_executable_execute_request {
                                    loaded_executable_handle: 1234
                                  })pb")))))
      .WillOnce(MockClientSessionReturnResponse(execute_response));

  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(
        loaded_executable_fetch_execute_result_response {
          device_time { key: "tpu" value: 1234.0 }
        }
      )pb",
      &fetch_execute_result_response));
  EXPECT_CALL(*session_,
              Enqueue(Pointee(Partially(EquivToProto(
                  R"pb(loaded_executable_fetch_execute_result_request {
                         result_status_handle: 2000
                       })pb")))))
      .WillOnce(MockClientSessionReturnResponse(fetch_execute_result_response));

  auto device_time = xla::CreateDeviceTimeMeasurement();

  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          executable.Execute({}, exec_options, std::nullopt));
  EXPECT_OK(result.status.Await());

  EXPECT_THAT(device_time->GetTotalDuration(
                  xla::DeviceTimeMeasurement::DeviceType::kTpu),
              absl::Microseconds(1234.0));
}

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
