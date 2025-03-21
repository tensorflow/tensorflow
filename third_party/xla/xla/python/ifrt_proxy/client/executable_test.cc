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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "xla/layout_util.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
#include "xla/python/ifrt/basic_device_list.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/future.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt_proxy/client/array.h"
#include "xla/python/ifrt_proxy/client/client_session.h"
#include "xla/python/ifrt_proxy/client/host_buffer.h"
#include "xla/python/ifrt_proxy/client/mock_client_session.h"
#include "xla/python/ifrt_proxy/client/mock_host_buffer.h"
#include "xla/python/ifrt_proxy/client/rpc_helper.h"
#include "xla/python/ifrt_proxy/client/version.h"
#include "xla/python/ifrt_proxy/common/ifrt_service.pb.h"
#include "xla/python/ifrt_proxy/common/test_utils.h"
#include "xla/python/ifrt_proxy/common/types.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::Optional;
using ::testing::Pointee;
using ::testing::Return;
using ::testing::SizeIs;
using ::testing::StrEq;
using ::tsl::protobuf::TextFormat;
using ::tsl::testing::IsOkAndHolds;
using ::tsl::testing::StatusIs;

#if defined(PLATFORM_GOOGLE)
using ::testing::EquivToProto;
using ::testing::proto::Partially;
#endif

namespace xla {
namespace ifrt {
namespace proxy {
namespace {

IfrtProxyVersion Version() {
  IfrtProxyVersion version;
  version.set_protocol_version(kClientMaxVersion);
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
        .WillRepeatedly(Return(Future<ClientSession::Response>(
            absl::InternalError("Request has no mock handlers"))));
  }

  std::shared_ptr<MockClientSession> session_;
  std::shared_ptr<RpcHelper> rpc_helper_;
  std::shared_ptr<ClientHostBufferStore> host_buffer_store_;
};

// TODO(b/315809436): Test needs rewrite because protobuf matchers are not OSS
#if defined(PLATFORM_GOOGLE)
TEST_F(LoadedExecutableTest, Metadata) {
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
  EXPECT_CALL(*session_, Enqueue(Pointee(Partially(EquivToProto(
                             R"pb(loaded_executable_metadata_request {
                                    loaded_executable_handle: 1234
                                  })pb")))))
      .WillOnce(MockClientSessionReturnResponse(response));

  MockClient client;
  LoadedExecutable executable(
      &client, rpc_helper_, /*handle=*/1234, /*name=*/"foo",
      /*num_devices=*/2, /*addressable_devices=*/{},
      /*fingerprint=*/"fingerprint",
      /*ready_future=*/Future<>(absl::OkStatus()),
      /*loaded_host_callbacks=*/{}, /*loaded_host_callback_handles=*/{});

  EXPECT_THAT(
      executable.GetParameterShardings(),
      Optional(ElementsAre(
          EquivToProto(R"pb(type: REPLICATED)pb"),
          EquivToProto(R"pb(type: OTHER
                            tile_shape {
                              element_type: BF16
                              dimensions: [ 2, 2 ]
                            }
                            tile_assignment_dimensions: [ 0, 1 ])pb"))));
  EXPECT_THAT(executable.GetOutputShardings(),
              Optional(ElementsAre(EquivToProto(R"pb(type: REPLICATED)pb"))));
  ASSERT_OK_AND_ASSIGN(auto parameter_layouts,
                       executable.GetParameterLayouts());
  ASSERT_EQ(parameter_layouts.size(), 2);
  EXPECT_EQ(parameter_layouts[0]->xla_layout(),
            xla::LayoutUtil::MakeDescendingLayout(/*rank=*/1));
  EXPECT_EQ(parameter_layouts[1]->xla_layout(),
            xla::LayoutUtil::MakeDescendingLayout(/*rank=*/2));
  ASSERT_OK_AND_ASSIGN(auto output_layouts, executable.GetOutputLayouts());
  ASSERT_EQ(output_layouts.size(), 1);
  EXPECT_EQ(output_layouts[0]->xla_layout(),
            xla::LayoutUtil::MakeDescendingLayout(/*rank=*/2));
  EXPECT_THAT(executable.GetOutputMemoryKinds(),
              IsOkAndHolds(ElementsAre(ElementsAre("foo"))));
}
#endif

// TODO(b/315809436): Test needs rewrite because protobuf matchers are not OSS
#if defined(PLATFORM_GOOGLE)
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
      /*num_devices=*/2, /*addressable_devices=*/{},
      /*fingerprint=*/"fingerprint",
      /*ready_future=*/Future<>(absl::OkStatus()),
      /*loaded_host_callbacks=*/{}, /*loaded_host_callback_handles=*/{});

  xla::ifrt::LoadedExecutable::ExecuteOptions exec_options;
  exec_options.fill_status = true;

  IfrtResponse execute_response;
  IfrtResponse check_future_response;

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
    TF_ASSERT_OK_AND_ASSIGN(
        *(*outputs)[0].mutable_sharding(),
        SingleDeviceSharding::Create(&device, MemoryKind())->ToProto());
    TF_ASSERT_OK_AND_ASSIGN(
        *(*outputs)[1].mutable_sharding(),
        SingleDeviceSharding::Create(&device, MemoryKind())->ToProto());
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
                                          &check_future_response));
  EXPECT_CALL(*session_,
              Enqueue(Pointee(Partially(EquivToProto(R"pb(check_future_request {
                                                            future_handle: 2000
                                                          })pb")))))
      .WillOnce(MockClientSessionReturnResponse(check_future_response));

  DeviceListRef devices = BasicDeviceList::Create({&device});

  std::vector<tsl::RCReference<xla::ifrt::Array>> args;
  for (const uint64_t handle : {1000, 1001}) {
    args.push_back(tsl::MakeRef<Array>(
        &client, rpc_helper_, DType(DType::kF32), Shape({2, 2}),
        OpaqueSharding::Create(devices, MemoryKind()), ArrayHandle{handle}));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto result,
      executable.Execute(absl::MakeSpan(args), exec_options, devices));

  EXPECT_THAT(result.status.Await(),
              StatusIs(absl::StatusCode::kUnknown, "injected error"));

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
              Enqueue(IfrtRequestOfType(IfrtRequest::kCheckFutureRequest)))
      .WillOnce(
          MockClientCaptureAndReturn(&requests_queue, check_future_response));

  TF_ASSERT_OK_AND_ASSIGN(
      result, executable.Execute(absl::MakeSpan(args), exec_options, devices));

  auto execute_req = requests_queue.Pop().loaded_executable_execute_request();
  auto check_future_req = requests_queue.Pop().check_future_request();

  EXPECT_THAT(result.status.Await(),
              StatusIs(absl::StatusCode::kUnknown, "injected error"));
  EXPECT_EQ(execute_req.result_status_handle(),
            check_future_req.future_handle());

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
#endif

// TODO(b/315809436): Test needs rewrite because protobuf matchers are not OSS
#if defined(PLATFORM_GOOGLE)
TEST_F(LoadedExecutableTest, Delete) {
  MockClient client;
  LoadedExecutable executable(
      &client, rpc_helper_, /*handle=*/1234, /*name=*/"foo",
      /*num_devices=*/2, /*addressable_devices=*/{},
      /*fingerprint=*/"fingerprint",
      /*ready_future=*/Future<>(absl::OkStatus()),
      /*loaded_host_callbacks=*/{}, /*loaded_host_callback_handles=*/{});

  {
    IfrtResponse response;
    ASSERT_TRUE(TextFormat::ParseFromString(
        R"pb(
          loaded_executable_delete_response { future_handle: 2000 }
        )pb",
        &response));
    EXPECT_CALL(*session_, Enqueue(Pointee(Partially(EquivToProto(
                               R"pb(loaded_executable_delete_request {
                                      loaded_executable_handle: 1234
                                    })pb")))))
        .WillOnce(MockClientSessionReturnResponse(response));

    ASSERT_TRUE(TextFormat::ParseFromString(
        R"pb(
          response_metadata {
            status {
              code: 2  # UNKNOWN
              message: "injected error"
            }
          }
        )pb",
        &response));
    EXPECT_CALL(
        *session_,
        Enqueue(Pointee(Partially(EquivToProto(R"pb(check_future_request {
                                                      future_handle: 2000
                                                    })pb")))))
        .WillOnce(MockClientSessionReturnResponse(response));

    Future<> result = executable.Delete();
    EXPECT_THAT(result.Await(),
                StatusIs(absl::StatusCode::kUnknown, StrEq("injected error")));
  }

  {
    IfrtResponse response;
    ASSERT_TRUE(TextFormat::ParseFromString(
        R"pb(
          loaded_executable_is_deleted_response { is_deleted: true }
        )pb",
        &response));
    EXPECT_CALL(*session_, Enqueue(Pointee(Partially(EquivToProto(
                               R"pb(loaded_executable_is_deleted_request {
                                      loaded_executable_handle: 1234
                                    })pb")))))
        .WillOnce(MockClientSessionReturnResponse(response));

    EXPECT_TRUE(executable.IsDeleted());
  }

  IfrtResponse response;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(
        loaded_executable_destruct_response {}
      )pb",
      &response));
  EXPECT_CALL(*session_, Enqueue(Pointee(Partially(EquivToProto(
                             R"pb(loaded_executable_destruct_request {
                                    loaded_executable_handle: 1234
                                  })pb")))))
      .WillOnce(MockClientSessionReturnResponse(response));
}
#endif

}  // namespace
}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
