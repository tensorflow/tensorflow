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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/tests/multi_process_gpu_test_base.h"
#include "xla/client/client_library.h"
#include "xla/future.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/c/pjrt_c_api_gpu.h"
#include "xla/pjrt/c/pjrt_c_api_helpers.h"
#include "xla/pjrt/c/pjrt_c_api_status_utils.h"
#include "xla/pjrt/c/pjrt_c_api_wrapper_impl.h"
#include "xla/pjrt/extensions/cross_host_transfers/pjrt_c_api_cross_host_transfers_extension.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/xla_data.pb.h"

namespace pjrt {
namespace {

std::string SuccessfulCrossHostTransferTestName(
    const ::testing::TestParamInfo<int>& info) {
  return absl::StrFormat("num_arrays_%d", info.param);
}

absl::StatusOr<PJRT_Client_Create_Args> BuildCreateArg(
    ::pjrt::PJRT_KeyValueCallbackData* kv_callback_data,
    const std::vector<PJRT_NamedValue>& c_options) {
  PJRT_Client_Create_Args args{};
  args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  args.create_options = c_options.data();
  args.num_options = c_options.size();
  args.kv_get_callback = kv_callback_data->c_kv_get;
  args.kv_get_user_arg = &kv_callback_data->kv_get_c_func;
  args.kv_put_callback = kv_callback_data->c_kv_put;
  args.kv_put_user_arg = &kv_callback_data->kv_put_c_func;
  args.kv_try_get_user_arg = &kv_callback_data->kv_try_get_c_func;
  args.kv_try_get_callback = kv_callback_data->c_kv_try_get;
  return args;
}

absl::Span<PJRT_Device* const> GetClientAddressableDevices(
    PJRT_Client* client, const PJRT_Api* api) {
  PJRT_Client_AddressableDevices_Args addr_args{};
  addr_args.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
  addr_args.client = client;
  PJRT_Error* error = api->PJRT_Client_AddressableDevices(&addr_args);
  CHECK(error == nullptr);
  return absl::MakeSpan(addr_args.addressable_devices,
                        addr_args.num_addressable_devices);
}

using SuccessfulCrossHostTransferTest = xla::MultiProcessGpuTestWithParam<int>;

TEST_P(SuccessfulCrossHostTransferTest, SuccessfulCrossHostTransfer) {
  const int num_arrays = GetParam();
  const bool is_sender = (rank() == 0);

  auto kv_store = CreateScopedKeyValueStore("c_api");
  std::shared_ptr<::pjrt::PJRT_KeyValueCallbackData> kv_callback_data =
      ::pjrt::ConvertToCKeyValueCallbacks(kv_store);
  xla::ClientLibrary::DestroyLocalInstances();

  const PJRT_Api* api = GetPjrtApi();
  PJRT_CrossHostTransfers_Extension* cross_host_transfers_extension =
      pjrt::FindExtension<PJRT_CrossHostTransfers_Extension>(
          api, PJRT_Extension_Type::PJRT_Extension_Type_CrossHostTransfers);
  ASSERT_NE(cross_host_transfers_extension, nullptr);
  ASSERT_NE(cross_host_transfers_extension
                ->PJRT_Transfers_PJRT_Buffer_CopyToRemoteDevice,
            nullptr);

  // Create the GPU client.
  absl::flat_hash_map<std::string, xla::PjRtValueType> options = {
      {"num_nodes", static_cast<int64_t>(num_ranks())},
      {"node_id", static_cast<int64_t>(rank())},
      {"visible_devices", std::vector<int64_t>({rank()})}};
  ASSERT_OK_AND_ASSIGN(std::vector<PJRT_NamedValue> c_options,
                       ::pjrt::ConvertToPjRtNamedValueList(options));
  ASSERT_OK_AND_ASSIGN(PJRT_Client_Create_Args create_arg,
                       BuildCreateArg(kv_callback_data.get(), c_options));
  std::unique_ptr<PJRT_Error, ::pjrt::PJRT_ErrorDeleter> error(
      api->PJRT_Client_Create(&create_arg), ::pjrt::MakeErrorDeleter(api));
  ASSERT_EQ(error, nullptr) << PjrtErrorToStatus(error.get(), api);
  std::unique_ptr<PJRT_Client, ::pjrt::PJRT_ClientDeleter> client_deleter(
      create_arg.client, ::pjrt::MakeClientDeleter(api));

  std::vector<int64_t> shape = {2, 3};
  xla::Shape xla_shape =
      xla::ShapeUtil::MakeShape(xla::F32, /*dimensions=*/shape);

  if (is_sender) {
    std::vector<PJRT_Buffer*> raw_buffers;
    std::vector<xla::GlobalDeviceId> dst_device_ids;
    std::vector<xla::CrossHostTransferKey> transfer_keys;
    raw_buffers.reserve(num_arrays);
    dst_device_ids.reserve(num_arrays);
    transfer_keys.reserve(num_arrays);
    for (int i = 0; i < num_arrays; ++i) {
      std::vector<float> data = {1, 2, 3, 4, 5, 6 * static_cast<float>(i)};
      PJRT_Client_BufferFromHostBuffer_Args args{};
      args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
      args.data = data.data();
      args.type = ::pjrt::ConvertToPjRtBufferType(xla_shape.element_type());
      args.dims = xla_shape.dimensions().data();
      args.num_dims = xla_shape.dimensions().size();
      args.host_buffer_semantics = ::pjrt::ConvertToPjRtHostBufferSemantics(
          xla::PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall);
      args.client = create_arg.client;
      args.device = GetClientAddressableDevices(create_arg.client, api)[0];

      std::unique_ptr<PJRT_Error, ::pjrt::PJRT_ErrorDeleter> transfer_error(
          api->PJRT_Client_BufferFromHostBuffer(&args),
          ::pjrt::MakeErrorDeleter(api));
      ASSERT_EQ(transfer_error, nullptr)
          << PjrtErrorToStatus(transfer_error.get(), api);
      ASSERT_OK(args.buffer->buffer->GetReadyFuture().Await());
      std::unique_ptr<PJRT_Event, PJRT_EventDeleter> event(
          args.done_with_host_buffer, MakeEventDeleter(api));

      raw_buffers.push_back(args.buffer);
      ASSERT_OK(event->future.Await());
      xla::GlobalDeviceId src_device_id =
          args.device->device->global_device_id();
      dst_device_ids.push_back(1 - src_device_id);
      transfer_keys.push_back(xla::CrossHostTransferKey(i));
    }

    PJRT_Transfers_PJRT_Client_CrossHostSendBuffers_Args send_args{};
    send_args.struct_size =
        PJRT_Transfers_PJRT_Client_CrossHostSendBuffers_Args_STRUCT_SIZE;
    send_args.client = create_arg.client;
    send_args.num_buffers = raw_buffers.size();
    send_args.buffers = raw_buffers.data();
    send_args.dst_global_device_ids = dst_device_ids.data();
    send_args.transfer_keys = transfer_keys.data();
    std::vector<PJRT_Event*> temp_events(raw_buffers.size());
    send_args.send_events = temp_events.data();
    cross_host_transfers_extension
        ->PJRT_Transfers_PJRT_Client_CrossHostSendBuffers(&send_args);

    for (int i = 0; i < num_arrays; ++i) {
      ASSERT_OK(send_args.send_events[i]->future.Await());
      std::unique_ptr<PJRT_Buffer, ::pjrt::PJRT_BufferDeleter> buffer(
          raw_buffers[i], ::pjrt::MakeBufferDeleter(api));
      std::unique_ptr<PJRT_Event, PJRT_EventDeleter> send_event(
          send_args.send_events[i], MakeEventDeleter(api));
      ASSERT_OK(send_event->future.Await());
    }
  } else {
    std::vector<xla::Literal> expected_literals;
    expected_literals.reserve(num_arrays);
    for (int i = 0; i < num_arrays; ++i) {
      expected_literals.push_back(xla::LiteralUtil::CreateR2<float>(
          {{1, 2, 3}, {4, 5, 6 * static_cast<float>(i)}}));
    }
    std::vector<xla::Shape> shapes;
    std::vector<xla::GlobalDeviceId> src_device_ids;
    std::vector<xla::CrossHostTransferKey> transfer_keys;
    std::vector<size_t> shape_num_dims;
    std::vector<const int64_t*> num_dims;
    std::vector<PJRT_Buffer_Type> element_types;
    std::vector<PJRT_Buffer_MemoryLayout*> layouts;
    shapes.reserve(num_arrays);
    src_device_ids.reserve(num_arrays);
    transfer_keys.reserve(num_arrays);
    shape_num_dims.reserve(num_arrays);
    num_dims.reserve(num_arrays);
    element_types.reserve(num_arrays);
    layouts.reserve(num_arrays);
    xla::GlobalDeviceId dst_device_id =
        GetClientAddressableDevices(create_arg.client, api)[0]
            ->device->global_device_id();
    for (int i = 0; i < num_arrays; ++i) {
      shapes.push_back(xla_shape);
      src_device_ids.push_back(xla::GlobalDeviceId(1 - dst_device_id));
      transfer_keys.push_back(xla::CrossHostTransferKey(i));
      shape_num_dims.push_back(shapes.back().dimensions().size());
      num_dims.push_back(shapes.back().dimensions().data());
      element_types.push_back(
          ::pjrt::ConvertToPjRtBufferType(shapes.back().element_type()));
      layouts.push_back(nullptr);
    }

    PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers_Args recv_args{};
    recv_args.struct_size =
        PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers_Args_STRUCT_SIZE;
    recv_args.client = create_arg.client;
    recv_args.num_shapes = shapes.size();
    recv_args.shape_num_dims = shape_num_dims.data();
    recv_args.num_dims = num_dims.data();
    recv_args.element_types = element_types.data();
    recv_args.layouts = layouts.data();
    recv_args.device = GetClientAddressableDevices(create_arg.client, api)[0];
    recv_args.src_global_device_ids = src_device_ids.data();
    recv_args.transfer_keys = transfer_keys.data();
    std::vector<PJRT_Buffer*> temp_buffers(shapes.size());
    recv_args.buffers = temp_buffers.data();
    cross_host_transfers_extension
        ->PJRT_Transfers_PJRT_Client_CrossHostReceiveBuffers(&recv_args);

    for (int i = 0; i < num_arrays; ++i) {
      ASSERT_OK(recv_args.buffers[i]->buffer->GetReadyFuture().Await());
      ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::Literal> recv_literal,
                           recv_args.buffers[i]->buffer->ToLiteral().Await());

      EXPECT_TRUE(
          xla::LiteralTestUtil::Equal(expected_literals[i], *recv_literal));
      std::unique_ptr<PJRT_Buffer, ::pjrt::PJRT_BufferDeleter> buffer(
          recv_args.buffers[i], ::pjrt::MakeBufferDeleter(api));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(SuccessfulCrossHostTransfer,
                         SuccessfulCrossHostTransferTest,
                         ::testing::ValuesIn({1, 2, 3}),
                         SuccessfulCrossHostTransferTestName);

}  // namespace
}  // namespace pjrt
