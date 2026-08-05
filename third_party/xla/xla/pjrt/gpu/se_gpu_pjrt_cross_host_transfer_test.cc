/* Copyright 2026 The OpenXLA Authors.

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
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/notification.h"
#include "xla/future.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client_test_helper.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;

TEST(StreamExecutorGpuClientTest, CrossHostTransferSendRecv) {
  GpuClientOptions options = GetTestGpuClientOptions(2);
  ASSERT_OK_AND_ASSIGN(auto client, GetStreamExecutorGpuClient(options));
  ASSERT_GE(client->devices().size(), 2);

  auto* const src_device = client->addressable_devices()[0];
  auto* const dst_device = client->addressable_devices()[1];

  Literal input_literal0 =
      LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  Literal input_literal1 =
      LiteralUtil::CreateR1<float>({5.0f, 6.0f, 7.0f, 8.0f});
  ASSERT_OK_AND_ASSIGN(auto* const src_memory_space,
                       src_device->default_memory_space());
  ASSERT_OK_AND_ASSIGN(auto src_buffer0, client->BufferFromHostLiteral(
                                             input_literal0, src_memory_space));
  ASSERT_OK_AND_ASSIGN(auto src_buffer1, client->BufferFromHostLiteral(
                                             input_literal1, src_memory_space));

  std::vector<Promise<std::string>> desc_promises;
  std::vector<Future<std::string>> desc_futures;
  for (int i = 0; i < 2; ++i) {
    std::tie(desc_promises.emplace_back(), desc_futures.emplace_back()) =
        MakePromise<std::string>();
  }

  ASSERT_OK_AND_ASSIGN(
      auto buffers,
      client->MakeCrossHostReceiveBuffers(
          {input_literal0.shape(), input_literal1.shape()}, dst_device,
          [&](absl::StatusOr<PjRtCrossHostRecvState> descriptors) {
            CHECK_OK(descriptors);
            CHECK_EQ(descriptors->descriptors.size(), desc_promises.size());

            for (int i = 0; i < descriptors->descriptors.size(); ++i) {
              const auto& desc = descriptors->descriptors[i];
              CHECK_EQ(desc.serialized_descriptors.size(), 1);
              desc_promises[i].Set(desc.serialized_descriptors[0]);
            }
          }));
  ASSERT_EQ(buffers.size(), 2);

  src_buffer0->CopyToRemoteDevice(desc_futures[0],
                                  [&](absl::Status s, bool dispatched) {
                                    ASSERT_TRUE(dispatched);
                                    ASSERT_OK(s);
                                  });
  src_buffer1->CopyToRemoteDevice(desc_futures[1],
                                  [&](absl::Status s, bool dispatched) {
                                    ASSERT_TRUE(dispatched);
                                    ASSERT_OK(s);
                                  });

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> recv_literal0,
                       buffers[0]->ToLiteral().Await());
  EXPECT_TRUE(LiteralTestUtil::Equal(input_literal0, *recv_literal0));

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> recv_literal1,
                       buffers[1]->ToLiteral().Await());
  EXPECT_TRUE(LiteralTestUtil::Equal(input_literal1, *recv_literal1));
}

TEST(StreamExecutorGpuClientTest, CrossHostTransferCancellation) {
  GpuClientOptions options = GetTestGpuClientOptions(2);
  ASSERT_OK_AND_ASSIGN(auto client, GetStreamExecutorGpuClient(options));
  ASSERT_GE(client->devices().size(), 2);

  auto* const dst_device = client->addressable_devices()[1];

  Shape shape = ShapeUtil::MakeShape(F32, {4});
  absl::Notification done;
  PjRtCrossHostRecvState recv_state;
  ASSERT_OK_AND_ASSIGN(
      auto buffers,
      client->MakeCrossHostReceiveBuffers(
          {shape}, dst_device,
          [&](absl::StatusOr<PjRtCrossHostRecvState> descriptors) {
            CHECK_OK(descriptors.status());
            recv_state = *std::move(descriptors);
            done.Notify();
          }));
  done.WaitForNotification();

  ASSERT_EQ(recv_state.descriptors.size(), 1);
  ASSERT_NE(recv_state.cancel_notifier, nullptr);

  absl::Notification cancel_done;
  absl::Status cancel_result;
  recv_state.cancel_notifier(
      recv_state.descriptors[0].serialized_descriptors[0],
      absl::CancelledError("test cancellation"), [&](absl::Status s) {
        cancel_result = s;
        cancel_done.Notify();
      });
  cancel_done.WaitForNotification();
  EXPECT_OK(cancel_result);

  ASSERT_EQ(buffers.size(), 1);
  auto literal_or = buffers[0]->ToLiteral().Await();
  EXPECT_THAT(literal_or.status(), StatusIs(absl::StatusCode::kCancelled));
}

TEST(StreamExecutorGpuClientTest, CrossHostTransferPartialCancellation) {
  GpuClientOptions options = GetTestGpuClientOptions(2);
  ASSERT_OK_AND_ASSIGN(auto client, GetStreamExecutorGpuClient(options));
  ASSERT_GE(client->devices().size(), 2);

  auto* const src_device = client->addressable_devices()[0];
  auto* const dst_device = client->addressable_devices()[1];

  Shape shape = ShapeUtil::MakeShape(F32, {4});
  absl::Notification done;
  PjRtCrossHostRecvState recv_state;
  ASSERT_OK_AND_ASSIGN(
      auto buffers,
      client->MakeCrossHostReceiveBuffers(
          {shape, shape}, dst_device,
          [&](absl::StatusOr<PjRtCrossHostRecvState> descriptors) {
            CHECK_OK(descriptors.status());
            recv_state = *std::move(descriptors);
            done.Notify();
          }));
  done.WaitForNotification();

  ASSERT_EQ(recv_state.descriptors.size(), 2);
  ASSERT_NE(recv_state.cancel_notifier, nullptr);

  Literal input_literal =
      LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  ASSERT_OK_AND_ASSIGN(auto* const src_memory_space,
                       src_device->default_memory_space());
  ASSERT_OK_AND_ASSIGN(auto src_buffer, client->BufferFromHostLiteral(
                                            input_literal, src_memory_space));
  src_buffer->CopyToRemoteDevice(
      recv_state.descriptors[0].serialized_descriptors[0],
      [&](absl::Status s, bool dispatched) {
        CHECK(dispatched);
        CHECK_OK(s);
      });

  absl::Notification cancel_done;
  absl::Status cancel_result;
  recv_state.cancel_notifier(
      recv_state.descriptors[1].serialized_descriptors[0],
      absl::CancelledError("test cancellation"), [&](absl::Status s) {
        cancel_result = s;
        cancel_done.Notify();
      });
  cancel_done.WaitForNotification();
  EXPECT_OK(cancel_result);

  ASSERT_EQ(buffers.size(), 2);
  EXPECT_THAT(buffers[0]->ToLiteral().Await().status(),
              StatusIs(absl::StatusCode::kCancelled));
  EXPECT_THAT(buffers[1]->ToLiteral().Await().status(),
              StatusIs(absl::StatusCode::kCancelled));
}

TEST(StreamExecutorGpuClientTest, CrossHostTransferPoisonedDescriptor) {
  GpuClientOptions options = GetTestGpuClientOptions(2);
  ASSERT_OK_AND_ASSIGN(auto client, GetStreamExecutorGpuClient(options));
  ASSERT_GE(client->devices().size(), 2);

  auto* const src_device = client->addressable_devices()[0];

  Literal input_literal =
      LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  ASSERT_OK_AND_ASSIGN(auto* src_memory_space,
                       src_device->default_memory_space());
  ASSERT_OK_AND_ASSIGN(auto src_buffer, client->BufferFromHostLiteral(
                                            input_literal, src_memory_space));

  auto [promise, future] = MakePromise<std::string>();

  absl::Notification send_done;
  absl::Status send_status;
  bool send_dispatched = true;
  src_buffer->CopyToRemoteDevice(future, [&](absl::Status s, bool dispatched) {
    send_status = s;
    send_dispatched = dispatched;
    send_done.Notify();
  });

  promise.Set(absl::InternalError("poisoned descriptor"));

  send_done.WaitForNotification();

  EXPECT_THAT(send_status,
              StatusIs(absl::StatusCode::kInternal, "poisoned descriptor"));
  EXPECT_FALSE(send_dispatched);
}

TEST(StreamExecutorGpuClientTest, CrossHostTransferSourceBufferError) {
  GpuClientOptions options = GetTestGpuClientOptions(2);
  ASSERT_OK_AND_ASSIGN(auto client, GetStreamExecutorGpuClient(options));
  ASSERT_GE(client->devices().size(), 2);

  auto* const src_device = client->addressable_devices()[0];
  auto* const dst_device = client->addressable_devices()[1];

  Shape shape = ShapeUtil::MakeShape(F32, {4});
  absl::Notification done;
  PjRtCrossHostRecvState recv_state;
  ASSERT_OK_AND_ASSIGN(
      auto buffers,
      client->MakeCrossHostReceiveBuffers(
          {shape}, dst_device,
          [&](absl::StatusOr<PjRtCrossHostRecvState> descriptors) {
            CHECK_OK(descriptors.status());
            recv_state = *std::move(descriptors);
            done.Notify();
          }));
  done.WaitForNotification();

  ASSERT_EQ(recv_state.descriptors.size(), 1);
  ASSERT_NE(recv_state.cancel_notifier, nullptr);

  ASSERT_OK_AND_ASSIGN(auto* src_memory_space,
                       src_device->default_memory_space());
  ASSERT_OK_AND_ASSIGN(
      auto src_buffer,
      client->CreateErrorBuffer(absl::InternalError("source buffer error"),
                                shape, src_memory_space));

  absl::Notification send_done;
  absl::Status send_status;
  bool send_dispatched = true;
  src_buffer->CopyToRemoteDevice(
      recv_state.descriptors[0].serialized_descriptors[0],
      [&](absl::Status s, bool dispatched) {
        send_status = s;
        send_dispatched = dispatched;
        send_done.Notify();
      });
  send_done.WaitForNotification();

  EXPECT_THAT(send_status,
              StatusIs(absl::StatusCode::kInternal, "source buffer error"));
  EXPECT_FALSE(send_dispatched);

  absl::Notification cancel_done;
  absl::Status cancel_result;
  recv_state.cancel_notifier(
      recv_state.descriptors[0].serialized_descriptors[0], send_status,
      [&](absl::Status s) {
        cancel_result = s;
        cancel_done.Notify();
      });
  cancel_done.WaitForNotification();
  EXPECT_OK(cancel_result);

  EXPECT_THAT(buffers[0]->ToLiteral().Await().status(),
              StatusIs(absl::StatusCode::kInternal, "source buffer error"));
}

TEST(StreamExecutorGpuClientTest, CrossHostTransferSendRecvZeroSizedBuffer) {
  GpuClientOptions options = GetTestGpuClientOptions(2);
  ASSERT_OK_AND_ASSIGN(auto client, GetStreamExecutorGpuClient(options));
  ASSERT_GE(client->devices().size(), 2);

  auto* const src_device = client->addressable_devices()[0];
  auto* const dst_device = client->addressable_devices()[1];

  ASSERT_OK_AND_ASSIGN(auto input_literal,
                       Literal::Make(ShapeUtil::MakeShape(F32, {0})));
  ASSERT_OK_AND_ASSIGN(auto* const src_memory_space,
                       src_device->default_memory_space());
  ASSERT_OK_AND_ASSIGN(auto src_buffer, client->BufferFromHostLiteral(
                                            input_literal, src_memory_space));

  auto [desc_promise, desc_future] = MakePromise<std::string>();

  ASSERT_OK_AND_ASSIGN(
      auto buffers,
      client->MakeCrossHostReceiveBuffers(
          {input_literal.shape()}, dst_device,
          [&](absl::StatusOr<PjRtCrossHostRecvState> descriptors) {
            CHECK_OK(descriptors);
            CHECK_EQ(descriptors->descriptors.size(), 1);

            const auto& desc = descriptors->descriptors[0];
            CHECK_EQ(desc.serialized_descriptors.size(), 1);
            desc_promise.Set(desc.serialized_descriptors[0]);
          }));
  ASSERT_EQ(buffers.size(), 1);

  src_buffer->CopyToRemoteDevice(desc_future,
                                 [&](absl::Status s, bool dispatched) {
                                   ASSERT_TRUE(dispatched);
                                   ASSERT_OK(s);
                                 });

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> recv_literal,
                       buffers[0]->ToLiteral().Await());
  EXPECT_TRUE(LiteralTestUtil::Equal(input_literal, *recv_literal));
}

}  // namespace
}  // namespace xla
