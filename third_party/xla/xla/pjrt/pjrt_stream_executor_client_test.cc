/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/pjrt/pjrt_stream_executor_client.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal.h"
#include "xla/literal_comparison.h"
#include "xla/literal_util.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

absl::StatusOr<std::unique_ptr<PjRtStreamExecutorClient>> GetClient() {
  LocalClient* local_client = xla::ClientLibrary::LocalClientOrDie();
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform("Host"));
  TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                      platform->ExecutorForDevice(0));
  auto device_state = std::make_unique<LocalDeviceState>(
      executor, local_client, LocalDeviceState::kSynchronous,
      /*max_inflight_computations=*/32,
      /*allow_event_reuse=*/false, /*use_callback_stream=*/false);
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  devices.emplace_back(std::make_unique<PjRtStreamExecutorDevice>(
      0, std::move(device_state), "cpu"));
  std::vector<std::unique_ptr<PjRtMemorySpace>> memory_spaces;
  memory_spaces.emplace_back(std::make_unique<PjRtStreamExecutorMemorySpace>(
      0, devices.back().get(), "cpu", 0));
  devices.back()->AttachMemorySpace(memory_spaces.back().get(),
                                    /*is_default=*/true);
  return std::make_unique<PjRtStreamExecutorClient>(
      "cpu", local_client, std::move(devices),
      /*process_index=*/0, std::move(memory_spaces), /*allocator=*/nullptr,
      /*host_memory_allocator=*/nullptr,
      /*should_stage_host_to_device_transfers=*/false,
      /*gpu_run_options=*/nullptr);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> ToyExecutable(
    PjRtStreamExecutorClient& client, Shape shape,
    absl::AnyInvocable<void(XlaBuilder&)> set_up_aliases) {
  CompileOptions compile_options;
  XlaBuilder builder("Add");
  auto a = Parameter(&builder, 0, shape, "a");
  auto b = Parameter(&builder, 1, shape, "b");
  auto c = Add(a, b);
  auto d = Add(c, c);
  Tuple(&builder, {c, d});
  set_up_aliases(builder);
  TF_ASSIGN_OR_RETURN(auto computation,
                      builder.Build(/*remove_dynamic_dimensions=*/true));
  TF_ASSIGN_OR_RETURN(auto executable,
                      client.Compile(computation, compile_options));
  return executable;
}

absl::Status ExecuteWithSameInputBuffer(
    absl::AnyInvocable<void(XlaBuilder&)> set_up_aliases) {
  auto shape = xla::ShapeUtil::MakeScalarShape(xla::F32);
  TF_ASSIGN_OR_RETURN(auto client, GetClient());
  TF_RET_CHECK(!client->addressable_devices().empty());
  auto* device0 = client->addressable_devices().front();
  TF_ASSIGN_OR_RETURN(auto buffer,
                      client->CreateUninitializedBuffer(
                          shape, *device0->default_memory_space()));
  TF_ASSIGN_OR_RETURN(auto executable,
                      ToyExecutable(*client, shape, std::move(set_up_aliases)));
  return executable->Execute({{buffer.get(), buffer.get()}}, /*options=*/{})
      .status();
}

TEST(PjRtStreamExecutorClientTest, DonateSameBufferTwice) {
  // f(a, a)
  auto status = ExecuteWithSameInputBuffer([](XlaBuilder& builder) {});
  ASSERT_TRUE(status.ok());

  // f(donate(a), a)
  status = ExecuteWithSameInputBuffer(
      [](XlaBuilder& builder) { builder.SetUpAlias({0}, 0, {}); });
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(), ::testing::HasSubstr("f(donate(a), a)"));

  // f(a, donate(a))
  status = ExecuteWithSameInputBuffer(
      [](XlaBuilder& builder) { builder.SetUpAlias({0}, 1, {}); });
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(), ::testing::HasSubstr("f(a, donate(a))"));

  // f(donate(a), donate(a))
  status = ExecuteWithSameInputBuffer([](XlaBuilder& builder) {
    builder.SetUpAlias({0}, 0, {});
    builder.SetUpAlias({1}, 1, {});
  });
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("f(donate(a), donate(a))"));
}

TEST(PjRtStreamExecutorClientTest, DonateWithControlDependency) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetClient());
  auto literal = LiteralUtil::CreateR2({{1, 2, 3}, {4, 5, 6}});
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->BufferFromHostLiteral(literal, client->memory_spaces()[0]));

  PjRtFuture<>::Promise promise = PjRtFuture<>::CreatePromise();
  PjRtFuture<> future(promise);
  auto blocked_buffer =
      std::move(*(buffer->DonateWithControlDependency(future)));
  EXPECT_TRUE(buffer->IsDeleted());

  buffer.reset();
  absl::Mutex mu;
  auto result_literal = std::make_shared<Literal>(
      ShapeUtil::DeviceShapeToHostShape(blocked_buffer->on_device_shape()));
  bool got_literal = false;
  blocked_buffer->ToLiteral(result_literal.get()).OnReady([&](absl::Status s) {
    absl::MutexLock l(&mu);
    TF_ASSERT_OK(s);
    got_literal = true;
  });
  blocked_buffer.reset();

  EXPECT_FALSE(got_literal);

  promise.Set();
  EXPECT_TRUE(future.IsReady());

  {
    absl::MutexLock l(&mu);
    mu.Await(absl::Condition(&got_literal));
  }

  TF_ASSERT_OK(literal_comparison::Equal(literal, *result_literal));
}

}  // namespace
}  // namespace xla
