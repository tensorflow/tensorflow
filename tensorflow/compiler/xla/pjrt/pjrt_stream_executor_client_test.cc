/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include "absl/functional/any_invocable.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

xla::StatusOr<std::unique_ptr<PjRtStreamExecutorClient>> GetClient() {
  LocalClient* local_client = xla::ClientLibrary::LocalClientOrDie();
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform("Host"));
  se::StreamExecutorConfig config;
  config.ordinal = 0;
  TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                      platform->GetExecutor(config));
  auto device_state = std::make_unique<LocalDeviceState>(
      executor, local_client, LocalDeviceState::kSynchronous,
      /*max_inflight_computations=*/32,
      /*allow_event_reuse=*/false, /*use_callback_stream=*/false);
  auto device = std::make_unique<PjRtStreamExecutorDevice>(
      0, std::move(device_state), "cpu");
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  devices.emplace_back(std::move(device));
  return std::make_unique<PjRtStreamExecutorClient>(
      "cpu", local_client, std::move(devices), /*process_index=*/0,
      /*allocator=*/nullptr, /*host_memory_allocator=*/nullptr,
      /*should_stage_host_to_device_transfers=*/false,
      /*gpu_run_options=*/nullptr);
}

StatusOr<std::unique_ptr<PjRtLoadedExecutable>> ToyExecutable(
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

Status ExecuteWithSameInputBuffer(
    absl::AnyInvocable<void(XlaBuilder&)> set_up_aliases) {
  auto shape = xla::ShapeUtil::MakeScalarShape(xla::F32);
  TF_ASSIGN_OR_RETURN(auto client, GetClient());
  TF_ASSIGN_OR_RETURN(auto* device0, client->LookupDevice(0));
  TF_ASSIGN_OR_RETURN(auto buffer,
                      client->CreateUninitializedBuffer(shape, device0));
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
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("f(donate(a), a)"));

  // f(a, donate(a))
  status = ExecuteWithSameInputBuffer(
      [](XlaBuilder& builder) { builder.SetUpAlias({0}, 1, {}); });
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(), ::testing::HasSubstr("f(a, donate(a))"));

  // f(donate(a), donate(a))
  status = ExecuteWithSameInputBuffer([](XlaBuilder& builder) {
    builder.SetUpAlias({0}, 0, {});
    builder.SetUpAlias({1}, 1, {});
  });
  ASSERT_FALSE(status.ok());
  EXPECT_THAT(status.error_message(),
              ::testing::HasSubstr("f(donate(a), donate(a))"));
}

}  // namespace
}  // namespace xla
