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

#include "xla/pjrt/se/pjrt_stream_executor_client.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/future.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal.h"
#include "xla/literal_comparison.h"
#include "xla/literal_util.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/common_pjrt_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/se/local_device_state.h"
#include "xla/service/computation_placer.h"
#include "xla/service/platform_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/path.h"

namespace xla {
namespace {

using ::testing::HasSubstr;

absl::StatusOr<std::unique_ptr<PjRtStreamExecutorClient>> GetClient() {
  LocalClient* local_client = xla::ClientLibrary::LocalClientOrDie();
  ASSIGN_OR_RETURN(se::Platform * platform, PlatformUtil::GetPlatform("Host"));
  ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                   platform->ExecutorForDevice(0));
  auto device_state = std::make_unique<LocalDeviceState>(
      executor, local_client, LocalDeviceState::kSynchronous,
      /*max_inflight_computations=*/32,
      /*allow_event_reuse=*/false, /*use_callback_stream=*/false);
  int local_device_id = device_state->local_device_id().value();
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  devices.emplace_back(std::make_unique<PjRtStreamExecutorDevice>(
      0, std::move(device_state), local_device_id, /*process_index=*/0,
      /*process_index_in_partition=*/0, /*partition_index=*/0, "cpu"));
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

// Variant of GetClient() that creates `num_devices` Host-platform devices, so
// multi-device code paths in CommonPjRtLoadedExecutable::Execute can be
// exercised without accelerator hardware.
absl::StatusOr<std::unique_ptr<PjRtStreamExecutorClient>> GetClientWithDevices(
    int num_devices) {
  LocalClient* local_client = xla::ClientLibrary::LocalClientOrDie();
  ASSIGN_OR_RETURN(se::Platform * platform, PlatformUtil::GetPlatform("Host"));
  if (platform->VisibleDeviceCount() < num_devices) {
    return absl::UnavailableError(
        absl::StrFormat("Host platform has %d devices, need %d",
                        platform->VisibleDeviceCount(), num_devices));
  }
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  devices.reserve(num_devices);
  std::vector<std::unique_ptr<PjRtMemorySpace>> memory_spaces;
  memory_spaces.reserve(num_devices);
  for (int i = 0; i < num_devices; ++i) {
    ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                     platform->ExecutorForDevice(i));
    auto device_state = std::make_unique<LocalDeviceState>(
        executor, local_client, LocalDeviceState::kSynchronous,
        /*max_inflight_computations=*/32,
        /*allow_event_reuse=*/false, /*use_callback_stream=*/false);
    int local_device_id = device_state->local_device_id().value();
    devices.emplace_back(std::make_unique<PjRtStreamExecutorDevice>(
        i, std::move(device_state), local_device_id, /*process_index=*/0,
        /*process_index_in_partition=*/0, /*partition_index=*/0, "cpu"));
    memory_spaces.emplace_back(std::make_unique<PjRtStreamExecutorMemorySpace>(
        i, devices.back().get(), "cpu", 0));
    devices.back()->AttachMemorySpace(memory_spaces.back().get(),
                                      /*is_default=*/true);
  }
  return std::make_unique<PjRtStreamExecutorClient>(
      "cpu", local_client, std::move(devices), /*process_index=*/0,
      std::move(memory_spaces), /*allocator=*/nullptr,
      /*host_memory_allocator=*/nullptr,
      /*should_stage_host_to_device_transfers=*/false,
      /*gpu_run_options=*/nullptr);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> ToyExecutable(
    PjRtStreamExecutorClient& client, Shape shape,
    absl::AnyInvocable<void(XlaBuilder&)> set_up_aliases,
    CompileOptions compile_options = {}) {
  XlaBuilder builder("Add");
  auto a = Parameter(&builder, 0, shape, "a");
  auto b = Parameter(&builder, 1, shape, "b");
  auto c = Add(a, b);
  auto d = Add(c, c);
  Tuple(&builder, {c, d});
  set_up_aliases(builder);
  ASSIGN_OR_RETURN(auto computation,
                   builder.Build(/*remove_dynamic_dimensions=*/true));
  ASSIGN_OR_RETURN(auto executable,
                   client.CompileAndLoad(computation, compile_options));
  return executable;
}

absl::Status ExecuteWithSameInputBuffer(
    absl::AnyInvocable<void(XlaBuilder&)> set_up_aliases) {
  auto shape = xla::ShapeUtil::MakeScalarShape(xla::F32);
  ASSIGN_OR_RETURN(auto client, GetClient());
  TF_RET_CHECK(!client->addressable_devices().empty());
  auto* device0 = client->addressable_devices().front();
  ASSIGN_OR_RETURN(auto buffer, client->CreateUninitializedBuffer(
                                    shape, *device0->default_memory_space()));
  ASSIGN_OR_RETURN(auto executable,
                   ToyExecutable(*client, shape, std::move(set_up_aliases)));
  xla::ExecuteOptions options;
  return executable->Execute({{buffer.get(), buffer.get()}}, options).status();
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
      client->BufferFromHostLiteral(literal, client->memory_spaces()[0],
                                    /*device_layout=*/nullptr));

  auto [promise, future] = MakePromise<>();
  auto blocked_buffer =
      std::move(*(buffer->DonateWithControlDependency(future)));
  EXPECT_TRUE(buffer->IsDeleted());

  buffer.reset();
  absl::Mutex mu;
  auto result_literal = std::make_shared<Literal>(
      ShapeUtil::DeviceShapeToHostShape(blocked_buffer->on_device_shape()));
  bool got_literal = false;
  blocked_buffer->ToLiteral(result_literal.get()).OnReady([&](absl::Status s) {
    absl::MutexLock l(mu);
    TF_ASSERT_OK(s);
    got_literal = true;
  });
  blocked_buffer.reset();

  EXPECT_FALSE(got_literal);

  promise.Set();
  EXPECT_TRUE(future.IsReady());

  {
    absl::MutexLock l(mu);
    mu.Await(absl::Condition(&got_literal));
  }

  TF_ASSERT_OK(literal_comparison::Equal(literal, *result_literal));
}

TEST(PjRtStreamExecutorClientTest, ExecuteWithInputError) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtStreamExecutorClient> client,
                          GetClient());
  Shape shape = xla::ShapeUtil::MakeScalarShape(F32);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> in_buffer,
      client->CreateErrorBuffer(
          absl::InternalError("test error"), shape,
          *client->addressable_devices()[0]->default_memory_space()));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtLoadedExecutable> executable,
      ToyExecutable(*client, shape, [](XlaBuilder& builder) {}));

  // Call Execute with the error buffer.
  ExecuteOptions options;
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> result,
      executable->Execute({{in_buffer.get(), in_buffer.get()}}, options));
  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0].size(), 2);

  for (const auto& buf : result[0]) {
    EXPECT_EQ(buf->on_device_shape(), shape);
    EXPECT_THAT(buf->GetReadyFuture().Await(),
                absl_testing::StatusIs(absl::StatusCode::kInternal,
                                       HasSubstr("test error")));
  }
}

TEST(PjRtStreamExecutorClientTest, DeserializeAndDump) {
  tsl::Env* env = tsl::Env::Default();
  EXPECT_TRUE(env);
  Shape shape = xla::ShapeUtil::MakeScalarShape(F32);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtStreamExecutorClient> client,
                          GetClient());
  std::string compile_dump_dir;
  EXPECT_TRUE(env->LocalTempFilename(&compile_dump_dir));
  CompileOptions compile_options;
  compile_options.executable_build_options.mutable_debug_options()
      ->set_xla_dump_to(compile_dump_dir);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtLoadedExecutable> executable,
      ToyExecutable(
          *client, shape, [](XlaBuilder& builder) {}, compile_options));
  std::string compile_dump_name, compile_dump_contents;
  {
    std::vector<std::string> matches;
    TF_ASSERT_OK(env->GetMatchingPaths(
        tsl::io::JoinPath(compile_dump_dir, "*after_optimizations.txt"),
        &matches));
    EXPECT_THAT(matches, testing::SizeIs(1));
    compile_dump_name = std::move(matches.front());
    TF_ASSERT_OK(
        tsl::ReadFileToString(env, compile_dump_name, &compile_dump_contents));
  }
  TF_ASSERT_OK_AND_ASSIGN(std::string serialized,
                          client->SerializeExecutable(*executable));
  std::string deserialize_dump_dir;
  EXPECT_TRUE(env->LocalTempFilename(&deserialize_dump_dir));
  EXPECT_NE(compile_dump_dir, deserialize_dump_dir);
  CompileOptions deserialize_options;
  deserialize_options.executable_build_options.mutable_debug_options()
      ->set_xla_dump_to(deserialize_dump_dir);
  LoadOptions load_options;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtLoadedExecutable> reloaded_executable,
      client->LoadSerializedExecutable(serialized, deserialize_options,
                                       load_options));
  std::string deserialize_dump_name, deserialize_dump_contents;
  {
    std::vector<std::string> matches;
    TF_ASSERT_OK(env->GetMatchingPaths(
        tsl::io::JoinPath(deserialize_dump_dir, "*after_optimizations.txt"),
        &matches));
    EXPECT_THAT(matches, testing::SizeIs(1));
    deserialize_dump_name = std::move(matches.front());
    TF_ASSERT_OK(tsl::ReadFileToString(env, deserialize_dump_name,
                                       &deserialize_dump_contents));
  }
  EXPECT_EQ(compile_dump_contents, deserialize_dump_contents);
}

TEST(PjRtStreamExecutorClientTest, ExecutePortableRemoteDevice) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtStreamExecutorClient> client,
                          GetClient());
  Shape shape = xla::ShapeUtil::MakeScalarShape(F32);
  ASSERT_FALSE(client->addressable_devices().empty());
  auto* device0 = client->addressable_devices().front();
  TF_ASSERT_OK_AND_ASSIGN(PjRtMemorySpace * memory_space,
                          device0->default_memory_space());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtBuffer> buffer,
      client->CreateUninitializedBuffer(shape, memory_space));

  CompileOptions compile_options;
  compile_options.compile_portable_executable = true;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<PjRtLoadedExecutable> executable,
      ToyExecutable(
          *client, shape, [](XlaBuilder& builder) {}, compile_options));

  auto remote_device = std::make_unique<PjRtStreamExecutorDevice>(
      1, /*local_device_state=*/nullptr, /*local_device_id=*/-1,
      /*process_index=*/1, /*process_index_in_partition=*/1,
      /*partition_index=*/0, "cpu");
  remote_device->SetClient(client.get());

  ExecuteOptions options;
  auto result_or = executable->ExecutePortable({buffer.get(), buffer.get()},
                                               remote_device.get(), options);
  EXPECT_THAT(result_or.status(),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     HasSubstr("not addressable")));
}

TEST(PjRtStreamExecutorClientTest, MakeAllocationReadyEventAsync) {
  TF_ASSERT_OK_AND_ASSIGN(auto client, GetClient());
  ASSERT_FALSE(client->addressable_devices().empty());
  auto* device0 = client->addressable_devices().front();
  TF_ASSERT_OK_AND_ASSIGN(PjRtMemorySpace * memory_space,
                          device0->default_memory_space());

  std::vector<int32_t> data(1024);
  std::iota(data.begin(), data.end(), 100);
  TF_ASSERT_OK_AND_ASSIGN(
      auto buffer, client->BufferFromHostBuffer(
                       data.data(), S32, {1024}, /*byte_strides=*/std::nullopt,
                       PjRtClient::HostBufferSemantics::kImmutableZeroCopy,
                       nullptr, memory_space, /*device_layout=*/nullptr));

  Shape shape = buffer->on_device_shape();
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          client->CreateAliasBuffer(shape, memory_space));
  std::unique_ptr<PjRtBuffer> unfulfilled_buffer = std::move(result.first);
  auto fulfill_callback = std::move(result.second);

  auto* common_buffer =
      static_cast<xla::CommonPjRtBuffer*>(unfulfilled_buffer.get());
  auto hold = common_buffer->GetBufferWithHold(
      xla::CommonPjRtBuffer::ScopedHold::kUsage);
  ASSERT_TRUE(hold.ok());
  auto unfulfilled_raw_buffer = hold.buffer()->raw_buffer();

  TF_ASSERT_OK_AND_ASSIGN(
      auto slice, unfulfilled_raw_buffer->Slice(0, 128 * sizeof(int32_t)));
  TF_ASSERT_OK_AND_ASSIGN(auto slice_ready_event,
                          slice->MakeAllocationReadyEvent());

  // Since the parent buffer is not yet fulfilled, the event must NOT be ready
  // yet.
  EXPECT_FALSE(slice_ready_event.async_value()->IsAvailable());

  // Fulfill the buffer.
  TF_ASSERT_OK(std::move(fulfill_callback)(buffer.get()));

  // Await the slice event.
  tsl::BlockUntilReady(slice_ready_event.async_value());
  EXPECT_TRUE(slice_ready_event.async_value()->IsConcrete());
  if (auto* error = slice_ready_event.async_value()->GetErrorIfPresent()) {
    ASSERT_OK(*error);
  }
}

// Regression test for the two-phase launch barrier: when any device's Prepare
// fails, no device may proceed to ExecuteLaunch. Before the fix, a device
// whose Prepare succeeded could observe `failed == 0` (because a failing peer
// had passed the preparing==0 barrier but not yet written `failed`) and enter
// ExecuteLaunch; with a real cross-device collective that device then hangs
// at the rendezvous waiting for a peer that never arrives. The test asserts
// the invariant directly (zero ExecuteLaunch calls) rather than via a hang,
// so it needs no collective. PjRtStreamExecutorClient is the production
// client with supports_two_phase_launch()==true, so Host-backed SE devices
// exercise the real barrier path without accelerator hardware.
TEST(PjRtStreamExecutorClientTest, TwoPhaseExecutePrepareFailureSkipsLaunch) {
  constexpr int kNumDevices = 2;
  // The pre-fix race is schedule-dependent (it fires only when a succeeding
  // device is last to the barrier), so repeat with the failing device
  // alternating. On Host-SE each iteration is microseconds.
  constexpr int kIterations = 50;

  auto client_or = GetClientWithDevices(kNumDevices);
  if (absl::IsUnavailable(client_or.status())) {
    GTEST_SKIP() << client_or.status();
  }
  ASSERT_OK_AND_ASSIGN(auto client, std::move(client_or));
  ASSERT_TRUE(client->supports_two_phase_launch());

  Shape shape = ShapeUtil::MakeScalarShape(F32);
  // CheckBufferCompatibilities rejects this on the failing device — different
  // on-device size from the compiled scalar parameter.
  Shape wrong_shape = ShapeUtil::MakeShape(F32, {2});

  CompileOptions compile_options;
  compile_options.executable_build_options.set_num_replicas(kNumDevices);
  DeviceAssignment assignment(kNumDevices, /*computation_count=*/1);
  for (int i = 0; i < kNumDevices; ++i) {
    assignment(i, 0) = i;
  }
  compile_options.executable_build_options.set_device_assignment(assignment);
  ASSERT_OK_AND_ASSIGN(
      auto executable,
      ToyExecutable(*client, shape, [](XlaBuilder&) {}, compile_options));
  ASSERT_EQ(executable->addressable_devices().size(), kNumDevices);

  std::atomic<int> launch_calls{0};
  tensorflow::down_cast<CommonPjRtLoadedExecutable*>(executable.get())
      ->SetExecuteLaunchHookForTesting([&](PjRtDevice*) {
        launch_calls.fetch_add(1, std::memory_order_relaxed);
      });

  std::vector<std::unique_ptr<PjRtBuffer>> ok_bufs(kNumDevices);
  std::vector<std::unique_ptr<PjRtBuffer>> wrong_bufs(kNumDevices);
  for (int d = 0; d < kNumDevices; ++d) {
    auto* mem = *client->addressable_devices()[d]->default_memory_space();
    ASSERT_OK_AND_ASSIGN(ok_bufs[d],
                         client->CreateUninitializedBuffer(shape, mem));
    ASSERT_OK_AND_ASSIGN(wrong_bufs[d],
                         client->CreateUninitializedBuffer(wrong_shape, mem));
  }

  absl::Notification done;
  absl::Status last_result;
  std::unique_ptr<tsl::Thread> t(tsl::Env::Default()->StartThread(
      tsl::ThreadOptions(), "TwoPhaseExecutePrepareFailureSkipsLaunch", [&] {
        for (int i = 0; i < kIterations; ++i) {
          int failing = i % kNumDevices;
          std::vector<std::vector<PjRtBuffer*>> args(kNumDevices);
          for (int d = 0; d < kNumDevices; ++d) {
            PjRtBuffer* b =
                (d == failing) ? wrong_bufs[d].get() : ok_bufs[d].get();
            args[d] = {b, b};
          }
          ExecuteOptions options;
          last_result = executable->Execute(args, options).status();
          if (last_result.ok()) {
            break;
          }
        }
        done.Notify();
      }));
  if (!done.WaitForNotificationWithTimeout(absl::Seconds(60))) {
    // Release rather than join. FAIL() records the diagnostic before return;
    // teardown then blocks (the client's per-device worker threads are parked
    // in mu.Await() inside the detached Execute call) and the test framework's
    // timeout reaps the process.
    t.release();
    FAIL() << "Execute() did not return within 60s with one device's Prepare "
              "failing; two-phase barrier exit path is wedged.";
  }
  t.reset();
  EXPECT_FALSE(last_result.ok()) << last_result;
  // The load-bearing assertion: with any Prepare failure, no device reaches
  // phase 2. On a regressed barrier this count is nonzero for some schedule.
  EXPECT_EQ(launch_calls.load(), 0)
      << "ExecuteLaunch was reached on " << launch_calls.load()
      << " device(s) across " << kIterations
      << " iterations despite a peer Prepare failure; the two-phase barrier "
         "let a succeeding device past before the failure was recorded.";
}

}  // namespace
}  // namespace xla
