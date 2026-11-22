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

// Separate from cpu_client_test.cc: host_callback.h includes the external FFI
// API, which JAX's CPU callbacks use and which cannot be mixed with
// xla/ffi/ffi.h.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/ffi/api/ffi.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

// Executed by the host callback below on the callback's thread, as JAX's
// Python CPU callbacks do. Set by the test.
PjRtLoadedExecutable* nested_executable = nullptr;

absl::StatusOr<float> RunNestedExecution() {
  HostCallbackScope host_callback_scope;
  // JAX gives each dispatching thread its own execution stream. On the
  // caller's stream the nested execution would also wait for the computation
  // queued behind the callback. Only asynchronous executions are paced.
  ExecuteOptions options;
  options.execution_stream_id = 1;
  options.execution_mode = ExecuteOptions::ExecutionMode::kAsynchronous;
  ABSL_ASSIGN_OR_RETURN(auto outputs, nested_executable->Execute({{}}, options));
  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<Literal> literal,
                   outputs[0][0]->ToLiteral().Await());
  return literal->Get<float>({});
}

ffi::Error ExecuteNested(ffi::AnyBuffer,
                         ffi::Result<ffi::BufferR0<ffi::F32>> result) {
  absl::StatusOr<float> value = RunNestedExecution();
  if (!value.ok()) {
    return ffi::Error::Internal(std::string(value.status().message()));
  }
  *result->typed_data() = *value;
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    kExecuteNested, ExecuteNested,
    ffi::Ffi::Bind().Arg<ffi::AnyBuffer>().Ret<ffi::BufferR0<ffi::F32>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$ExecuteNested",
                         "Host", kExecuteNested);

// The id of the thread that last ran the host callback below.
int64_t callback_thread_id = 0;

ffi::Error RecordThread(ffi::AnyBuffer,
                        ffi::Result<ffi::BufferR0<ffi::F32>> result) {
  callback_thread_id = tsl::Env::Default()->GetCurrentThreadId();
  *result->typed_data() = 0.0f;
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    kRecordThread, RecordThread,
    ffi::Ffi::Bind().Arg<ffi::AnyBuffer>().Ret<ffi::BufferR0<ffi::F32>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$RecordThread",
                         "Host", kRecordThread);

// Run by the host callback below inside a host callback scope, as JAX's CPU
// callbacks are. Set by the test; called concurrently by multi device programs.
std::function<absl::Status()>* callback_body = nullptr;

ffi::Error RunCallbackBody(ffi::AnyBuffer,
                           ffi::Result<ffi::BufferR0<ffi::F32>> result) {
  HostCallbackScope host_callback_scope;
  absl::Status status = (*callback_body)();
  if (!status.ok()) {
    return ffi::Error::Internal(std::string(status.message()));
  }
  *result->typed_data() = 0.0f;
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER(
    kRunCallbackBody, RunCallbackBody,
    ffi::Ffi::Bind().Arg<ffi::AnyBuffer>().Ret<ffi::BufferR0<ffi::F32>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$RunCallbackBody",
                         "Host", kRunCallbackBody);

constexpr absl::string_view kCallbackBodyProgram = R"(
  HloModule callback_body
  ENTRY main {
    p0 = f32[] parameter(0)
    ROOT result = f32[] custom-call(p0),
        custom_call_target="__xla_test$$RunCallbackBody",
        api_version=API_VERSION_TYPED_FFI
  })";

constexpr absl::string_view kConstantProgram = R"(
  HloModule constant
  ENTRY main {
    ROOT result = f32[] constant(42)
  })";

constexpr absl::string_view kNegateProgram = R"(
  HloModule negate
  ENTRY main {
    p0 = f32[] parameter(0)
    ROOT result = f32[] negate(p0)
  })";

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
    PjRtClient& client, absl::string_view program, int num_replicas = 1) {
  ABSL_ASSIGN_OR_RETURN(auto module, ParseAndReturnUnverifiedModule(program));
  CompileOptions options;
  options.executable_build_options.set_num_replicas(num_replicas);
  return client.CompileAndLoad(XlaComputation(module->ToProto()),
                               std::move(options));
}

// Crashes the test unless it finishes within 60 s, since a deadlocked client
// cannot be destroyed. Declare it first, so that it outlives the client.
class Watchdog {
 public:
  Watchdog()
      : thread_(tsl::Env::Default()->StartThread({}, "watchdog", [this] {
          CHECK(done_.WaitForNotificationWithTimeout(absl::Seconds(60)))
              << "The test deadlocked.";
        })) {}
  ~Watchdog() { done_.Notify(); }

 private:
  absl::Notification done_;
  std::unique_ptr<tsl::Thread> thread_;
};

// One f32[] input per device for a multi device program, provided later
// through the transfer managers.
struct PendingInputs {
  std::vector<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
      transfer_managers;
  std::vector<std::unique_ptr<PjRtBuffer>> buffers;
  std::vector<std::vector<PjRtBuffer*>> argument_handles;
};

absl::StatusOr<PendingInputs> CreatePendingInputs(PjRtClient& client,
                                                  int num_devices) {
  PendingInputs inputs;
  for (int i = 0; i < num_devices; ++i) {
    ABSL_ASSIGN_OR_RETURN(PjRtMemorySpace * memory_space,
                     client.addressable_devices()[i]->default_memory_space());
    ABSL_ASSIGN_OR_RETURN(auto transfer_manager,
                     client.CreateBuffersForAsyncHostToDevice(
                         {ShapeUtil::MakeShape(F32, {})}, memory_space));
    inputs.buffers.push_back(transfer_manager->RetrieveBuffer(0));
    inputs.argument_handles.push_back({inputs.buffers.back().get()});
    inputs.transfer_managers.push_back(std::move(transfer_manager));
  }
  return inputs;
}

TEST(CpuClientHostCallbackTest,
     NestedExecutionDoesNotWaitForComputationsQueuedBehindTheCallback) {
  CpuClientOptions options;
  options.max_inflight_computations_per_device = 2;
  // Two devices give the client at least two execute threads: the callback
  // program holds one while the nested execution runs on another.
  options.cpu_device_count = 2;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetXlaPjrtCpuClient(std::move(options)));
  ASSERT_OK_AND_ASSIGN(auto callback_executable, Compile(*client, R"(
    HloModule callback
    ENTRY main {
      p0 = f32[] parameter(0)
      ROOT result = f32[] custom-call(p0),
          custom_call_target="__xla_test$$ExecuteNested",
          api_version=API_VERSION_TYPED_FFI
    })"));
  ASSERT_OK_AND_ASSIGN(auto constant_executable, Compile(*client, R"(
    HloModule constant
    ENTRY main {
      ROOT result = f32[] constant(42)
    })"));
  nested_executable = constant_executable.get();

  ASSERT_OK_AND_ASSIGN(
      PjRtMemorySpace * memory_space,
      client->addressable_devices()[0]->default_memory_space());
  ASSERT_OK_AND_ASSIGN(auto transfer_manager,
                       client->CreateBuffersForAsyncHostToDevice(
                           {ShapeUtil::MakeShape(F32, {})}, memory_space));
  std::unique_ptr<PjRtBuffer> pending = transfer_manager->RetrieveBuffer(0);

  // Each dispatch holds one of the two units until its computation finishes.
  // The callback program waits for its input, and the computation queued
  // behind it on the same execution stream takes the second unit.
  ASSERT_OK_AND_ASSIGN(auto callback_result,
                       callback_executable->Execute({{pending.get()}}, {}));
  ASSERT_OK_AND_ASSIGN(auto queued_result,
                       constant_executable->Execute({{}}, {}));

  // Once its input arrives, the callback program runs the nested execution,
  // which must not wait for the unit held by the queued computation.
  Literal input = LiteralUtil::CreateR0(0.0f);
  ASSERT_OK(transfer_manager->TransferLiteralToBuffer(0, input, [] {}));
  absl::Notification done;
  callback_result[0][0]->GetReadyFuture().OnReady(
      [&](absl::Status) { done.Notify(); });
  // A deadlocked client cannot be destroyed, so crash instead of hanging.
  CHECK(done.WaitForNotificationWithTimeout(absl::Seconds(60)))
      << "The host callback's nested execution waited for a unit held by a "
         "computation queued behind the callback.";
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<Literal> literal,
                       callback_result[0][0]->ToLiteral().Await());
  EXPECT_EQ(literal->Get<float>({}), 42.0f);
  nested_executable = nullptr;
}

constexpr absl::string_view kRecordThreadProgram = R"(
  HloModule record_thread
  ENTRY main {
    p0 = f32[] parameter(0)
    ROOT result = f32[] custom-call(p0),
        custom_call_target="__xla_test$$RecordThread",
        api_version=API_VERSION_TYPED_FFI
  })";

// JAX executes programs with host callbacks synchronously, so that no
// computation can be queued behind one while its callback runs.
TEST(CpuClientHostCallbackTest,
     SynchronousExecutionWaitsForItsInputOnTheCallingThread) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetXlaPjrtCpuClient(CpuClientOptions()));
  ASSERT_OK_AND_ASSIGN(auto executable, Compile(*client, kRecordThreadProgram));
  ASSERT_OK_AND_ASSIGN(
      PjRtMemorySpace * memory_space,
      client->addressable_devices()[0]->default_memory_space());
  ASSERT_OK_AND_ASSIGN(auto transfer_manager,
                       client->CreateBuffersForAsyncHostToDevice(
                           {ShapeUtil::MakeShape(F32, {})}, memory_space));
  std::unique_ptr<PjRtBuffer> pending = transfer_manager->RetrieveBuffer(0);

  Literal input = LiteralUtil::CreateR0(0.0f);
  std::unique_ptr<tsl::Thread> producer(
      tsl::Env::Default()->StartThread({}, "producer", [&] {
        // Gives the execution below time to start waiting for its input.
        absl::SleepFor(absl::Milliseconds(100));
        CHECK_OK(transfer_manager->TransferLiteralToBuffer(0, input, [] {}));
      }));
  ExecuteOptions options;
  options.execution_mode = ExecuteOptions::ExecutionMode::kSynchronous;
  ASSERT_OK_AND_ASSIGN(auto result,
                       executable->Execute({{pending.get()}}, options));
  ASSERT_OK(result[0][0]->GetReadyFuture().Await());
  EXPECT_EQ(callback_thread_id, tsl::Env::Default()->GetCurrentThreadId());
}

TEST(CpuClientHostCallbackTest, SynchronousExecutionReportsAnInputError) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetXlaPjrtCpuClient(CpuClientOptions()));
  ASSERT_OK_AND_ASSIGN(auto executable, Compile(*client, kRecordThreadProgram));
  ASSERT_OK_AND_ASSIGN(
      PjRtMemorySpace * memory_space,
      client->addressable_devices()[0]->default_memory_space());
  ASSERT_OK_AND_ASSIGN(auto transfer_manager,
                       client->CreateBuffersForAsyncHostToDevice(
                           {ShapeUtil::MakeShape(F32, {})}, memory_space));
  std::unique_ptr<PjRtBuffer> pending = transfer_manager->RetrieveBuffer(0);

  callback_thread_id = 0;
  std::unique_ptr<tsl::Thread> producer(
      tsl::Env::Default()->StartThread({}, "producer", [&] {
        absl::SleepFor(absl::Milliseconds(100));
        transfer_manager->SetBufferError(0, absl::InternalError("bad input"));
      }));
  ExecuteOptions options;
  options.execution_mode = ExecuteOptions::ExecutionMode::kSynchronous;
  ASSERT_OK_AND_ASSIGN(auto result,
                       executable->Execute({{pending.get()}}, options));
  EXPECT_THAT(result[0][0]->GetReadyFuture().Await(),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("bad input")));
  EXPECT_EQ(callback_thread_id, 0);
}

TEST(CpuClientHostCallbackTest, SynchronousExecutionTakesNoPacingUnit) {
  CpuClientOptions options;
  options.max_inflight_computations_per_device = 1;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetXlaPjrtCpuClient(std::move(options)));
  ASSERT_OK_AND_ASSIGN(auto negate_executable,
                       Compile(*client, kNegateProgram));
  ASSERT_OK_AND_ASSIGN(auto constant_executable,
                       Compile(*client, kConstantProgram));
  ASSERT_OK_AND_ASSIGN(
      PjRtMemorySpace * memory_space,
      client->addressable_devices()[0]->default_memory_space());
  ASSERT_OK_AND_ASSIGN(auto transfer_manager,
                       client->CreateBuffersForAsyncHostToDevice(
                           {ShapeUtil::MakeShape(F32, {})}, memory_space));
  std::unique_ptr<PjRtBuffer> pending = transfer_manager->RetrieveBuffer(0);

  // The asynchronous execution holds the only unit until its input arrives.
  ExecuteOptions async_options;
  async_options.execution_mode = ExecuteOptions::ExecutionMode::kAsynchronous;
  ASSERT_OK_AND_ASSIGN(
      auto async_result,
      negate_executable->Execute({{pending.get()}}, async_options));

  // Provides that input once the synchronous execution below returns, or
  // after a timeout if it waits for the unit.
  absl::Notification sync_done;
  bool sync_waited = false;
  Literal input = LiteralUtil::CreateR0(1.0f);
  std::unique_ptr<tsl::Thread> producer(
      tsl::Env::Default()->StartThread({}, "producer", [&] {
        sync_waited =
            !sync_done.WaitForNotificationWithTimeout(absl::Seconds(10));
        CHECK_OK(transfer_manager->TransferLiteralToBuffer(0, input, [] {}));
      }));
  // On another stream, so that it does not wait for the asynchronous one.
  ExecuteOptions sync_options;
  sync_options.execution_mode = ExecuteOptions::ExecutionMode::kSynchronous;
  sync_options.execution_stream_id = 1;
  ASSERT_OK_AND_ASSIGN(auto sync_result,
                       constant_executable->Execute({{}}, sync_options));
  sync_done.Notify();
  producer.reset();
  EXPECT_FALSE(sync_waited);
}

// A synchronous execution dispatched from a host callback does not wait for
// its inputs: they may come from computations queued behind the program that
// runs the callback.
TEST(CpuClientHostCallbackTest,
     SynchronousExecutionFromCallbackDoesNotWaitForItsInputs) {
  Watchdog watchdog;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetXlaPjrtCpuClient(CpuClientOptions()));
  ASSERT_OK_AND_ASSIGN(auto callback_executable,
                       Compile(*client, kCallbackBodyProgram));
  ASSERT_OK_AND_ASSIGN(auto negate_executable,
                       Compile(*client, kNegateProgram));
  ASSERT_OK_AND_ASSIGN(PendingInputs inputs, CreatePendingInputs(*client, 1));

  // The callback program waits for its input, so the negation of its output
  // is queued behind it before the callback runs.
  ExecuteOptions async_options;
  async_options.execution_mode = ExecuteOptions::ExecutionMode::kAsynchronous;
  ASSERT_OK_AND_ASSIGN(
      auto callback_results,
      callback_executable->Execute(inputs.argument_handles, async_options));
  ASSERT_OK_AND_ASSIGN(
      auto queued_results,
      negate_executable->Execute({{callback_results[0][0].get()}}, {}));

  // The callback negates the queued computation's output synchronously and
  // returns without reading the result.
  ExecuteOptions sync_options;
  sync_options.execution_mode = ExecuteOptions::ExecutionMode::kSynchronous;
  std::vector<std::unique_ptr<PjRtBuffer>> nested_results;
  std::function<absl::Status()> body = [&]() -> absl::Status {
    ABSL_ASSIGN_OR_RETURN(auto outputs,
                     negate_executable->Execute({{queued_results[0][0].get()}},
                                                sync_options));
    nested_results = std::move(outputs[0]);
    return absl::OkStatus();
  };
  callback_body = &body;

  Literal input = LiteralUtil::CreateR0(0.0f);
  ASSERT_OK(
      inputs.transfer_managers[0]->TransferLiteralToBuffer(0, input, [] {}));
  EXPECT_OK(callback_results[0][0]->GetReadyFuture().Await());
  EXPECT_OK(queued_results[0][0]->GetReadyFuture().Await());
  ASSERT_EQ(nested_results.size(), 1);
  EXPECT_OK(nested_results[0]->GetReadyFuture().Await());
  callback_body = nullptr;
}

// A multi device program with host callbacks also finishes before a
// synchronous Execute returns, and takes no pacing units. So computations
// dispatched after it cannot hold the units that work started from its
// callbacks needs.
TEST(CpuClientHostCallbackTest,
     MultiDeviceSynchronousExecutionFinishesBeforeReturning) {
  Watchdog watchdog;
  CpuClientOptions options;
  // The programs run on two of the four devices. Four devices give the client
  // at least four execute threads, so the callbacks' computations do not wait
  // for the two threads that the program holds.
  options.cpu_device_count = 4;
  options.max_inflight_computations_per_device = 1;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetXlaPjrtCpuClient(std::move(options)));
  ASSERT_OK_AND_ASSIGN(auto callback_executable,
                       Compile(*client, kCallbackBodyProgram, 2));
  ASSERT_OK_AND_ASSIGN(auto queued_executable,
                       Compile(*client, kConstantProgram, 2));
  ASSERT_OK_AND_ASSIGN(auto constant_executable,
                       Compile(*client, kConstantProgram));
  ASSERT_OK_AND_ASSIGN(PendingInputs inputs, CreatePendingInputs(*client, 2));

  // Each callback waits for an asynchronous computation on device 0 that a
  // thread of its own dispatches, as thread_map's threads do. That thread is
  // outside the callback's scope, so its computation takes a pacing unit.
  std::function<absl::Status()> body = [&]() -> absl::Status {
    absl::Status status;
    std::unique_ptr<tsl::Thread> thread(
        tsl::Env::Default()->StartThread({}, "thread", [&] {
          ExecuteOptions options;
          options.execution_stream_id =
              tsl::Env::Default()->GetCurrentThreadId();
          options.execution_mode = ExecuteOptions::ExecutionMode::kAsynchronous;
          auto outputs = constant_executable->Execute({{}}, options);
          status = outputs.ok() ? (*outputs)[0][0]->GetReadyFuture().Await()
                                : outputs.status();
        }));
    thread.reset();
    return status;
  };
  callback_body = &body;

  Literal input = LiteralUtil::CreateR0(0.0f);
  std::unique_ptr<tsl::Thread> producer(
      tsl::Env::Default()->StartThread({}, "producer", [&] {
        // Gives the execution below time to start waiting for its inputs.
        absl::SleepFor(absl::Milliseconds(100));
        for (auto& transfer_manager : inputs.transfer_managers) {
          CHECK_OK(transfer_manager->TransferLiteralToBuffer(0, input, [] {}));
        }
      }));
  ExecuteOptions sync_options;
  sync_options.execution_mode = ExecuteOptions::ExecutionMode::kSynchronous;
  ASSERT_OK_AND_ASSIGN(
      auto callback_results,
      callback_executable->Execute(inputs.argument_handles, sync_options));
  // Checked last: a failure report here would delay the dispatch below.
  bool finished = true;
  for (const auto& device_results : callback_results) {
    finished &= device_results[0]->GetReadyFuture().IsReady();
  }
  // Queued behind the program, this would hold the only unit of its devices.
  ASSERT_OK_AND_ASSIGN(auto queued_results,
                       queued_executable->Execute({{}, {}}, {}));
  for (const auto& device_results : queued_results) {
    EXPECT_OK(device_results[0]->GetReadyFuture().Await());
  }
  EXPECT_TRUE(finished);
  producer.reset();
  callback_body = nullptr;
}

// A host callback does not wait for a multi device synchronous execution that
// it dispatches: collectives launch one at a time, so that execution waits for
// the program running the callback. Its launches run outside the callback's
// scope, and take no pacing units either.
TEST(CpuClientHostCallbackTest,
     NestedMultiDeviceSynchronousExecutionDoesNotBlockTheCallback) {
  Watchdog watchdog;
  CpuClientOptions options;
  options.cpu_device_count = 2;
  options.max_inflight_computations_per_device = 1;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetXlaPjrtCpuClient(std::move(options)));
  ASSERT_OK_AND_ASSIGN(auto callback_executable,
                       Compile(*client, kCallbackBodyProgram, 2));
  ASSERT_OK_AND_ASSIGN(auto constant_executable,
                       Compile(*client, kConstantProgram, 2));
  ASSERT_OK_AND_ASSIGN(PendingInputs inputs, CreatePendingInputs(*client, 2));

  ExecuteOptions sync_options;
  sync_options.execution_mode = ExecuteOptions::ExecutionMode::kSynchronous;
  absl::Mutex mu;
  std::vector<std::unique_ptr<PjRtBuffer>> nested_outputs;
  std::function<absl::Status()> body = [&]() -> absl::Status {
    ABSL_ASSIGN_OR_RETURN(auto outputs,
                     constant_executable->Execute({{}, {}}, sync_options));
    absl::MutexLock lock(mu);
    for (auto& device_outputs : outputs) {
      nested_outputs.push_back(std::move(device_outputs[0]));
    }
    return absl::OkStatus();
  };
  callback_body = &body;

  // Queues a computation behind the program, which holds the only unit of
  // each device until it runs, then provides the program's inputs.
  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> queued_results;
  Literal input = LiteralUtil::CreateR0(0.0f);
  std::unique_ptr<tsl::Thread> producer(
      tsl::Env::Default()->StartThread({}, "producer", [&] {
        absl::SleepFor(absl::Milliseconds(100));
        auto results = constant_executable->Execute({{}, {}}, {});
        CHECK_OK(results.status());
        queued_results = *std::move(results);
        for (auto& transfer_manager : inputs.transfer_managers) {
          CHECK_OK(transfer_manager->TransferLiteralToBuffer(0, input, [] {}));
        }
      }));
  ASSERT_OK_AND_ASSIGN(
      auto callback_results,
      callback_executable->Execute(inputs.argument_handles, sync_options));
  producer.reset();
  for (const auto& device_results : queued_results) {
    EXPECT_OK(device_results[0]->GetReadyFuture().Await());
  }
  // Two callbacks, each with an output per device.
  ASSERT_EQ(nested_outputs.size(), 4);
  for (const auto& output : nested_outputs) {
    EXPECT_OK(output->GetReadyFuture().Await());
  }
  callback_body = nullptr;
}

}  // namespace
}  // namespace xla
