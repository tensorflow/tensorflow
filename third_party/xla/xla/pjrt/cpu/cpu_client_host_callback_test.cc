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

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "xla/ffi/api/ffi.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

// Executed by the host callback below on the callback's thread, as JAX's
// Python CPU callbacks do. Set by the test.
PjRtLoadedExecutable* nested_executable = nullptr;

absl::StatusOr<float> RunNestedExecution() {
  HostCallbackScope host_callback_scope;
  // JAX gives each dispatching thread its own execution stream. On the
  // caller's stream the nested execution would also wait for the computation
  // queued behind the callback.
  ExecuteOptions options;
  options.execution_stream_id = 1;
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

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> Compile(
    PjRtClient& client, absl::string_view program) {
  ABSL_ASSIGN_OR_RETURN(auto module, ParseAndReturnUnverifiedModule(program));
  return client.CompileAndLoad(XlaComputation(module->ToProto()), {});
}

TEST(CpuClientHostCallbackTest,
     NestedExecutionDoesNotWaitForComputationsQueuedBehindTheCallback) {
  CpuClientOptions options;
  options.max_inflight_computations_per_device = 2;
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<PjRtClient> client,
                       GetPjRtCpuClient(std::move(options)));
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

}  // namespace
}  // namespace xla
