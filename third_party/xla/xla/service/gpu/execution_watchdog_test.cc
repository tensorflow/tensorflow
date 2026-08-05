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

#include "xla/service/gpu/execution_watchdog.h"

#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

// Helpers that mimic the ExecuteThunks / ExecuteThunksImpl ownership split:
// - ExecuteThunks owns ExecutionWatchdogScope for the whole execution.
// - ExecuteThunksImpl only Arms the watchdog, then returns (async enqueue).
// With async allocators, block_host_until_done=false; the scope must outlive
// that return or the HangWatchdog guard is released too early.

absl::StatusOr<std::unique_ptr<ExecutionWatchdogScope>> CreateArmedScope(
    absl::Duration timeout, GpuExecutableRunOptions* gpu_run_options,
    bool block_host_until_done) {
  DebugOptions debug_options;
  debug_options.set_xla_gpu_execution_terminate_timeout(
      absl::FormatDuration(timeout));
  ABSL_ASSIGN_OR_RETURN(std::optional<ExecutionWatchdogScope> maybe_scope,
                   ExecutionWatchdogScope::Create(
                       &debug_options, "test_module",
                       /*device_ordinal=*/0, gpu_run_options,
                       /*stream=*/nullptr, block_host_until_done));
  if (!maybe_scope.has_value()) {
    return absl::InternalError("expected ExecutionWatchdogScope");
  }
  auto scope =
      std::make_unique<ExecutionWatchdogScope>(std::move(*maybe_scope));
  scope->Arm();
  return scope;
}

TEST(ExecutionWatchdogScopeTest, CreateReturnsNulloptWhenTimeoutDisabled) {
  DebugOptions debug_options;
  ASSERT_OK_AND_ASSIGN(
      std::optional<ExecutionWatchdogScope> scope,
      ExecutionWatchdogScope::Create(&debug_options, "test_module",
                                     /*device_ordinal=*/0,
                                     /*gpu_run_options=*/nullptr,
                                     /*stream=*/nullptr,
                                     /*block_host_until_done=*/true));
  EXPECT_FALSE(scope.has_value());
}

TEST(ExecutionWatchdogScopeTest, CreateReturnsScopeWhenTimeoutEnabled) {
  DebugOptions debug_options;
  debug_options.set_xla_gpu_execution_terminate_timeout("30s");
  ASSERT_OK_AND_ASSIGN(
      std::optional<ExecutionWatchdogScope> scope,
      ExecutionWatchdogScope::Create(&debug_options, "test_module",
                                     /*device_ordinal=*/0,
                                     /*gpu_run_options=*/nullptr,
                                     /*stream=*/nullptr,
                                     /*block_host_until_done=*/false));
  ASSERT_TRUE(scope.has_value());
  EXPECT_FALSE(scope->IsArmed());
}

// Positive: caller (ExecuteThunks) keeps the scope after dispatch returns.
// This is the ownership model that addresses the async-enqueue lifetime bug:
// ExecuteThunksImpl returns, but the watchdog guard remains alive.
TEST(ExecutionWatchdogScopeTest,
     TimeoutFiresAfterDispatchReturnsIfCallerKeepsScope) {
  constexpr absl::Duration kTimeout = absl::Milliseconds(80);

  std::atomic<bool> handler_called{false};
  absl::Notification handler_done;
  GpuExecutableRunOptions gpu_run_options;
  gpu_run_options.set_execution_timeout_handler(
      [&](absl::string_view /*action*/, absl::Duration /*timeout*/) {
        handler_called.store(true);
        handler_done.Notify();
      });

  // Async allocator path: block_host_until_done=false.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ExecutionWatchdogScope> scope,
                       CreateArmedScope(kTimeout, &gpu_run_options,
                                        /*block_host_until_done=*/false));

  // Simulate ExecuteThunksImpl returning after enqueue. The caller still owns
  // `scope`, so the HangWatchdog guard must remain active.
  EXPECT_TRUE(handler_done.WaitForNotificationWithTimeout(absl::Seconds(5)))
      << "timeout handler should fire while caller still owns the scope";
  EXPECT_TRUE(handler_called.load());
}

// HangWatchdog (not a manual call site) must invoke execution_timeout_handler
// after the configured timeout, with the expected action/timeout arguments.
// This covers the automatic path that P1 wiring tests intentionally skip by
// calling the handler directly.
TEST(ExecutionWatchdogScopeTest,
     HangWatchdogAutoInvokesExecutionTimeoutHandler) {
  constexpr absl::Duration kTimeout = absl::Milliseconds(100);

  std::atomic<int> handler_calls{0};
  std::string observed_action;
  absl::Duration observed_timeout = absl::ZeroDuration();
  absl::Notification handler_done;
  absl::Mutex mu;

  GpuExecutableRunOptions gpu_run_options;
  gpu_run_options.set_execution_timeout_handler(
      [&](absl::string_view action, absl::Duration timeout) {
        absl::MutexLock lock(mu);
        ++handler_calls;
        observed_action = std::string(action);
        observed_timeout = timeout;
        handler_done.Notify();
      });

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ExecutionWatchdogScope> scope,
                       CreateArmedScope(kTimeout, &gpu_run_options,
                                        /*block_host_until_done=*/false));

  // Handler must not be invoked before the timeout window.
  absl::SleepFor(absl::Milliseconds(20));
  EXPECT_EQ(handler_calls.load(), 0);

  // Do not call execution_timeout_handler() manually. HangWatchdog alone must
  // drive the callback once the deadline expires.
  EXPECT_TRUE(handler_done.WaitForNotificationWithTimeout(absl::Seconds(5)))
      << "HangWatchdog should auto-invoke execution_timeout_handler";

  absl::MutexLock lock(mu);
  EXPECT_EQ(handler_calls.load(), 1);
  EXPECT_THAT(observed_action, ::testing::HasSubstr("XLA GPU execution"));
  EXPECT_EQ(observed_timeout, kTimeout);
}

// Negative: releasing the scope at dispatch return (old stack-local guard
// behavior) cancels the watchdog before the timeout can fire.
TEST(ExecutionWatchdogScopeTest,
     TimeoutDoesNotFireIfScopeReleasedAtDispatchReturn) {
  constexpr absl::Duration kTimeout = absl::Milliseconds(200);

  std::atomic<int> handler_calls{0};
  GpuExecutableRunOptions gpu_run_options;
  gpu_run_options.set_execution_timeout_handler(
      [&](absl::string_view /*action*/, absl::Duration /*timeout*/) {
        handler_calls.fetch_add(1);
      });

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<ExecutionWatchdogScope> scope,
                       CreateArmedScope(kTimeout, &gpu_run_options,
                                        /*block_host_until_done=*/false));

  // Simulate the old bug: guard lifetime ends when ExecuteThunksImpl returns.
  scope.reset();

  // Wait well past the configured timeout. If the guard was correctly dropped,
  // the handler must not run.
  absl::SleepFor(kTimeout * 3);
  EXPECT_EQ(handler_calls.load(), 0)
      << "releasing the scope at dispatch return must drop the watchdog "
         "guard before timeout";
}

TEST(ExecutionWatchdogScopeTest, ArmIsIdempotent) {
  DebugOptions debug_options;
  debug_options.set_xla_gpu_execution_terminate_timeout("30s");

  ASSERT_OK_AND_ASSIGN(
      std::optional<ExecutionWatchdogScope> scope,
      ExecutionWatchdogScope::Create(&debug_options, "test_module",
                                     /*device_ordinal=*/0,
                                     /*gpu_run_options=*/nullptr,
                                     /*stream=*/nullptr,
                                     /*block_host_until_done=*/true));
  ASSERT_TRUE(scope.has_value());
  scope->Arm();
  EXPECT_TRUE(scope->IsArmed());
  scope->Arm();
  EXPECT_TRUE(scope->IsArmed());
}

}  // namespace
}  // namespace xla::gpu
