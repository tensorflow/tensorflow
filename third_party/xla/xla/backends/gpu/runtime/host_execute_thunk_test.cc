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

#include "xla/backends/gpu/runtime/host_execute_thunk.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/host_offloading/host_offloading_executable.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace gpu {
namespace {

se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

TEST(HostExecuteStartThunkTest, SingleArgSingleResult) {
  se::StreamExecutor* stream_executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  static constexpr char const* kHloModule = R"(
    HloModule module
    ENTRY add_inplace {
      p0 = s32[] parameter(0)
      ROOT add = s32[] add(p0, p0)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(kHloModule, {}));

  se::DeviceMemoryBase arg = stream_executor->Allocate(1 * sizeof(int32_t));
  se::DeviceMemoryBase result = stream_executor->Allocate(1 * sizeof(int32_t));

  TF_ASSERT_OK(stream->Memset32(&arg, 5, 4));
  TF_ASSERT_OK(stream->MemZero(&result, 4));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_arg(/*index=*/0, 4, /*color=*/0);
  BufferAllocation alloc_result(/*index=*/1, 4, /*color=*/0);

  BufferAllocation::Slice slice_arg(&alloc_arg, 0, 4);
  BufferAllocation::Slice slice_result(&alloc_result, 0, 4);

  HostExecuteStartThunk thunk(Thunk::ThunkInfo(), *hlo_module,
                              {{slice_arg, ShapeUtil::MakeShape(S32, {})}},
                              {{slice_result, ShapeUtil::MakeShape(S32, {})}});

  se::StreamExecutorMemoryAllocator allocator(stream_executor);
  ExecutableRunOptions executable_run_options;
  executable_run_options.set_device_to_host_stream(stream.get());
  executable_run_options.set_host_to_device_stream(stream.get());
  ServiceExecutableRunOptions service_executable_run_options(
      executable_run_options);

  BufferAllocations allocations({arg, result}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      service_executable_run_options, allocations, stream.get(), stream.get(),
      nullptr, nullptr);

  TF_ASSERT_OK(
      thunk.Initialize(Thunk::InitializeParams{/*executor=*/stream_executor}));
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));

  TF_ASSERT_OK_AND_ASSIGN(auto execute_event,
                          thunk.async_events()->ExtractEvent(
                              stream_executor, RunId(params.execution_id)));

  tsl::BlockUntilReady(execute_event);
  EXPECT_FALSE(execute_event.IsError());
  TF_ASSERT_OK(stream->WaitFor(execute_event.get().get()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  xla::Literal result_literal(ShapeUtil::MakeShape(S32, {}));
  TF_ASSERT_OK(
      stream->Memcpy(result_literal.untyped_data(), result, result.size()));
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR0<int32_t>(10),
                                     result_literal));
}

TEST(HostExecuteStartThunkTest, MultiArgMultipleResult) {
  se::StreamExecutor* stream_executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  static constexpr char const* kHloModule = R"(
    HloModule module
    ENTRY add_inplace {
      p0 = s32[] parameter(0)
      p1 = s32[] parameter(1)
      add = s32[] add(p0, p1)
      mul = s32[] multiply(p0, p1)
      ROOT tuple = (s32[], s32[]) tuple(add, mul)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(kHloModule, {}));

  se::DeviceMemoryBase arg0 = stream_executor->Allocate(1 * sizeof(int32_t));
  se::DeviceMemoryBase arg1 = stream_executor->Allocate(1 * sizeof(int32_t));
  se::DeviceMemoryBase result0 = stream_executor->Allocate(1 * sizeof(int32_t));
  se::DeviceMemoryBase result1 = stream_executor->Allocate(1 * sizeof(int32_t));

  TF_ASSERT_OK(stream->Memset32(&arg0, 5, 4));
  TF_ASSERT_OK(stream->Memset32(&arg1, 3, 4));
  TF_ASSERT_OK(stream->MemZero(&result0, 4));
  TF_ASSERT_OK(stream->MemZero(&result1, 4));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_arg0(/*index=*/0, 4, /*color=*/0);
  BufferAllocation alloc_result0(/*index=*/1, 4, /*color=*/0);
  BufferAllocation alloc_arg1(/*index=*/2, 4, /*color=*/0);
  BufferAllocation alloc_result1(/*index=*/3, 4, /*color=*/0);

  BufferAllocation::Slice slice_arg0(&alloc_arg0, 0, 4);
  BufferAllocation::Slice slice_result0(&alloc_result0, 0, 4);
  BufferAllocation::Slice slice_arg1(&alloc_arg1, 0, 4);
  BufferAllocation::Slice slice_result1(&alloc_result1, 0, 4);

  HostExecuteStartThunk thunk(Thunk::ThunkInfo(), *hlo_module,
                              {{slice_arg0, ShapeUtil::MakeShape(S32, {})},
                               {slice_arg1, ShapeUtil::MakeShape(S32, {})}},
                              {{slice_result0, ShapeUtil::MakeShape(S32, {})},
                               {slice_result1, ShapeUtil::MakeShape(S32, {})}});

  se::StreamExecutorMemoryAllocator allocator(stream_executor);
  ExecutableRunOptions executable_run_options;
  executable_run_options.set_device_to_host_stream(stream.get());
  executable_run_options.set_host_to_device_stream(stream.get());
  ServiceExecutableRunOptions service_executable_run_options(
      executable_run_options);
  BufferAllocations allocations({arg0, result0, arg1, result1}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      service_executable_run_options, allocations, stream.get(), stream.get(),
      nullptr, nullptr);

  TF_ASSERT_OK(
      thunk.Initialize(Thunk::InitializeParams{/*executor=*/stream_executor}));
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));

  TF_ASSERT_OK_AND_ASSIGN(auto execute_event,
                          thunk.async_events()->ExtractEvent(
                              stream_executor, RunId(params.execution_id)));

  tsl::BlockUntilReady(execute_event);
  EXPECT_FALSE(execute_event.IsError());
  TF_ASSERT_OK(stream->WaitFor(execute_event.get().get()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  xla::Literal result_literal0(ShapeUtil::MakeShape(S32, {}));
  TF_ASSERT_OK(
      stream->Memcpy(result_literal0.untyped_data(), result0, result0.size()));
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR0<int32_t>(8),
                                     result_literal0));

  xla::Literal result_literal1(ShapeUtil::MakeShape(S32, {}));
  TF_ASSERT_OK(
      stream->Memcpy(result_literal1.untyped_data(), result1, result1.size()));
  EXPECT_TRUE(LiteralTestUtil::Equal(LiteralUtil::CreateR0<int32_t>(15),
                                     result_literal1));
}

TEST(HostExecuteStartThunkTest, ArgAndResultPinnedOnHost) {
  se::StreamExecutor* stream_executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  static constexpr char const* kHloModule = R"(
    HloModule module
    ENTRY add_inplace {
      p0 = s32[] parameter(0)
      ROOT add = s32[] add(p0, p0)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(kHloModule, {}));

  TF_ASSERT_OK_AND_ASSIGN(
      auto arg_memory_allocation,
      stream_executor->HostMemoryAllocate(1 * sizeof(int32_t)));

  constexpr int32_t kArgValue = 5;
  std::memcpy(arg_memory_allocation->opaque(), &kArgValue, sizeof(kArgValue));

  TF_ASSERT_OK_AND_ASSIGN(
      auto result_memory_allocation,
      stream_executor->HostMemoryAllocate(1 * sizeof(int32_t)));

  se::DeviceMemoryBase arg(arg_memory_allocation->opaque(),
                           arg_memory_allocation->size());
  se::DeviceMemoryBase result(result_memory_allocation->opaque(),
                              result_memory_allocation->size());

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_arg(/*index=*/0, 4, /*color=*/0);
  BufferAllocation alloc_result(/*index=*/1, 4, /*color=*/0);

  BufferAllocation::Slice slice_arg(&alloc_arg, 0, 4);
  BufferAllocation::Slice slice_result(&alloc_result, 0, 4);

  HostExecuteStartThunk thunk(Thunk::ThunkInfo(), *hlo_module,
                              {{slice_arg, ShapeUtil::MakeShape(S32, {})}},
                              {{slice_result, ShapeUtil::MakeShape(S32, {})}});

  se::StreamExecutorMemoryAllocator allocator(stream_executor);
  ExecutableRunOptions executable_run_options;
  executable_run_options.set_device_to_host_stream(stream.get());
  executable_run_options.set_host_to_device_stream(stream.get());
  ServiceExecutableRunOptions service_executable_run_options(
      executable_run_options);
  BufferAllocations allocations({arg, result}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      service_executable_run_options, allocations, stream.get(), stream.get(),
      nullptr, nullptr);

  TF_ASSERT_OK(
      thunk.Initialize(Thunk::InitializeParams{/*executor=*/stream_executor}));
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));

  TF_ASSERT_OK_AND_ASSIGN(auto execute_event,
                          thunk.async_events()->ExtractEvent(
                              stream_executor, RunId(params.execution_id)));
  tsl::BlockUntilReady(execute_event);
  EXPECT_FALSE(execute_event.IsError());
  TF_ASSERT_OK(stream->WaitFor(execute_event.get().get()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  EXPECT_EQ(*static_cast<int32_t*>(result_memory_allocation->opaque()), 10);
}

TEST(HostExecuteStartThunkTest, ArgAndResultNonRegisteredHostMemory) {
  se::StreamExecutor* stream_executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  static constexpr char const* kHloModule = R"(
    HloModule module
    ENTRY add_inplace {
      p0 = s32[] parameter(0)
      ROOT add = s32[] add(p0, p0)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(kHloModule, {}));

  alignas(xla::cpu::Align()) int32_t arg_value = 5;
  alignas(xla::cpu::Align()) int32_t result_value = 0;

  se::DeviceMemoryBase arg(&arg_value, sizeof(int32_t));
  se::DeviceMemoryBase result(&result_value, sizeof(int32_t));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_arg(/*index=*/0, 4, /*color=*/0);
  BufferAllocation alloc_result(/*index=*/1, 4, /*color=*/0);

  BufferAllocation::Slice slice_arg(&alloc_arg, 0, 4);
  BufferAllocation::Slice slice_result(&alloc_result, 0, 4);

  HostExecuteStartThunk thunk(Thunk::ThunkInfo(), *hlo_module,
                              {{slice_arg, ShapeUtil::MakeShape(S32, {})}},
                              {{slice_result, ShapeUtil::MakeShape(S32, {})}});

  se::StreamExecutorMemoryAllocator allocator(stream_executor);
  ExecutableRunOptions executable_run_options;
  executable_run_options.set_device_to_host_stream(stream.get());
  executable_run_options.set_host_to_device_stream(stream.get());
  ServiceExecutableRunOptions service_executable_run_options(
      executable_run_options);
  BufferAllocations allocations({arg, result}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      service_executable_run_options, allocations, stream.get(), stream.get(),
      nullptr, nullptr);

  TF_ASSERT_OK(
      thunk.Initialize(Thunk::InitializeParams{/*executor=*/stream_executor}));
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));

  TF_ASSERT_OK_AND_ASSIGN(auto execute_event,
                          thunk.async_events()->ExtractEvent(
                              stream_executor, RunId(params.execution_id)));
  tsl::BlockUntilReady(execute_event);
  EXPECT_FALSE(execute_event.IsError());
  TF_ASSERT_OK(stream->WaitFor(execute_event.get().get()));
  TF_ASSERT_OK(stream->BlockHostUntilDone());

  EXPECT_EQ(result_value, 10);
}

TEST(HostExecuteStartThunkTest, TestErrorPropagationFromExecuteEvent) {
#ifdef NDEBUG
  GTEST_SKIP() << "Skipping test in optimized mode because XLA:CPU won't "
                  "check for the alignment error.";
  return;
#endif
  se::StreamExecutor* stream_executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  static constexpr char const* kHloModule = R"(
    HloModule module
    ENTRY add_inplace {
      p0 = s32[] parameter(0)
      ROOT add = s32[] add(p0, p0)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_module,
                          ParseAndReturnUnverifiedModule(kHloModule, {}));

  // Don't align on purpose. XLA:CPU expects the arguments to be aligned. This
  // will cause an internal error in XLA:CPU which should be propagated to the
  // event.
  int32_t arg_value = 5;
  int32_t result_value = 0;

  se::DeviceMemoryBase arg(&arg_value, sizeof(int32_t));
  se::DeviceMemoryBase result(&result_value, sizeof(int32_t));

  // Prepare buffer allocations for recording command buffer.
  BufferAllocation alloc_arg(/*index=*/0, 4, /*color=*/0);
  BufferAllocation alloc_result(/*index=*/1, 4, /*color=*/0);

  BufferAllocation::Slice slice_arg(&alloc_arg, 0, 4);
  BufferAllocation::Slice slice_result(&alloc_result, 0, 4);

  HostExecuteStartThunk thunk(Thunk::ThunkInfo(), *hlo_module,
                              {{slice_arg, ShapeUtil::MakeShape(S32, {})}},
                              {{slice_result, ShapeUtil::MakeShape(S32, {})}});

  se::StreamExecutorMemoryAllocator allocator(stream_executor);
  ExecutableRunOptions executable_run_options;
  executable_run_options.set_device_to_host_stream(stream.get());
  executable_run_options.set_host_to_device_stream(stream.get());
  ServiceExecutableRunOptions service_executable_run_options(
      executable_run_options);
  BufferAllocations allocations({arg, result}, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      service_executable_run_options, allocations, stream.get(), stream.get(),
      nullptr, nullptr);

  TF_ASSERT_OK(
      thunk.Initialize(Thunk::InitializeParams{/*executor=*/stream_executor}));
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));

  TF_ASSERT_OK_AND_ASSIGN(auto execute_event,
                          thunk.async_events()->ExtractEvent(
                              stream_executor, RunId(params.execution_id)));
  tsl::BlockUntilReady(execute_event);
  EXPECT_TRUE(execute_event.IsError());
}

TEST(HostExecuteDoneThunkTest, WaitingOnAvailableEvent) {
  se::StreamExecutor* stream_executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  auto async_events = std::make_shared<HostExecuteAsyncEvents>();

  HostExecuteDoneThunk thunk(Thunk::ThunkInfo(), async_events);
  ExecutableRunOptions executable_run_options;
  executable_run_options.set_device_to_host_stream(stream.get());
  executable_run_options.set_host_to_device_stream(stream.get());
  ServiceExecutableRunOptions service_executable_run_options(
      executable_run_options);

  BufferAllocations allocations({}, 0, nullptr);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      service_executable_run_options, allocations, stream.get(), stream.get(),
      nullptr, nullptr);
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto available_event,
        async_events->CreateEvent(stream_executor, RunId(params.execution_id)));

    available_event.SetStateConcrete();
  }

  TF_ASSERT_OK(
      thunk.Initialize(Thunk::InitializeParams{/*executor=*/stream_executor}));
  TF_ASSERT_OK(thunk.ExecuteOnStream(params));
}

TEST(HostExecuteDoneThunkTest, WaitingOnErrorEvent) {
  se::StreamExecutor* stream_executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  auto async_events = std::make_shared<HostExecuteAsyncEvents>();

  HostExecuteDoneThunk thunk(Thunk::ThunkInfo(), async_events);
  ExecutableRunOptions executable_run_options;
  executable_run_options.set_device_to_host_stream(stream.get());
  executable_run_options.set_host_to_device_stream(stream.get());
  ServiceExecutableRunOptions service_executable_run_options(
      executable_run_options);

  BufferAllocations allocations({}, 0, nullptr);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      service_executable_run_options, allocations, stream.get(), stream.get(),
      nullptr, nullptr);
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto error_event,
        async_events->CreateEvent(stream_executor, RunId(params.execution_id)));
    error_event.SetError(Internal("Test error"));
  }

  TF_ASSERT_OK(
      thunk.Initialize(Thunk::InitializeParams{/*executor=*/stream_executor}));
  EXPECT_THAT(thunk.ExecuteOnStream(params),
              absl_testing::StatusIs(absl::StatusCode::kInternal));
}

}  // namespace

}  // namespace gpu
}  // namespace xla
