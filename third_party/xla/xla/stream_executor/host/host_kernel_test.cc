/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/host/host_kernel.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/host/host_kernel_c_api.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/env.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/platform/test_benchmark.h"
#include "tsl/platform/threadpool.h"

namespace stream_executor::host {

static SE_HOST_KernelError* AddI32(const SE_HOST_KernelCallFrame* call_frame) {
  const SE_HOST_KernelArg& lhs = call_frame->args[0];
  const SE_HOST_KernelArg& rhs = call_frame->args[1];
  const SE_HOST_KernelArg& out = call_frame->args[2];

  int32_t* lhs_ptr = reinterpret_cast<int32_t*>(lhs.data);
  int32_t* rhs_ptr = reinterpret_cast<int32_t*>(rhs.data);
  int32_t* out_ptr = reinterpret_cast<int32_t*>(out.data);

  const auto zstep = call_frame->thread_dims->x * call_frame->thread_dims->y;
  const auto ystep = call_frame->thread_dims->x;

  uint64_t i = call_frame->thread->x + call_frame->thread->y * ystep +
               call_frame->thread->z * zstep;
  *(out_ptr + i) = *(lhs_ptr + i) + *(rhs_ptr + i);

  return nullptr;
}

static const char* llvm_kernel_add = R"(
%SE_HOST_KernelCallFrame = type { ptr, ptr, i64, ptr }
%struct.SE_HOST_KernelArg = type { ptr, i64 }

define ptr @LlvmAddI32(ptr noundef %0) {
  %2 = getelementptr inbounds %SE_HOST_KernelCallFrame, ptr %0, i32 0, i32 3
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds %struct.SE_HOST_KernelArg, ptr %3, i64 1
  %5 = getelementptr inbounds %struct.SE_HOST_KernelArg, ptr %3, i64 2
  %6 = load ptr, ptr %3, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = load ptr, ptr %5, align 8
  %9 = getelementptr inbounds %SE_HOST_KernelCallFrame, ptr %0, i32 0, i32 1
  %10 = load ptr, ptr %9, align 8
  %11 = load i64, ptr %10, align 8
  %12 = getelementptr inbounds i32, ptr %6, i64 %11
  %13 = load i32, ptr %12, align 4
  %14 = getelementptr inbounds i32, ptr %7, i64 %11
  %15 = load i32, ptr %14, align 4
  %16 = add nsw i32 %13, %15
  %17 = getelementptr inbounds i32, ptr %8, i64 %11
  store i32 %16, ptr %17, align 4
  ret ptr null
}
)";

static absl::StatusOr<StreamExecutor*> NewStreamExecutor() {
  TF_ASSIGN_OR_RETURN(auto platform, PlatformManager::PlatformWithName("Host"));
  TF_ASSIGN_OR_RETURN(auto stream_exec,
                      platform->ExecutorForDevice(/*ordinal=*/0));
  return stream_exec;
}

TEST(HostKernelTest, InternalAddition1D) {
  auto tp = std::make_shared<tsl::thread::ThreadPool>(tsl::Env::Default(),
                                                      "XLAEigen", 2);

  HostKernel kernel(/*arity=*/3, AddI32, tp);

  std::vector<int32_t> lhs = {1, 2, 3, 4};
  std::vector<int32_t> rhs = {5, 6, 7, 8};
  std::vector<int32_t> out = {0, 0, 0, 0};

  DeviceMemoryBase lhs_mem(lhs.data(), lhs.size() * sizeof(int32_t));
  DeviceMemoryBase rhs_mem(rhs.data(), rhs.size() * sizeof(int32_t));
  DeviceMemoryBase out_mem(out.data(), out.size() * sizeof(int32_t));
  std::vector<DeviceMemoryBase> args = {lhs_mem, rhs_mem, out_mem};

  TF_ASSERT_OK(kernel.Launch(ThreadDim(4), args));

  std::vector<int32_t> expected = {6, 8, 10, 12};
  EXPECT_EQ(out, expected);
}

TEST(HostKernelTest, InternalAddition3D) {
  auto tp = std::make_shared<tsl::thread::ThreadPool>(tsl::Env::Default(),
                                                      "XLAEigen", 2);

  HostKernel kernel(/*arity=*/3, AddI32, tp);

  // Lets pretend there is a 3-dimensional 2x2x3 data
  std::vector<int32_t> lhs = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int32_t> rhs = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
  std::vector<int32_t> out = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  DeviceMemoryBase lhs_mem(lhs.data(), lhs.size() * sizeof(int32_t));
  DeviceMemoryBase rhs_mem(rhs.data(), rhs.size() * sizeof(int32_t));
  DeviceMemoryBase out_mem(out.data(), out.size() * sizeof(int32_t));
  std::vector<DeviceMemoryBase> args = {lhs_mem, rhs_mem, out_mem};

  TF_ASSERT_OK(kernel.Launch(ThreadDim(2, 2, 3), args));

  std::vector<int32_t> expected = {11, 13, 15, 17, 19, 21,
                                   23, 25, 27, 29, 31, 33};
  EXPECT_EQ(out, expected);
}

TEST(HostKernelTest, Addition3D) {
  // Lets pretend there is a 3-dimensional 2x2x3 data
  std::vector<int32_t> lhs = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int32_t> rhs = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
  std::vector<int32_t> out = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  DeviceMemoryBase lhs_mem(lhs.data(), lhs.size() * sizeof(int32_t));
  DeviceMemoryBase rhs_mem(rhs.data(), rhs.size() * sizeof(int32_t));
  DeviceMemoryBase out_mem(out.data(), out.size() * sizeof(int32_t));
  std::vector<DeviceMemoryBase> args = {lhs_mem, rhs_mem, out_mem};

  MultiKernelLoaderSpec spec(/*arity=*/3);
  spec.AddInProcessSymbol(reinterpret_cast<void*>(AddI32), "Addition_kernel");

  TF_ASSERT_OK_AND_ASSIGN(auto executor, NewStreamExecutor());
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  TF_ASSERT_OK_AND_ASSIGN(auto add, executor->LoadKernel(spec));

  const KernelArgsDeviceMemoryArray kargs{args, /*shared_memory_bytes=*/0};
  TF_ASSERT_OK(stream->Launch(ThreadDim(2, 2, 3), BlockDim(1), *add, kargs));

  std::vector<int32_t> expected = {11, 13, 15, 17, 19, 21,
                                   23, 25, 27, 29, 31, 33};
  EXPECT_EQ(out, expected);
}

TEST(HostKernelTest, JitAddition) {
  std::vector<int32_t> lhs = {1, 2, 3, 4};
  std::vector<int32_t> rhs = {5, 6, 7, 8};
  std::vector<int32_t> out = {0, 0, 0, 0};

  DeviceMemoryBase lhs_mem(lhs.data(), lhs.size() * sizeof(int32_t));
  DeviceMemoryBase rhs_mem(rhs.data(), rhs.size() * sizeof(int32_t));
  DeviceMemoryBase out_mem(out.data(), out.size() * sizeof(int32_t));
  std::vector<DeviceMemoryBase> args = {lhs_mem, rhs_mem, out_mem};

  MultiKernelLoaderSpec spec(/*arity=*/3);
  spec.AddLlvmHostKernel(llvm_kernel_add, "LlvmAddI32", "LlvmAddI32",
                         absl::Span<std::string>());

  TF_ASSERT_OK_AND_ASSIGN(auto executor, NewStreamExecutor());
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  TF_ASSERT_OK_AND_ASSIGN(auto add, executor->LoadKernel(spec));

  const KernelArgsDeviceMemoryArray kargs{args, /*shared_memory_bytes=*/0};
  TF_ASSERT_OK(stream->Launch(ThreadDim(4), BlockDim(1), *add, kargs));

  std::vector<int32_t> expected = {6, 8, 10, 12};
  EXPECT_EQ(out, expected);
}

TEST(HostKernelTest, LaunchAsync) {
  auto* no_op = +[](const SE_HOST_KernelCallFrame*) {
    return static_cast<SE_HOST_KernelError*>(nullptr);
  };

  auto thread_pool = std::make_shared<tsl::thread::ThreadPool>(
      tsl::Env::Default(), "benchmark", tsl::port::MaxParallelism());

  std::atomic<size_t> num_tasks = 0;

  HostKernel::TaskRunner runner = [&](HostKernel::Task task) {
    num_tasks.fetch_add(1, std::memory_order_relaxed);
    thread_pool->Schedule(std::move(task));
  };

  HostKernel host_kernel(/*arity=*/0, no_op);
  auto event = host_kernel.Launch(ThreadDim(4, 4, 4),
                                  absl::Span<const SE_HOST_KernelArg>(),
                                  std::move(runner));

  tsl::BlockUntilReady(event);
  EXPECT_TRUE(event.IsConcrete());
  EXPECT_EQ(num_tasks.load(std::memory_order_relaxed), 4 * 4 * 4 - 1);
}

TEST(HostKernelTest, LaunchAsyncError) {
  // SE_HOST_KernelError type is not defined so we simply return a non-nullptr
  // pointer to signal error to the runtime.
  auto* maybe_error = +[](const SE_HOST_KernelCallFrame* call_frame) {
    if (call_frame->thread->x == 2 && call_frame->thread->z == 2) {
      return reinterpret_cast<SE_HOST_KernelError*>(0xDEADBEEF);
    }
    return static_cast<SE_HOST_KernelError*>(nullptr);
  };

  auto thread_pool = std::make_shared<tsl::thread::ThreadPool>(
      tsl::Env::Default(), "benchmark", tsl::port::MaxParallelism());

  std::atomic<size_t> num_tasks = 0;

  HostKernel::TaskRunner runner = [&](HostKernel::Task task) {
    num_tasks.fetch_add(1, std::memory_order_relaxed);
    thread_pool->Schedule(std::move(task));
  };

  HostKernel host_kernel(/*arity=*/0, maybe_error);
  auto event = host_kernel.Launch(ThreadDim(4, 4, 4),
                                  absl::Span<const SE_HOST_KernelArg>(),
                                  std::move(runner));

  tsl::BlockUntilReady(event);
  ASSERT_TRUE(event.IsError());
  EXPECT_TRUE(absl::StrContains(event.GetError().message(),
                                "Failed to call host kernel:"));
  EXPECT_EQ(num_tasks.load(std::memory_order_relaxed), 4 * 4 * 4 - 1);
}

//===----------------------------------------------------------------------===//
// Performance benchmarks below
//===----------------------------------------------------------------------===//

// We benchmark HostKernel launch overheads so we use a noop kernel as we are
// only interested on how fast we can launch kernel tasks.
static SE_HOST_KernelError* NoOp(const SE_HOST_KernelCallFrame*) {
  return nullptr;
}

static void BM_HostKernelSyncLaunch(benchmark::State& state) {
  int32_t tdim_x = state.range(0);

  HostKernel kernel(/*arity=*/0, NoOp);
  absl::Span<const SE_HOST_KernelArg> args;

  for (auto _ : state) {
    benchmark::DoNotOptimize(kernel.Launch(ThreadDim(tdim_x), args));
  }
}

static void BM_HostKernelAsyncLaunch(benchmark::State& state) {
  int32_t tdim_x = state.range(0);

  auto thread_pool = std::make_shared<tsl::thread::ThreadPool>(
      tsl::Env::Default(), "benchmark", tsl::port::MaxParallelism());

  auto task_runner = [&thread_pool](HostKernel::Task task) {
    thread_pool->Schedule(std::move(task));
  };

  HostKernel kernel(/*arity=*/0, NoOp);
  absl::Span<const SE_HOST_KernelArg> args;

  for (auto _ : state) {
    auto event = kernel.Launch(ThreadDim(tdim_x), args, task_runner);
    tsl::BlockUntilReady(event);
  }
}

BENCHMARK(BM_HostKernelSyncLaunch)
    ->MeasureProcessCPUTime()
    ->Arg(1)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64);

BENCHMARK(BM_HostKernelAsyncLaunch)
    ->MeasureProcessCPUTime()
    ->Arg(1)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64);

}  // namespace stream_executor::host
