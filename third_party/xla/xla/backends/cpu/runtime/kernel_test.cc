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

#include "xla/backends/cpu/runtime/kernel.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/kernel_c_api.h"
#include "xla/runtime/workgroup_dim.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/cpu_info.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {

static XLA_CPU_KernelError* AddI32(const XLA_CPU_KernelCallFrame* call_frame) {
  const XLA_CPU_KernelArg& lhs = call_frame->args[0];
  const XLA_CPU_KernelArg& rhs = call_frame->args[1];
  const XLA_CPU_KernelArg& out = call_frame->args[2];

  int32_t* lhs_ptr = reinterpret_cast<int32_t*>(lhs.data);
  int32_t* rhs_ptr = reinterpret_cast<int32_t*>(rhs.data);
  int32_t* out_ptr = reinterpret_cast<int32_t*>(out.data);

  int64_t zstep = call_frame->workgroup_dim->x * call_frame->workgroup_dim->y;
  int64_t ystep = call_frame->workgroup_dim->x;

  int64_t i = call_frame->workgroup_id->x +
              call_frame->workgroup_id->y * ystep +
              call_frame->workgroup_id->z * zstep;
  *(out_ptr + i) = *(lhs_ptr + i) + *(rhs_ptr + i);

  return nullptr;
}

TEST(KernelTest, InternalAddition1D) {
  Kernel kernel(/*arity=*/3, AddI32);

  std::vector<int32_t> lhs = {1, 2, 3, 4};
  std::vector<int32_t> rhs = {5, 6, 7, 8};
  std::vector<int32_t> out = {0, 0, 0, 0};

  Kernel::DeviceMemoryBase lhs_mem(lhs.data(), lhs.size() * sizeof(int32_t));
  Kernel::DeviceMemoryBase rhs_mem(rhs.data(), rhs.size() * sizeof(int32_t));
  Kernel::DeviceMemoryBase out_mem(out.data(), out.size() * sizeof(int32_t));
  std::vector<Kernel::DeviceMemoryBase> args = {lhs_mem, rhs_mem, out_mem};

  TF_ASSERT_OK(kernel.Launch(WorkgroupDim{4, 1, 1}, args));

  std::vector<int32_t> expected = {6, 8, 10, 12};
  EXPECT_EQ(out, expected);
}

TEST(KernelTest, InternalAddition3D) {
  Kernel kernel(/*arity=*/3, AddI32);

  // Lets pretend there is a 3-dimensional 2x2x3 data
  std::vector<int32_t> lhs = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int32_t> rhs = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
  std::vector<int32_t> out = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  Kernel::DeviceMemoryBase lhs_mem(lhs.data(), lhs.size() * sizeof(int32_t));
  Kernel::DeviceMemoryBase rhs_mem(rhs.data(), rhs.size() * sizeof(int32_t));
  Kernel::DeviceMemoryBase out_mem(out.data(), out.size() * sizeof(int32_t));
  std::vector<Kernel::DeviceMemoryBase> args = {lhs_mem, rhs_mem, out_mem};

  TF_ASSERT_OK(kernel.Launch(WorkgroupDim{2, 2, 3}, args));

  std::vector<int32_t> expected = {11, 13, 15, 17, 19, 21,
                                   23, 25, 27, 29, 31, 33};
  EXPECT_EQ(out, expected);
}

TEST(KernelTest, LaunchAsync) {
  auto thread_pool = std::make_shared<tsl::thread::ThreadPool>(
      tsl::Env::Default(), "test", tsl::port::MaxParallelism());
  Eigen::ThreadPoolDevice device(thread_pool->AsEigenThreadPool(),
                                 thread_pool->NumThreads());

  std::atomic<int32_t> num_tasks = 0;
  XLA_CPU_KernelArg arg = {&num_tasks, sizeof(num_tasks)};

  auto* no_op = +[](const XLA_CPU_KernelCallFrame* call_frame) {
    auto* n = reinterpret_cast<std::atomic<int32_t>*>(call_frame->args[0].data);
    n->fetch_add(1);
    return static_cast<XLA_CPU_KernelError*>(nullptr);
  };

  Kernel host_kernel(/*arity=*/1, no_op);
  auto event = host_kernel.Launch(WorkgroupDim{4, 4, 4}, {arg}, &device);

  tsl::BlockUntilReady(event);
  EXPECT_TRUE(event.IsConcrete());
  EXPECT_EQ(num_tasks.load(), 4 * 4 * 4);
}

TEST(KernelTest, LaunchAsyncError) {
  auto thread_pool = std::make_shared<tsl::thread::ThreadPool>(
      tsl::Env::Default(), "test", tsl::port::MaxParallelism());
  Eigen::ThreadPoolDevice device(thread_pool->AsEigenThreadPool(),
                                 thread_pool->NumThreads());

  // XLA_CPU_KernelError type is not defined so we simply return a non-nullptr
  // pointer to signal error to the runtime.
  auto* maybe_error = +[](const XLA_CPU_KernelCallFrame* call_frame) {
    if (call_frame->workgroup_id->x == 2 && call_frame->workgroup_id->z == 2) {
      return reinterpret_cast<XLA_CPU_KernelError*>(0xDEADBEEF);
    }
    return static_cast<XLA_CPU_KernelError*>(nullptr);
  };

  Kernel host_kernel(/*arity=*/0, maybe_error);
  auto event = host_kernel.Launch(
      WorkgroupDim{4, 4, 4}, absl::Span<const XLA_CPU_KernelArg>(), &device);

  tsl::BlockUntilReady(event);
  ASSERT_TRUE(event.IsError());
  EXPECT_TRUE(absl::StrContains(event.GetError().message(),
                                "Failed to call host kernel:"));
}

//===----------------------------------------------------------------------===//
// Performance benchmarks below
//===----------------------------------------------------------------------===//

// We benchmark Kernel launch overheads so we use a noop kernel as we are
// only interested on how fast we can launch kernel tasks.
static XLA_CPU_KernelError* NoOp(const XLA_CPU_KernelCallFrame*) {
  return nullptr;
}

static void BM_KernelSyncLaunch(benchmark::State& state) {
  uint64_t dim_x = state.range(0);

  Kernel kernel(/*arity=*/0, NoOp);
  absl::Span<const XLA_CPU_KernelArg> args;

  for (auto _ : state) {
    benchmark::DoNotOptimize(kernel.Launch(WorkgroupDim{dim_x, 1, 1}, args));
  }
}

static void BM_KernelAsyncLaunch(benchmark::State& state) {
  uint64_t dim_x = state.range(0);

  auto thread_pool = std::make_shared<tsl::thread::ThreadPool>(
      tsl::Env::Default(), "benchmark", tsl::port::MaxParallelism());
  Eigen::ThreadPoolDevice device(thread_pool->AsEigenThreadPool(),
                                 thread_pool->NumThreads());

  Kernel kernel(/*arity=*/0, NoOp);
  absl::Span<const XLA_CPU_KernelArg> args;

  for (auto _ : state) {
    auto event = kernel.Launch(WorkgroupDim{dim_x, 1, 1}, args, &device);
    tsl::BlockUntilReady(event);
  }
}

BENCHMARK(BM_KernelSyncLaunch)
    ->MeasureProcessCPUTime()
    ->Arg(1)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256);

BENCHMARK(BM_KernelAsyncLaunch)
    ->MeasureProcessCPUTime()
    ->Arg(1)
    ->Arg(4)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256);

}  // namespace xla::cpu
