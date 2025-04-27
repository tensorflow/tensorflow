/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/stream_executor/kernel.h"

#include <cstdint>
#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>

#include "absl/types/span.h"
#include "benchmark/benchmark.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/kernel_spec.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/typed_kernel_factory.h"
#include "tsl/platform/test.h"
#include "tsl/platform/test_benchmark.h"

namespace stream_executor {

// Struct for testing custom kernel arguments with C++ structs.
struct Data {};

// Compile time checks to make sure that we correctly infer the storage type
// from packed arguments.
template <typename... Args>
using ArgsStorage = typename KernelArgsPackedTuple<Args...>::Storage;

// We automatically remove const and reference from integral arguments types.
static_assert(
    std::is_same_v<ArgsStorage<int32_t, const int32_t, int32_t&, const int32_t>,
                   std::tuple<int32_t, int32_t, int32_t, int32_t>>);

// We automatically remove const and reference from struct arguments types.
static_assert(std::is_same_v<ArgsStorage<Data, const Data, Data&, const Data>,
                             std::tuple<Data, Data, Data, Data>>);

// We pass DeviceMemoryBase as an opaque pointer.
static_assert(std::is_same_v<
              ArgsStorage<DeviceMemoryBase, const DeviceMemoryBase,
                          DeviceMemoryBase&, const DeviceMemoryBase&>,
              std::tuple<const void*, const void*, const void*, const void*>>);

// We pass DeviceMemory<T> as an opaque pointer.
static_assert(std::is_same_v<
              ArgsStorage<DeviceMemory<float>, const DeviceMemory<float>,
                          DeviceMemory<float>&, const DeviceMemory<float>&>,
              std::tuple<const void*, const void*, const void*, const void*>>);

// We accept pointers to DeviceMemoryBase and extract opaque pointers from them.
static_assert(
    std::is_same_v<ArgsStorage<DeviceMemoryBase*, const DeviceMemoryBase*>,
                   std::tuple<const void*, const void*>>);

static StreamExecutor* NewStreamExecutor() {
  Platform* platform = PlatformManager::PlatformWithName("Host").value();
  return platform->ExecutorForDevice(/*ordinal=*/0).value();
}

TEST(KernelTest, PackDeviceMemoryArguments) {
  DeviceMemoryBase a(reinterpret_cast<void*>(0x12345678));
  DeviceMemoryBase b(reinterpret_cast<void*>(0x87654321));

  auto args = PackKernelArgs<DeviceMemoryBase>({a, b}, 0).value();
  ASSERT_EQ(args->number_of_arguments(), 2);

  auto packed = args->argument_addresses();
  const void* ptr0 = *reinterpret_cast<const void* const*>(packed[0]);
  const void* ptr1 = *reinterpret_cast<const void* const*>(packed[1]);

  ASSERT_EQ(ptr0, a.opaque());
  ASSERT_EQ(ptr1, b.opaque());
}

TEST(KernelTest, PackPodArguments) {
  auto args = std::make_unique<KernelArgsPackedArray<4>>();
  args->add_argument(1);
  args->add_argument(2.0f);
  args->add_argument(3.0);

  ASSERT_EQ(args->number_of_arguments(), 3);

  auto packed = args->argument_addresses();
  int32_t i32 = *reinterpret_cast<const int32_t*>(packed[0]);
  float f32 = *reinterpret_cast<const float*>(packed[1]);
  double f64 = *reinterpret_cast<const double*>(packed[2]);

  ASSERT_EQ(i32, 1);
  ASSERT_EQ(f32, 2.0f);
  ASSERT_EQ(f64, 3.0);
}

TEST(KernelTest, PackTupleArguments) {
  auto args = PackKernelArgs(/*shmem_bytes=*/0, 1, 2.0f, 3.0);
  ASSERT_EQ(args->number_of_arguments(), 3);

  auto packed = args->argument_addresses();
  int32_t i32 = *reinterpret_cast<const int32_t*>(packed[0]);
  float f32 = *reinterpret_cast<const float*>(packed[1]);
  double f64 = *reinterpret_cast<const double*>(packed[2]);

  ASSERT_EQ(i32, 1);
  ASSERT_EQ(f32, 2.0f);
  ASSERT_EQ(f64, 3.0);
}

TEST(KernelTest, FailToCreateTypedKernelFromEmptySpec) {
  MultiKernelLoaderSpec empty_spec(/*arity=*/0);

  auto executor = NewStreamExecutor();
  auto kernel = TypedKernelFactory<>::Create(executor, empty_spec);
  EXPECT_FALSE(kernel.ok());
}

//===----------------------------------------------------------------------===//
// Performance benchmarks below
//===----------------------------------------------------------------------===//

static void BM_PackDeviceMemoryArgs(benchmark::State& state) {
  std::vector<DeviceMemoryBase> args(state.range(0));
  for (int i = 0; i < state.range(0); ++i) {
    args[i] = DeviceMemoryBase(reinterpret_cast<void*>(0x12345678), 42);
  }

  for (auto s : state) {
    auto packed = PackKernelArgs<DeviceMemoryBase>(args, 0);
    benchmark::DoNotOptimize(packed);
  }
}

BENCHMARK(BM_PackDeviceMemoryArgs)
    ->Arg(4)
    ->Arg(8)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024);

}  // namespace stream_executor
