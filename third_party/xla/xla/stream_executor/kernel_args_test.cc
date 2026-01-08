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

#include "xla/stream_executor/kernel_args.h"

#include <cstdint>
#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "benchmark/benchmark.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel_metadata.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor {

// Struct for testing custom kernel arguments with C++ structs.
struct Data {};

// Struct for testing custom kernel argument packing.
struct CustomData {
  int32_t value;
};

template <>
struct KernelArgPacking<CustomData> {
  using Type = int32_t;
  static int32_t Pack(CustomData data) { return data.value + 1; }
};

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

// We pass DeviceAddressBase as an opaque pointer.
static_assert(
    std::is_same_v<ArgsStorage<DeviceAddressBase, const DeviceAddressBase,
                               DeviceAddressBase&, const DeviceAddressBase&>,
                   std::tuple<void*, void*, void*, void*>>);

// We pass DeviceAddress<T> as an opaque pointer.
static_assert(std::is_same_v<
              ArgsStorage<DeviceAddress<float>, const DeviceAddress<float>,
                          DeviceAddress<float>&, const DeviceAddress<float>&>,
              std::tuple<float*, float*, float*, float*>>);

// We accept pointers to DeviceAddressBase and extract opaque pointers from
// them.
static_assert(
    std::is_same_v<ArgsStorage<DeviceAddressBase*, const DeviceAddressBase*>,
                   std::tuple<void*, const void*>>);

// We use out template specialization to pack custom struct as int32_t.
static_assert(std::is_same_v<ArgsStorage<CustomData>, std::tuple<int32_t>>);

TEST(KernelTest, PackDeviceAddressArguments) {
  DeviceAddressBase a(reinterpret_cast<void*>(0x12345678));
  DeviceAddressBase b(reinterpret_cast<void*>(0x87654321));

  auto args = PackKernelArgs(std::vector<DeviceAddressBase>({a, b}), 0);
  ASSERT_EQ(args->number_of_arguments(), 2);

  auto packed = args->argument_addresses();
  const void* ptr0 = *reinterpret_cast<const void* const*>(packed[0]);
  const void* ptr1 = *reinterpret_cast<const void* const*>(packed[1]);

  ASSERT_EQ(ptr0, a.opaque());
  ASSERT_EQ(ptr1, b.opaque());
}

TEST(KernelTest, PackPodArguments) {
  auto args = std::make_unique<KernelArgsPackedArray>(4);
  args->add_argument(1);
  args->add_argument(2.0f);
  args->add_argument(3.0);
  args->add_argument(CustomData{42});

  ASSERT_EQ(args->number_of_arguments(), 4);

  auto packed = args->argument_addresses();
  int32_t i32 = *reinterpret_cast<const int32_t*>(packed[0]);
  float f32 = *reinterpret_cast<const float*>(packed[1]);
  double f64 = *reinterpret_cast<const double*>(packed[2]);
  int32_t custom = *reinterpret_cast<const int32_t*>(packed[3]);

  ASSERT_EQ(i32, 1);
  ASSERT_EQ(f32, 2.0f);
  ASSERT_EQ(f64, 3.0);
  ASSERT_EQ(custom, 43);
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

TEST(KernelTest, PackTupleWithCutomPacking) {
  auto args = PackKernelArgs(/*shmem_bytes=*/0, CustomData{42});
  ASSERT_EQ(args->number_of_arguments(), 1);

  auto packed = args->argument_addresses();
  int32_t i32 = *reinterpret_cast<const int32_t*>(packed[0]);

  ASSERT_EQ(i32, 43);
}

TEST(KernelTest, PackArgumentsWithInt64) {
  std::vector<KernelArg> args;
  DeviceAddressBase somemem(reinterpret_cast<void*>(0x12345678));
  int64_t someint64 = 1234;
  args.emplace_back(somemem);
  args.emplace_back(someint64);
  TF_ASSERT_OK_AND_ASSIGN(auto packed_args_ptr,
                          PackKernelArgs(args, KernelMetadata()));
  ASSERT_EQ(packed_args_ptr->number_of_arguments(), 2);
  ASSERT_EQ(packed_args_ptr->number_of_shared_bytes(), 0);
  const auto packed = packed_args_ptr->argument_addresses();
  ASSERT_EQ(packed.size(), 2);
  ASSERT_EQ(*reinterpret_cast<const void* const*>(packed[0]), somemem.opaque());
  ASSERT_EQ(*reinterpret_cast<const int64_t*>(packed[1]), someint64);
}

//===----------------------------------------------------------------------===//
// Performance benchmarks below
//===----------------------------------------------------------------------===//

static void BM_PackDeviceAddressArgs(benchmark::State& state) {
  std::vector<DeviceAddressBase> args(state.range(0));
  for (int i = 0; i < state.range(0); ++i) {
    args[i] = DeviceAddressBase(reinterpret_cast<void*>(0x12345678), 42);
  }

  for (auto s : state) {
    auto packed = PackKernelArgs(args, 0);
    benchmark::DoNotOptimize(packed);
  }
}

BENCHMARK(BM_PackDeviceAddressArgs)
    ->Arg(4)
    ->Arg(8)
    ->Arg(32)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024);

}  // namespace stream_executor
