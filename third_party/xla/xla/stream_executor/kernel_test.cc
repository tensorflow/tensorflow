/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include "xla/stream_executor/device_memory.h"
#include "tsl/platform/test_benchmark.h"

namespace stream_executor {

// TODO(ezhulenev): Add tests for packing custom arguments.

//===----------------------------------------------------------------------===//
// Performance benchmarks below
//===----------------------------------------------------------------------===//

static void BM_PackDeviceMemoryArgs(benchmark::State& state) {
  std::vector<DeviceMemoryBase> args(state.range(0));
  for (int i = 0; i < state.range(0); ++i) {
    args[i] = DeviceMemoryBase(reinterpret_cast<void*>(0x12345678), 42);
  }

  for (auto s : state) {
    auto packed = PackKernelArgs(args, 0);
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
