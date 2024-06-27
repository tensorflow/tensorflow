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

#include "xla/ffi/call_frame.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/strings/str_cat.h"
#include "xla/stream_executor/device_memory.h"
#include "tsl/platform/test_benchmark.h"

namespace xla::ffi {

void BM_AddBufferArg(benchmark::State& state) {
  size_t num_args = state.range();

  se::DeviceMemoryBase memory(nullptr, 1024);
  std::vector<int64_t> dims = {1, 2, 3, 4};

  for (auto _ : state) {
    CallFrameBuilder builder(num_args, /*num_rets=*/0);
    for (size_t i = 0; i < num_args; ++i) {
      builder.AddBufferArg(se::DeviceMemoryBase(nullptr, 1024),
                           PrimitiveType::F32, dims);
    }

    CallFrame call_frame = builder.Build();
  }
}

void BM_AddAttributes(benchmark::State& state) {
  size_t num_attrs = state.range();

  CallFrameBuilder::FlatAttributesMap flat_attrs;
  for (size_t i = 0; i < num_attrs; ++i) {
    flat_attrs.try_emplace(absl::StrCat("attr_", i), 42);
  }

  for (auto _ : state) {
    CallFrameBuilder::AttributesBuilder attrs_builder;
    attrs_builder.Append(flat_attrs);

    CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
    builder.AddAttributes(attrs_builder.Build());

    CallFrame call_frame = builder.Build();
  }
}

BENCHMARK(BM_AddBufferArg)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16);
BENCHMARK(BM_AddAttributes)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16);

}  // namespace xla::ffi
