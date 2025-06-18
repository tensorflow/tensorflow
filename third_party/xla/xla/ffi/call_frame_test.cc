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
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "xla/ffi/api/c_api.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/xla_data.pb.h"

namespace xla::ffi {

TEST(CallFrameTest, UpdateCallFrame) {
  se::DeviceMemoryBase mem0(reinterpret_cast<void*>(0x12345678), 1024);
  se::DeviceMemoryBase mem1(reinterpret_cast<void*>(0x87654321), 1024);

  std::vector<int64_t> dims = {1, 2, 3, 4};

  CallFrameBuilder::AttributesBuilder attrs_builder;
  attrs_builder.Insert("attr1", "value1");
  attrs_builder.Insert("attr2", "value2");

  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/1);
  builder.AddBufferArg(mem0, PrimitiveType::F32, dims);
  builder.AddBufferRet(mem1, PrimitiveType::F32, dims);
  builder.AddAttributes(attrs_builder.Build());

  // Keep call frame wrapped in optional to be able to destroy it and test that
  // updated call frame does not reference any destroyed memory.
  std::optional<CallFrame> call_frame = builder.Build();

  {  // Construct XLA_FFI_CallFrame from the original call frame.
    XLA_FFI_CallFrame ffi_call_frame = call_frame->Build(
        /*api=*/nullptr, /*ctx=*/nullptr, XLA_FFI_ExecutionStage_EXECUTE);

    EXPECT_EQ(ffi_call_frame.args.size, 1);
    EXPECT_EQ(ffi_call_frame.args.types[0], XLA_FFI_ArgType_BUFFER);
    EXPECT_EQ(static_cast<XLA_FFI_Buffer*>(ffi_call_frame.args.args[0])->data,
              mem0.opaque());

    EXPECT_EQ(ffi_call_frame.rets.size, 1);
    EXPECT_EQ(ffi_call_frame.rets.types[0], XLA_FFI_ArgType_BUFFER);
    EXPECT_EQ(static_cast<XLA_FFI_Buffer*>(ffi_call_frame.rets.rets[0])->data,
              mem1.opaque());

    EXPECT_EQ(ffi_call_frame.attrs.size, 2);
  }

  CallFrame updated_call_frame =
      std::move(call_frame)->CopyWithBuffers({mem1}, {mem0}).value();

  {  // Construct XLA_FFI_CallFrame from the updated call frame.
    XLA_FFI_CallFrame ffi_call_frame = updated_call_frame.Build(
        /*api=*/nullptr, /*ctx=*/nullptr, XLA_FFI_ExecutionStage_EXECUTE);

    EXPECT_EQ(ffi_call_frame.args.size, 1);
    EXPECT_EQ(ffi_call_frame.args.types[0], XLA_FFI_ArgType_BUFFER);
    EXPECT_EQ(static_cast<XLA_FFI_Buffer*>(ffi_call_frame.args.args[0])->data,
              mem1.opaque());

    EXPECT_EQ(ffi_call_frame.rets.size, 1);
    EXPECT_EQ(ffi_call_frame.rets.types[0], XLA_FFI_ArgType_BUFFER);
    EXPECT_EQ(static_cast<XLA_FFI_Buffer*>(ffi_call_frame.rets.rets[0])->data,
              mem0.opaque());

    EXPECT_EQ(ffi_call_frame.attrs.size, 2);
  }

  TF_ASSERT_OK(updated_call_frame.UpdateWithBuffers({mem0}, {mem1}));

  {  // Construct XLA_FFI_CallFrame from the call frame updated in place.
    XLA_FFI_CallFrame ffi_call_frame = updated_call_frame.Build(
        /*api=*/nullptr, /*ctx=*/nullptr, XLA_FFI_ExecutionStage_EXECUTE);

    EXPECT_EQ(ffi_call_frame.args.size, 1);
    EXPECT_EQ(ffi_call_frame.args.types[0], XLA_FFI_ArgType_BUFFER);
    EXPECT_EQ(static_cast<XLA_FFI_Buffer*>(ffi_call_frame.args.args[0])->data,
              mem0.opaque());

    EXPECT_EQ(ffi_call_frame.rets.size, 1);
    EXPECT_EQ(ffi_call_frame.rets.types[0], XLA_FFI_ArgType_BUFFER);
    EXPECT_EQ(static_cast<XLA_FFI_Buffer*>(ffi_call_frame.rets.rets[0])->data,
              mem1.opaque());

    EXPECT_EQ(ffi_call_frame.attrs.size, 2);
  }
}

//===----------------------------------------------------------------------===//
// Performance benchmarks below
//===----------------------------------------------------------------------===//

void BM_AddBufferArg(benchmark::State& state) {
  size_t num_args = state.range(0);

  se::DeviceMemoryBase memory(reinterpret_cast<void*>(0x12345678), 1024);
  std::vector<int64_t> dims = {1, 2, 3, 4};

  for (auto _ : state) {
    CallFrameBuilder builder(num_args, /*num_rets=*/0);
    for (size_t i = 0; i < num_args; ++i) {
      builder.AddBufferArg(memory, PrimitiveType::F32, dims);
    }

    CallFrame call_frame = builder.Build();
  }
}

void BM_AddAttributes(benchmark::State& state) {
  size_t num_attrs = state.range(0);

  CallFrameBuilder::AttributesMap attrs;
  for (size_t i = 0; i < num_attrs; ++i) {
    attrs.try_emplace(absl::StrCat("attr_", i), 42);
  }

  for (auto _ : state) {
    CallFrameBuilder::AttributesBuilder attrs_builder;
    attrs_builder.Append(attrs);

    CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
    builder.AddAttributes(attrs_builder.Build());

    CallFrame call_frame = builder.Build();
  }
}

void BM_UpdateCallFrame(benchmark::State& state) {
  size_t num_args = state.range(0);

  se::DeviceMemoryBase memory(reinterpret_cast<void*>(0x12345678), 1024);
  std::vector<int64_t> dims = {1, 2, 3, 4};

  CallFrameBuilder builder(num_args, /*num_rets=*/0);
  for (size_t i = 0; i < num_args; ++i) {
    builder.AddBufferArg(se::DeviceMemoryBase(nullptr, 1024),
                         PrimitiveType::F32, dims);
  }
  CallFrame call_frame = builder.Build();

  std::vector<se::DeviceMemoryBase> updated_args(num_args, memory);

  for (auto _ : state) {
    auto updated_call_frame =
        call_frame.CopyWithBuffers(updated_args, /*rets=*/{});
    benchmark::DoNotOptimize(updated_call_frame);
  }
}

void BM_UpdateCallFrameInPlace(benchmark::State& state) {
  size_t num_args = state.range(0);

  se::DeviceMemoryBase memory(reinterpret_cast<void*>(0x12345678), 1024);
  std::vector<int64_t> dims = {1, 2, 3, 4};

  CallFrameBuilder builder(num_args, /*num_rets=*/0);
  for (size_t i = 0; i < num_args; ++i) {
    builder.AddBufferArg(se::DeviceMemoryBase(nullptr, 1024),
                         PrimitiveType::F32, dims);
  }
  CallFrame call_frame = builder.Build();

  std::vector<se::DeviceMemoryBase> updated_args(num_args, memory);

  for (auto _ : state) {
    benchmark::DoNotOptimize(
        call_frame.UpdateWithBuffers(updated_args, /*rets=*/{}));
  }
}

#define BENCHMARK_SIZES(name) \
  BENCHMARK(name)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)->Arg(64)

BENCHMARK_SIZES(BM_AddBufferArg);
BENCHMARK_SIZES(BM_AddAttributes);
BENCHMARK_SIZES(BM_UpdateCallFrame);
BENCHMARK_SIZES(BM_UpdateCallFrameInPlace);

}  // namespace xla::ffi
