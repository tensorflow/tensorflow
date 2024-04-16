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

#include "xla/ffi/api/ffi.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/ffi_api.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/test.h"
#include "tsl/platform/test_benchmark.h"

namespace xla::ffi {

namespace {

using ::testing::HasSubstr;
using ::tsl::testing::StatusIs;

}  // namespace

TEST(FfiTest, DataTypeEnumValue) {
  // Verify that xla::PrimitiveType and xla::ffi::DataType use the same
  // integer value for encoding data types.
  auto encoded = [](auto value) { return static_cast<uint8_t>(value); };

  EXPECT_EQ(encoded(PrimitiveType::PRED), encoded(DataType::PRED));

  EXPECT_EQ(encoded(PrimitiveType::S8), encoded(DataType::S8));
  EXPECT_EQ(encoded(PrimitiveType::S16), encoded(DataType::S16));
  EXPECT_EQ(encoded(PrimitiveType::S32), encoded(DataType::S32));
  EXPECT_EQ(encoded(PrimitiveType::S64), encoded(DataType::S64));

  EXPECT_EQ(encoded(PrimitiveType::U8), encoded(DataType::U8));
  EXPECT_EQ(encoded(PrimitiveType::U16), encoded(DataType::U16));
  EXPECT_EQ(encoded(PrimitiveType::U32), encoded(DataType::U32));
  EXPECT_EQ(encoded(PrimitiveType::U64), encoded(DataType::U64));

  EXPECT_EQ(encoded(PrimitiveType::F16), encoded(DataType::F16));
  EXPECT_EQ(encoded(PrimitiveType::F32), encoded(DataType::F32));
  EXPECT_EQ(encoded(PrimitiveType::F64), encoded(DataType::F64));

  EXPECT_EQ(encoded(PrimitiveType::BF16), encoded(DataType::BF16));
}

TEST(FfiTest, BufferBaseArgument) {
  std::vector<float> storage(4, 0.0f);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(float));

  CallFrameBuilder builder;
  builder.AddBufferArg(memory, PrimitiveType::F32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Arg<BufferBase>().To([&](auto buffer) {
    EXPECT_EQ(buffer.data, storage.data());
    EXPECT_EQ(buffer.dimensions.size(), 2);
    return Error::Success();
  });
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, BufferArgument) {
  std::vector<float> storage(4, 0.0f);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(float));

  CallFrameBuilder builder;
  builder.AddBufferArg(memory, PrimitiveType::F32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto handler =
      Ffi::Bind().Arg<BufferR2<DataType::F32>>().To([&](auto buffer) {
        EXPECT_EQ(buffer.data, storage.data());
        EXPECT_EQ(buffer.dimensions.size(), 2);
        return Error::Success();
      });
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, BufferBaseResult) {
  std::vector<float> storage(4, 0.0f);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(float));

  CallFrameBuilder builder;
  builder.AddBufferRet(memory, PrimitiveType::F32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto handler =
      Ffi::Bind().Ret<BufferBase>().To([&](Result<BufferBase> buffer) {
        EXPECT_EQ(buffer->data, storage.data());
        EXPECT_EQ(buffer->dimensions.size(), 2);
        return Error::Success();
      });
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, MissingBufferArgument) {
  CallFrameBuilder builder;
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Arg<BufferR1<DataType::F32>>().To(
      [](auto) { return Error::Success(); });
  auto status = Call(*handler, call_frame);

  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("Wrong number of arguments")));
}

TEST(FfiTest, WrongRankBufferArgument) {
  std::vector<int32_t> storage(4, 0.0);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(int32_t));

  CallFrameBuilder builder;
  builder.AddBufferArg(memory, PrimitiveType::F32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Arg<BufferR1<DataType::F32>>().To(
      [](auto) { return Error::Success(); });
  auto status = Call(*handler, call_frame);

  EXPECT_THAT(status,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Wrong buffer rank: expected 1 but got 2")));
}

TEST(FfiTest, WrongTypeBufferArgument) {
  std::vector<int32_t> storage(4, 0.0);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(int32_t));

  CallFrameBuilder builder;
  builder.AddBufferArg(memory, PrimitiveType::S32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Arg<BufferR2<DataType::F32>>().To(
      [](auto) { return Error::Success(); });
  auto status = Call(*handler, call_frame);

  EXPECT_THAT(
      status,
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Wrong buffer dtype: expected F32 but got S32")));
}

TEST(FfiTest, AutoBinding) {
  static constexpr char kI32[] = "i32";

  auto handler = Ffi::BindTo(+[](BufferBase buffer, Attr<int32_t, kI32> foo) {
    EXPECT_EQ(*foo, 42);
    return Error::Success();
  });

  std::vector<float> storage(4, 0.0f);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(float));

  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert(kI32, 42);

  CallFrameBuilder builder;
  builder.AddBufferArg(memory, PrimitiveType::F32, /*dims=*/{2, 2});
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto status = Call(*handler, call_frame);
  TF_ASSERT_OK(status);
}

TEST(FfiTest, AutoBindingResult) {
  auto handler =
      Ffi::BindTo(+[](Result<BufferBase> buffer) { return Error::Success(); });

  CallFrameBuilder builder;
  builder.AddBufferRet(se::DeviceMemoryBase(), PrimitiveType::F32, /*dims=*/{});
  auto call_frame = builder.Build();

  auto status = Call(*handler, call_frame);
  TF_ASSERT_OK(status);
}

struct I32AndF32 {
  int32_t i32;
  float f32;
};

XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(I32AndF32, StructMember<int32_t>("i32"),
                                      StructMember<float>("f32"));

TEST(FfiTest, AutoBindingStructs) {
  auto handler = Ffi::BindTo(+[](I32AndF32 attrs) {
    EXPECT_EQ(attrs.i32, 42);
    EXPECT_EQ(attrs.f32, 42.0f);
    return Error::Success();
  });

  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("i32", 42);
  attrs.Insert("f32", 42.0f);

  CallFrameBuilder builder;
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto status = Call(*handler, call_frame);
  TF_ASSERT_OK(status);
}

TEST(FfiTest, AutoBindingDictionary) {
  auto handler = Ffi::BindTo(+[](Dictionary attrs) {
    EXPECT_EQ(*attrs.get<int32_t>("i32"), 42);
    EXPECT_EQ(*attrs.get<float>("f32"), 42.0f);
    return Error::Success();
  });

  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("i32", 42);
  attrs.Insert("f32", 42.0f);

  CallFrameBuilder builder;
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto status = Call(*handler, call_frame);
  TF_ASSERT_OK(status);
}

// Use opaque struct to define a platform stream type just like platform
// stream for GPU backend (e.g. `CUstream_st`  and `cudaStream_t`).
struct TestStreamSt;
using TestStream = TestStreamSt*;

template <>
struct CtxBinding<TestStream> {
  using Ctx = PlatformStream<TestStream>;
};

TEST(FfiTest, BindingPlatformStreamInference) {
  // We only check that it compiles.
  (void)Ffi::BindTo(+[](TestStream stream) { return Error::Success(); });
}

TEST(FfiTest, ArrayAttr) {
  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("arr", std::vector<int32_t>({1, 2, 3, 4}));

  CallFrameBuilder builder;
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [&](Span<const int32_t> arr) {
    EXPECT_EQ(arr.size(), 4);
    EXPECT_EQ(arr[0], 1);
    EXPECT_EQ(arr[1], 2);
    EXPECT_EQ(arr[2], 3);
    EXPECT_EQ(arr[3], 4);
    return Error::Success();
  };

  auto handler = Ffi::Bind().Attr<Span<const int32_t>>("arr").To(fn);
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, PointerAttr) {
  std::string foo = "foo";

  // Test for convenience attr binding that casts i64 attribute to user-type
  // pointers. It's up to the user to guarantee that pointer is valid.
  auto ptr = reinterpret_cast<uintptr_t>(&foo);
  static_assert(sizeof(ptr) == sizeof(int64_t));

  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("ptr", static_cast<int64_t>(ptr));

  CallFrameBuilder builder;
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [&](const std::string* str) {
    EXPECT_EQ(*str, "foo");
    return Error::Success();
  };

  auto handler = Ffi::Bind().Attr<Pointer<std::string>>("ptr").To(fn);
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

//===----------------------------------------------------------------------===//
// Performance benchmarks are below.
//===----------------------------------------------------------------------===//

static CallFrameBuilder WithBufferArgs(size_t num_args, size_t rank = 4) {
  se::DeviceMemoryBase memory;
  std::vector<int64_t> dims(4, 1);

  CallFrameBuilder builder;
  for (size_t i = 0; i < num_args; ++i) {
    builder.AddBufferArg(memory, PrimitiveType::F32, dims);
  }
  return builder;
}

//===----------------------------------------------------------------------===//
// BM_BufferBaseArgX1
//===----------------------------------------------------------------------===//

void BM_BufferBaseArgX1(benchmark::State& state) {
  auto call_frame = WithBufferArgs(1).Build();

  auto handler = Ffi::Bind().Arg<BufferBase>().To([](auto buffer) {
    benchmark::DoNotOptimize(buffer);
    return Error::Success();
  });
  for (auto _ : state) {
    CHECK_OK(Call(*handler, call_frame));
  }
}

BENCHMARK(BM_BufferBaseArgX1);

//===----------------------------------------------------------------------===//
// BM_BufferBaseArgX4
//===----------------------------------------------------------------------===//

void BM_BufferBaseArgX4(benchmark::State& state) {
  auto call_frame = WithBufferArgs(4).Build();

  auto handler = Ffi::Bind()
                     .Arg<BufferBase>()
                     .Arg<BufferBase>()
                     .Arg<BufferBase>()
                     .Arg<BufferBase>()
                     .To([](auto b0, auto b1, auto b2, auto b3) {
                       benchmark::DoNotOptimize(b0);
                       benchmark::DoNotOptimize(b1);
                       benchmark::DoNotOptimize(b2);
                       benchmark::DoNotOptimize(b3);
                       return Error::Success();
                     });

  for (auto _ : state) {
    CHECK_OK(Call(*handler, call_frame));
  }
}

BENCHMARK(BM_BufferBaseArgX4);

//===----------------------------------------------------------------------===//
// BM_BufferArgX1
//===----------------------------------------------------------------------===//

void BM_BufferArgX1(benchmark::State& state) {
  auto call_frame = WithBufferArgs(1).Build();

  auto handler = Ffi::Bind().Arg<BufferR4<DataType::F32>>().To([](auto buffer) {
    benchmark::DoNotOptimize(buffer);
    return Error::Success();
  });
  for (auto _ : state) {
    CHECK_OK(Call(*handler, call_frame));
  }
}

BENCHMARK(BM_BufferArgX1);

//===----------------------------------------------------------------------===//
// BM_BufferArgX4
//===----------------------------------------------------------------------===//

void BM_BufferArgX4(benchmark::State& state) {
  auto call_frame = WithBufferArgs(4).Build();

  auto handler = Ffi::Bind()
                     .Arg<BufferR4<DataType::F32>>()
                     .Arg<BufferR4<DataType::F32>>()
                     .Arg<BufferR4<DataType::F32>>()
                     .Arg<BufferR4<DataType::F32>>()
                     .To([](auto b0, auto b1, auto b2, auto b3) {
                       benchmark::DoNotOptimize(b0);
                       benchmark::DoNotOptimize(b1);
                       benchmark::DoNotOptimize(b2);
                       benchmark::DoNotOptimize(b3);
                       return Error::Success();
                     });

  for (auto _ : state) {
    CHECK_OK(Call(*handler, call_frame));
  }
}

BENCHMARK(BM_BufferArgX4);

//===----------------------------------------------------------------------===//
// BM_TupleOfI32Attrs
//===----------------------------------------------------------------------===//

struct TupleOfI32 {
  int64_t i32_0;
  int64_t i32_1;
  int64_t i32_2;
  int64_t i32_3;
};

XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(TupleOfI32,
                                      StructMember<int32_t>("i32_0"),
                                      StructMember<int32_t>("i32_1"),
                                      StructMember<int32_t>("i32_2"),
                                      StructMember<int32_t>("i32_3"));

void BM_TupleOfI32Attrs(benchmark::State& state) {
  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("i32_0", 1);
  attrs.Insert("i32_1", 2);
  attrs.Insert("i32_2", 3);
  attrs.Insert("i32_3", 4);

  CallFrameBuilder builder;
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Attrs<TupleOfI32>().To([](auto tuple) {
    benchmark::DoNotOptimize(tuple);
    return Error::Success();
  });

  for (auto _ : state) {
    CHECK_OK(Call(*handler, call_frame));
  }
}

BENCHMARK(BM_TupleOfI32Attrs);

}  // namespace xla::ffi
