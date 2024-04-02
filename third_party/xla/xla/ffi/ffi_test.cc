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

#include "xla/ffi/ffi.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/types/span.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/ffi_api.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/test.h"

namespace xla::ffi {

using ::testing::_;
using ::testing::HasSubstr;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;
using ::tsl::testing::StatusIs;

TEST(FfiTest, StaticRegistration) {
  static constexpr auto* noop = +[] { return absl::OkStatus(); };

  // Use explicit binding specification.
  XLA_FFI_DEFINE_HANDLER(NoOp0, noop, Ffi::Bind());

  // Automatically infer binding specification from function signature.
  XLA_FFI_DEFINE_HANDLER(NoOp1, noop);

  XLA_FFI_REGISTER_HANDLER(GetXlaFfiApi(), "no-op-0", "Host", NoOp0);
  XLA_FFI_REGISTER_HANDLER(GetXlaFfiApi(), "no-op-1", "Host", NoOp1,
                           XLA_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE);

  auto handler0 = FindHandler("no-op-0", "Host");
  auto handler1 = FindHandler("no-op-1", "Host");

  TF_ASSERT_OK(handler0.status());
  TF_ASSERT_OK(handler1.status());

  ASSERT_EQ(handler0->traits, 0);
  ASSERT_EQ(handler1->traits, XLA_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE);

  EXPECT_THAT(StaticRegisteredHandlers("Host"),
              UnorderedElementsAre(Pair("no-op-0", _), Pair("no-op-1", _)));
}

TEST(FfiTest, ForwardError) {
  auto call_frame = CallFrameBuilder().Build();
  auto handler = Ffi::Bind().To([] { return absl::AbortedError("Ooops!"); });
  auto status = Call(*handler, call_frame);
  ASSERT_EQ(status.message(), "Ooops!");
}

TEST(FfiTest, WrongNumArgs) {
  CallFrameBuilder builder;
  builder.AddBufferArg(se::DeviceMemoryBase(nullptr), PrimitiveType::F32, {});
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Arg<BufferBase>().Arg<BufferBase>().To(
      [](BufferBase, BufferBase) { return absl::OkStatus(); });

  auto status = Call(*handler, call_frame);

  ASSERT_EQ(status.message(),
            "Wrong number of arguments: expected 2 but got 1");
}

TEST(FfiTest, WrongNumAttrs) {
  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("i32", 42);
  attrs.Insert("f32", 42.0f);

  CallFrameBuilder builder;
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Attr<int32_t>("i32").To(
      [](int32_t) { return absl::OkStatus(); });

  auto status = Call(*handler, call_frame);

  ASSERT_EQ(status.message(),
            "Wrong number of attributes: expected 1 but got 2");
}

TEST(FfiTest, BuiltinAttributes) {
  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("i32", 42);
  attrs.Insert("f32", 42.0f);
  attrs.Insert("str", "foo");

  CallFrameBuilder builder;
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [&](int32_t i32, float f32, std::string_view str) {
    EXPECT_EQ(i32, 42);
    EXPECT_EQ(f32, 42.0f);
    EXPECT_EQ(str, "foo");
    return absl::OkStatus();
  };

  auto handler = Ffi::Bind()
                     .Attr<int32_t>("i32")
                     .Attr<float>("f32")
                     .Attr<std::string_view>("str")
                     .To(fn);

  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, BuiltinAttributesAutoBinding) {
  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("i32", 42);
  attrs.Insert("f32", 42.0f);
  attrs.Insert("str", "foo");

  CallFrameBuilder builder;
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  static constexpr char kI32[] = "i32";
  static constexpr char kF32[] = "f32";
  static constexpr char kStr[] = "str";

  auto fn = [&](Attr<int32_t, kI32> i32, Attr<float, kF32> f32,
                Attr<std::string_view, kStr> str) {
    EXPECT_EQ(*i32, 42);
    EXPECT_EQ(*f32, 42.0f);
    EXPECT_EQ(*str, "foo");
    return absl::OkStatus();
  };

  auto handler = Ffi::BindTo(fn);
  auto status = Call(*handler, call_frame);
  TF_ASSERT_OK(status);
}

TEST(FfiTest, ArrayAttr) {
  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("arr", std::vector<int32_t>({1, 2, 3, 4}));

  CallFrameBuilder builder;
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [&](absl::Span<const int32_t> arr) {
    EXPECT_EQ(arr.size(), 4);
    EXPECT_EQ(arr[0], 1);
    EXPECT_EQ(arr[1], 2);
    EXPECT_EQ(arr[2], 3);
    EXPECT_EQ(arr[3], 4);
    return absl::OkStatus();
  };

  auto handler = Ffi::Bind().Attr<absl::Span<const int32_t>>("arr").To(fn);
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
    return absl::OkStatus();
  };

  auto handler = Ffi::Bind().Attr<Pointer<std::string>>("ptr").To(fn);
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, AttrsAsDictionary) {
  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("i32", 42);
  attrs.Insert("f32", 42.0f);
  attrs.Insert("str", "foo");

  CallFrameBuilder builder;
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [&](Dictionary dict) {
    EXPECT_EQ(dict.size(), 3);

    EXPECT_TRUE(dict.contains("i32"));
    EXPECT_TRUE(dict.contains("f32"));
    EXPECT_TRUE(dict.contains("str"));

    auto i32 = dict.get<int32_t>("i32");
    auto f32 = dict.get<float>("f32");
    auto str = dict.get<std::string_view>("str");

    EXPECT_TRUE(i32.has_value());
    EXPECT_TRUE(f32.has_value());
    EXPECT_TRUE(str.has_value());

    if (i32) EXPECT_EQ(*i32, 42);
    if (f32) EXPECT_EQ(*f32, 42.0f);
    if (str) EXPECT_EQ(*str, "foo");

    EXPECT_FALSE(dict.contains("i64"));
    EXPECT_FALSE(dict.get<int64_t>("i32").has_value());
    EXPECT_FALSE(dict.get<int64_t>("i64").has_value());

    return absl::OkStatus();
  };

  auto handler = Ffi::Bind().Attrs().To(fn);
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, DictionaryAttr) {
  CallFrameBuilder::FlatAttributesMap dict0;
  dict0.try_emplace("i32", 42);

  CallFrameBuilder::FlatAttributesMap dict1;
  dict1.try_emplace("f32", 42.0f);

  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("dict0", dict0);
  attrs.Insert("dict1", dict1);

  CallFrameBuilder builder;
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [&](Dictionary dict0, Dictionary dict1) {
    EXPECT_EQ(dict0.size(), 1);
    EXPECT_EQ(dict1.size(), 1);

    EXPECT_TRUE(dict0.contains("i32"));
    EXPECT_TRUE(dict1.contains("f32"));

    auto i32 = dict0.get<int32_t>("i32");
    auto f32 = dict1.get<float>("f32");

    EXPECT_TRUE(i32.has_value());
    EXPECT_TRUE(f32.has_value());

    if (i32) EXPECT_EQ(*i32, 42);
    if (f32) EXPECT_EQ(*f32, 42.0f);

    return absl::OkStatus();
  };

  auto handler =
      Ffi::Bind().Attr<Dictionary>("dict0").Attr<Dictionary>("dict1").To(fn);

  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

struct PairOfI32AndF32 {
  int32_t i32;
  float f32;
};

XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(PairOfI32AndF32,
                                      StructMember<int32_t>("i32"),
                                      StructMember<float>("f32"));

TEST(FfiTest, StructAttr) {
  CallFrameBuilder::FlatAttributesMap dict;
  dict.try_emplace("i32", 42);
  dict.try_emplace("f32", 42.0f);

  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("str", "foo");
  attrs.Insert("i32_and_f32", dict);

  CallFrameBuilder builder;
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [&](std::string_view str, PairOfI32AndF32 i32_and_f32) {
    EXPECT_EQ(str, "foo");
    EXPECT_EQ(i32_and_f32.i32, 42);
    EXPECT_EQ(i32_and_f32.f32, 42.0f);
    return absl::OkStatus();
  };

  auto handler = Ffi::Bind()
                     .Attr<std::string_view>("str")
                     .Attr<PairOfI32AndF32>("i32_and_f32")
                     .To(fn);

  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, AttrsAsStruct) {
  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("i32", 42);
  attrs.Insert("f32", 42.0f);

  CallFrameBuilder builder;
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [&](PairOfI32AndF32 i32_and_f32) {
    EXPECT_EQ(i32_and_f32.i32, 42);
    EXPECT_EQ(i32_and_f32.f32, 42.0f);
    return absl::OkStatus();
  };

  auto handler = Ffi::Bind().Attrs<PairOfI32AndF32>().To(fn);
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, DecodingErrors) {
  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("i32", 42);
  attrs.Insert("i64", 42);
  attrs.Insert("f32", 42.0f);
  attrs.Insert("str", "foo");

  CallFrameBuilder builder;
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [](int32_t, int64_t, float, std::string_view) {
    return absl::OkStatus();
  };

  auto handler = Ffi::Bind()
                     .Attr<int32_t>("not_i32_should_fail")
                     .Attr<int64_t>("not_i64_should_fail")
                     .Attr<float>("f32")
                     .Attr<std::string_view>("not_str_should_fail")
                     .To(fn);

  auto status = Call(*handler, call_frame);

  EXPECT_TRUE(absl::StrContains(
      status.message(),
      "Failed to decode all FFI handler operands (bad operands at: 0, 1, 3)"));

  EXPECT_TRUE(absl::StrContains(
      status.message(), "Attribute name mismatch: i32 vs not_i32_should_fail"));

  EXPECT_TRUE(absl::StrContains(
      status.message(), "Attribute name mismatch: i64 vs not_i64_should_fail"));

  EXPECT_TRUE(absl::StrContains(
      status.message(), "Attribute name mismatch: str vs not_str_should_fail"));
}

TEST(FfiTest, BufferBaseArgument) {
  std::vector<float> storage(4, 0.0f);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(float));

  CallFrameBuilder builder;
  builder.AddBufferArg(memory, PrimitiveType::F32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto fn = [&](BufferBase buffer) {
    EXPECT_EQ(buffer.dtype, PrimitiveType::F32);
    EXPECT_EQ(buffer.data.opaque(), storage.data());
    EXPECT_EQ(buffer.dimensions.size(), 2);
    return absl::OkStatus();
  };

  {  // Test explicit binding signature declaration.
    auto handler = Ffi::Bind().Arg<BufferBase>().To(fn);
    auto status = Call(*handler, call_frame);
    TF_ASSERT_OK(status);
  }

  {  // Test inferring binding signature from a handler type.
    auto handler = Ffi::BindTo(fn);
    auto status = Call(*handler, call_frame);
    TF_ASSERT_OK(status);
  }
}

TEST(FfiTest, TypedAndRankedBufferArgument) {
  std::vector<float> storage(4, 0.0f);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(float));

  CallFrameBuilder builder;
  builder.AddBufferArg(memory, PrimitiveType::F32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto fn = [&](BufferR2<PrimitiveType::F32> buffer) {
    EXPECT_EQ(buffer.data.opaque(), storage.data());
    EXPECT_EQ(buffer.dimensions.size(), 2);
    return absl::OkStatus();
  };

  {  // Test explicit binding signature declaration.
    auto handler = Ffi::Bind().Arg<BufferR2<PrimitiveType::F32>>().To(fn);
    auto status = Call(*handler, call_frame);
    TF_ASSERT_OK(status);
  }

  {  // Test inferring binding signature from a handler type.
    auto handler = Ffi::BindTo(fn);
    auto status = Call(*handler, call_frame);
    TF_ASSERT_OK(status);
  }
}

TEST(FfiTest, WrongRankBufferArgument) {
  std::vector<int32_t> storage(4, 0.0);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(int32_t));

  CallFrameBuilder builder;
  builder.AddBufferArg(memory, PrimitiveType::F32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Arg<BufferR1<PrimitiveType::F32>>().To(
      [](auto) { return absl::OkStatus(); });
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

  auto handler = Ffi::Bind().Arg<BufferR2<PrimitiveType::F32>>().To(
      [](auto) { return absl::OkStatus(); });
  auto status = Call(*handler, call_frame);

  EXPECT_THAT(
      status,
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Wrong buffer dtype: expected f32 but got s32")));
}

TEST(FfiTest, RemainingArgs) {
  std::vector<float> storage(4, 0.0f);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(float));

  CallFrameBuilder builder;
  builder.AddBufferArg(memory, PrimitiveType::F32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto fn = [&](RemainingArgs args) {
    EXPECT_EQ(args.size(), 1);
    EXPECT_TRUE(args.get<BufferBase>(0).has_value());
    EXPECT_FALSE(args.get<BufferBase>(1).has_value());
    return absl::OkStatus();
  };

  auto handler = Ffi::Bind().RemainingArgs().To(fn);
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, RunOptionsCtx) {
  auto call_frame = CallFrameBuilder().Build();
  auto* expected = reinterpret_cast<se::Stream*>(0x01234567);

  ServiceExecutableRunOptions opts;
  opts.mutable_run_options()->set_stream(expected);

  auto fn = [&](const se::Stream* run_options) {
    EXPECT_EQ(run_options, expected);
    return absl::OkStatus();
  };

  auto handler = Ffi::Bind().Ctx<Stream>().To(fn);
  auto status = Call(*handler, call_frame, {&opts});

  TF_ASSERT_OK(status);
}

}  // namespace xla::ffi
