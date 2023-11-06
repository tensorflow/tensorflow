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

#include "xla/ffi/ffi.h"

#include <cstdint>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "xla/ffi/call_frame.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/test.h"

namespace xla::ffi {

TEST(FfiTest, StaticRegistration) {
  static constexpr auto* noop = +[] { return absl::OkStatus(); };

  XLA_FFI_DEFINE_HANDLER(NoOp, noop, Ffi::Bind());
  XLA_FFI_REGISTER_HANDLER(GetXlaFfiApi(), "no-op", NoOp);

  auto handler = FindHandler("no-op");
  TF_ASSERT_OK(handler.status());
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

  auto handler = Ffi::Bind().Arg<Buffer>().Arg<Buffer>().To(
      [](Buffer, Buffer) { return absl::OkStatus(); });

  auto status = Call(*handler, call_frame);

  ASSERT_EQ(status.message(),
            "Wrong number of arguments: expected 2 but got 1");
}

TEST(FfiTest, WrongNumAttrs) {
  CallFrameBuilder builder;
  builder.AddI32Attr("i32", 42);
  builder.AddF32Attr("f32", 42.0f);
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Attr<int32_t>("i32").To(
      [](int32_t) { return absl::OkStatus(); });

  auto status = Call(*handler, call_frame);

  ASSERT_EQ(status.message(),
            "Wrong number of attributes: expected 1 but got 2");
}

TEST(FfiTest, BuiltinAttributes) {
  CallFrameBuilder builder;
  builder.AddI32Attr("i32", 42);
  builder.AddF32Attr("f32", 42.0f);
  builder.AddStringAttr("str", "foo");
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

TEST(FfiTest, DecodingErrors) {
  CallFrameBuilder builder;
  builder.AddI32Attr("i32", 42);
  builder.AddF32Attr("f32", 42.0f);
  builder.AddStringAttr("str", "foo");
  auto call_frame = builder.Build();

  auto fn = [](int32_t, float, std::string_view) { return absl::OkStatus(); };

  auto handler = Ffi::Bind()
                     .Attr<int32_t>("not_i32_should_fail")
                     .Attr<float>("f32")
                     .Attr<std::string_view>("not_str_should_fail")
                     .To(fn);

  auto status = Call(*handler, call_frame);

  ASSERT_EQ(
      status.message(),
      "Failed to decode all FFI handler operands (bad operands at: 0, 2)");
}

TEST(FfiTest, BufferArgument) {
  std::vector<float> storage(4, 0.0f);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(float));

  CallFrameBuilder builder;
  builder.AddBufferArg(memory, PrimitiveType::F32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto fn = [&](Buffer buffer) {
    EXPECT_EQ(buffer.data.opaque(), storage.data());
    EXPECT_EQ(buffer.primitive_type, PrimitiveType::F32);
    EXPECT_EQ(buffer.dimensions.size(), 2);
    return absl::OkStatus();
  };

  auto handler = Ffi::Bind().Arg<Buffer>().To(fn);
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, RunOptionsCtx) {
  auto call_frame = CallFrameBuilder().Build();
  auto* expected = reinterpret_cast<ServiceExecutableRunOptions*>(0x01234567);

  auto fn = [&](const ServiceExecutableRunOptions* run_options) {
    EXPECT_EQ(run_options, expected);
    return absl::OkStatus();
  };

  auto handler = Ffi::Bind().Ctx<ServiceExecutableRunOptions>().To(fn);
  auto status = Call(*handler, call_frame, {expected});

  TF_ASSERT_OK(status);
}

}  // namespace xla::ffi
