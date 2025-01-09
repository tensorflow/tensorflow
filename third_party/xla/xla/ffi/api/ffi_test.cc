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
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/synchronization/blocking_counter.h"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/call_frame.h"
#include "xla/ffi/execution_context.h"
#include "xla/ffi/execution_state.h"
#include "xla/ffi/ffi_api.h"
#include "xla/ffi/type_id_registry.h"
#include "xla/primitive_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/chain.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/test.h"
#include "tsl/platform/test_benchmark.h"
#include "tsl/platform/threadpool.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::ffi {

enum class Int32BasedEnum : int32_t {
  kOne = 1,
  kTwo = 2,
};

static constexpr int64_t kI32MaxValue = std::numeric_limits<int32_t>::max();

enum class Int64BasedEnum : int64_t {
  kOne = kI32MaxValue + 1,
  kTwo = kI32MaxValue + 2,
};

}  // namespace xla::ffi

XLA_FFI_REGISTER_ENUM_ATTR_DECODING(::xla::ffi::Int32BasedEnum);
XLA_FFI_REGISTER_ENUM_ATTR_DECODING(::xla::ffi::Int64BasedEnum);

namespace xla::ffi {

struct PairOfI32AndF32 {
  int32_t i32;
  float f32;
};

struct TupleOfI32 {
  int32_t i32_0;
  int32_t i32_1;
  int32_t i32_2;
  int32_t i32_3;
};

}  // namespace xla::ffi

XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(::xla::ffi::PairOfI32AndF32,
                                      ::xla::ffi::StructMember<int32_t>("i32"),
                                      ::xla::ffi::StructMember<float>("f32"));
XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(
    ::xla::ffi::TupleOfI32, ::xla::ffi::StructMember<int32_t>("i32_0"),
    ::xla::ffi::StructMember<int32_t>("i32_1"),
    ::xla::ffi::StructMember<int32_t>("i32_2"),
    ::xla::ffi::StructMember<int32_t>("i32_3"));

namespace xla::ffi {

using ::testing::HasSubstr;
using ::tsl::testing::StatusIs;

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

  EXPECT_EQ(encoded(PrimitiveType::C64), encoded(DataType::C64));
  EXPECT_EQ(encoded(PrimitiveType::C128), encoded(DataType::C128));

  EXPECT_EQ(encoded(PrimitiveType::TOKEN), encoded(DataType::TOKEN));

  EXPECT_EQ(encoded(PrimitiveType::F8E5M2), encoded(DataType::F8E5M2));
  EXPECT_EQ(encoded(PrimitiveType::F8E4M3), encoded(DataType::F8E4M3));
  EXPECT_EQ(encoded(PrimitiveType::F8E4M3FN), encoded(DataType::F8E4M3FN));
  EXPECT_EQ(encoded(PrimitiveType::F8E4M3B11FNUZ),
            encoded(DataType::F8E4M3B11FNUZ));
  EXPECT_EQ(encoded(PrimitiveType::F8E5M2FNUZ), encoded(DataType::F8E5M2FNUZ));
  EXPECT_EQ(encoded(PrimitiveType::F8E4M3FNUZ), encoded(DataType::F8E4M3FNUZ));
  EXPECT_EQ(encoded(PrimitiveType::F8E3M4), encoded(DataType::F8E3M4));
}

TEST(FfiTest, DataTypeByteWidth) {
  EXPECT_EQ(0, ByteWidth(DataType::TOKEN));
  EXPECT_EQ(0, ByteWidth(DataType::INVALID));

  EXPECT_EQ(primitive_util::ByteWidth(PrimitiveType::PRED),
            ByteWidth(DataType::PRED));

  EXPECT_EQ(primitive_util::ByteWidth(PrimitiveType::S8),
            ByteWidth(DataType::S8));
  EXPECT_EQ(primitive_util::ByteWidth(PrimitiveType::S16),
            ByteWidth(DataType::S16));
  EXPECT_EQ(primitive_util::ByteWidth(PrimitiveType::S32),
            ByteWidth(DataType::S32));
  EXPECT_EQ(primitive_util::ByteWidth(PrimitiveType::S64),
            ByteWidth(DataType::S64));

  EXPECT_EQ(primitive_util::ByteWidth(PrimitiveType::U8),
            ByteWidth(DataType::U8));
  EXPECT_EQ(primitive_util::ByteWidth(PrimitiveType::U16),
            ByteWidth(DataType::U16));
  EXPECT_EQ(primitive_util::ByteWidth(PrimitiveType::U32),
            ByteWidth(DataType::U32));
  EXPECT_EQ(primitive_util::ByteWidth(PrimitiveType::U64),
            ByteWidth(DataType::U64));

  EXPECT_EQ(primitive_util::ByteWidth(PrimitiveType::F16),
            ByteWidth(DataType::F16));
  EXPECT_EQ(primitive_util::ByteWidth(PrimitiveType::F32),
            ByteWidth(DataType::F32));
  EXPECT_EQ(primitive_util::ByteWidth(PrimitiveType::F64),
            ByteWidth(DataType::F64));

  EXPECT_EQ(primitive_util::ByteWidth(PrimitiveType::BF16),
            ByteWidth(DataType::BF16));

  EXPECT_EQ(primitive_util::ByteWidth(PrimitiveType::C64),
            ByteWidth(DataType::C64));
  EXPECT_EQ(primitive_util::ByteWidth(PrimitiveType::C128),
            ByteWidth(DataType::C128));

  EXPECT_EQ(primitive_util::ByteWidth(PrimitiveType::F8E5M2),
            ByteWidth(DataType::F8E5M2));
  EXPECT_EQ(primitive_util::ByteWidth(PrimitiveType::F8E4M3),
            ByteWidth(DataType::F8E4M3));
  EXPECT_EQ(primitive_util::ByteWidth(PrimitiveType::F8E4M3FN),
            ByteWidth(DataType::F8E4M3FN));
  EXPECT_EQ(primitive_util::ByteWidth(PrimitiveType::F8E4M3B11FNUZ),
            ByteWidth(DataType::F8E4M3B11FNUZ));
  EXPECT_EQ(primitive_util::ByteWidth(PrimitiveType::F8E5M2FNUZ),
            ByteWidth(DataType::F8E5M2FNUZ));
  EXPECT_EQ(primitive_util::ByteWidth(PrimitiveType::F8E4M3FNUZ),
            ByteWidth(DataType::F8E4M3FNUZ));
  EXPECT_EQ(primitive_util::ByteWidth(PrimitiveType::F8E3M4),
            ByteWidth(DataType::F8E3M4));
}

TEST(FfiTest, ErrorEnumValue) {
  // Verify that absl::StatusCode and xla::ffi::ErrorCode use the same
  // integer value for encoding error (status) codes.
  auto encoded = [](auto value) { return static_cast<uint8_t>(value); };

  EXPECT_EQ(encoded(absl::StatusCode::kOk), encoded(ErrorCode::kOk));
  EXPECT_EQ(encoded(absl::StatusCode::kCancelled),
            encoded(ErrorCode::kCancelled));
  EXPECT_EQ(encoded(absl::StatusCode::kUnknown), encoded(ErrorCode::kUnknown));
  EXPECT_EQ(encoded(absl::StatusCode::kInvalidArgument),
            encoded(ErrorCode::kInvalidArgument));
  EXPECT_EQ(encoded(absl::StatusCode::kNotFound),
            encoded(ErrorCode::kNotFound));
  EXPECT_EQ(encoded(absl::StatusCode::kAlreadyExists),
            encoded(ErrorCode::kAlreadyExists));
  EXPECT_EQ(encoded(absl::StatusCode::kPermissionDenied),
            encoded(ErrorCode::kPermissionDenied));
  EXPECT_EQ(encoded(absl::StatusCode::kResourceExhausted),
            encoded(ErrorCode::kResourceExhausted));
  EXPECT_EQ(encoded(absl::StatusCode::kFailedPrecondition),
            encoded(ErrorCode::kFailedPrecondition));
  EXPECT_EQ(encoded(absl::StatusCode::kAborted), encoded(ErrorCode::kAborted));
  EXPECT_EQ(encoded(absl::StatusCode::kOutOfRange),
            encoded(ErrorCode::kOutOfRange));
  EXPECT_EQ(encoded(absl::StatusCode::kUnimplemented),
            encoded(ErrorCode::kUnimplemented));
  EXPECT_EQ(encoded(absl::StatusCode::kInternal),
            encoded(ErrorCode::kInternal));
  EXPECT_EQ(encoded(absl::StatusCode::kUnavailable),
            encoded(ErrorCode::kUnavailable));
  EXPECT_EQ(encoded(absl::StatusCode::kDataLoss),
            encoded(ErrorCode::kDataLoss));
  EXPECT_EQ(encoded(absl::StatusCode::kUnauthenticated),
            encoded(ErrorCode::kUnauthenticated));
}

TEST(FfiTest, Expected) {
  ErrorOr<int32_t> value(42);
  EXPECT_TRUE(value.has_value());
  EXPECT_FALSE(value.has_error());
  EXPECT_EQ(*value, 42);

  ErrorOr<int32_t> error(Error(ErrorCode::kInternal, "Test error"));
  EXPECT_FALSE(error.has_value());
  EXPECT_TRUE(error.has_error());
  EXPECT_THAT(error.error().message(), HasSubstr("Test error"));
}

TEST(FfiTest, FutureSetAvailable) {
  Promise promise;
  Future future(promise);

  promise.SetAvailable();
  future.OnReady([](const std::optional<Error>& error) {
    EXPECT_FALSE(error.has_value());
  });
}

TEST(FfiTest, FutureSetError) {
  Promise promise;
  Future future(promise);

  promise.SetError(Error(ErrorCode::kInternal, "Test error"));
  future.OnReady([](const std::optional<Error>& error) {
    EXPECT_TRUE(error.has_value());
    EXPECT_THAT(error->message(), HasSubstr("Test error"));
  });
}

TEST(FfiTest, FutureSetAvailableFromThreadPool) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "ffi-test", 2);

  Promise promise;
  Future future(promise);

  // We write and read to and from the shared variable to check that `OnReady`
  // callback is correctly synchronized with memory writes done in a thread
  // that completes the promise.
  int32_t value = 0;

  absl::BlockingCounter counter(1);

  future.OnReady([&](const std::optional<Error>& error) {
    EXPECT_FALSE(error.has_value());
    EXPECT_EQ(value, 42);
    counter.DecrementCount();
  });

  pool.Schedule([&]() {
    value = 42;
    promise.SetAvailable();
  });

  counter.Wait();
}

TEST(FfiTest, FutureSetErrorFromThreadPool) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "ffi-test", 2);

  Promise promise;
  Future future(promise);

  // We write and read to and from the shared variable to check that `OnReady`
  // callback is correctly synchronized with memory writes done in a thread
  // that completes the promise.
  int32_t value = 0;

  absl::BlockingCounter counter(1);

  future.OnReady([&](const std::optional<Error>& error) {
    EXPECT_TRUE(error.has_value());
    EXPECT_THAT(error->message(), HasSubstr("Test error"));
    EXPECT_EQ(value, 42);
    counter.DecrementCount();
  });

  pool.Schedule([&]() {
    value = 42;
    promise.SetError(Error(ErrorCode::kInternal, "Test error"));
  });

  counter.Wait();
}

TEST(FfiTest, FutureRace) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "ffi-test", 2);

  // Schedule `SetAvailable` and `OnReady` on a thread pool to detect
  // potential data races. Do this in a loop to make sure that we have
  // a good chance of triggering a data race if there is one.
  for (int32_t i = 0; i < 1000; ++i) {
    Promise promise;
    Future future(promise);

    absl::BlockingCounter counter(1);

    pool.Schedule([&]() { promise.SetAvailable(); });
    pool.Schedule([&]() {
      future.OnReady([&](const std::optional<Error>& error) {
        EXPECT_FALSE(error.has_value());
        counter.DecrementCount();
      });
    });

    counter.Wait();
  }
}

TEST(FfiTest, ReturnError) {
  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().To(
      []() { return Error(ErrorCode::kInternal, "Test error"); });

  auto status = Call(*handler, call_frame);
  EXPECT_EQ(status, absl::InternalError("Test error"));
}

TEST(FfiTest, AnyBufferArgument) {
  std::vector<float> storage(4, 0.0f);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(float));

  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/0);
  builder.AddBufferArg(memory, PrimitiveType::F32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Arg<AnyBuffer>().To([&](auto buffer) {
    EXPECT_EQ(buffer.untyped_data(), storage.data());
    EXPECT_EQ(buffer.template typed_data<float>(),
              reinterpret_cast<float*>(storage.data()));
    EXPECT_EQ(buffer.template reinterpret_data<int32_t>(),
              reinterpret_cast<int32_t*>(storage.data()));
    EXPECT_EQ(buffer.dimensions().size(), 2);
    return Error::Success();
  });
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, BufferArgument) {
  std::vector<float> storage(4, 0.0f);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(float));

  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/0);
  builder.AddBufferArg(memory, PrimitiveType::F32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Arg<BufferR2<F32>>().To([&](auto buffer) {
    EXPECT_EQ(buffer.typed_data(), storage.data());
    EXPECT_EQ(buffer.dimensions().size(), 2);
    return Error::Success();
  });
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, AnyBufferResult) {
  std::vector<float> storage(4, 0.0f);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(float));

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/1);
  builder.AddBufferRet(memory, PrimitiveType::F32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Ret<AnyBuffer>().To([&](Result<AnyBuffer> buffer) {
    EXPECT_EQ(buffer->untyped_data(), storage.data());
    EXPECT_EQ(buffer->template typed_data<float>(),
              reinterpret_cast<float*>(storage.data()));
    EXPECT_EQ(buffer->dimensions().size(), 2);
    return Error::Success();
  });
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, MissingBufferArgument) {
  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Arg<BufferR1<F32>>().To(
      [](auto) { return Error::Success(); });
  auto status = Call(*handler, call_frame);

  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("Wrong number of arguments")));
}

TEST(FfiTest, WrongRankBufferArgument) {
  std::vector<int32_t> storage(4, 0.0);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(int32_t));

  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/0);
  builder.AddBufferArg(memory, PrimitiveType::F32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Arg<BufferR1<F32>>().To(
      [](auto) { return Error::Success(); });
  auto status = Call(*handler, call_frame);

  EXPECT_THAT(status,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Wrong buffer rank: expected 1 but got 2")));
}

TEST(FfiTest, WrongTypeBufferArgument) {
  std::vector<int32_t> storage(4, 0.0);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(int32_t));

  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/0);
  builder.AddBufferArg(memory, PrimitiveType::S32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto handler = Ffi::Bind().Arg<BufferR2<F32>>().To(
      [](auto) { return Error::Success(); });
  auto status = Call(*handler, call_frame);

  EXPECT_THAT(
      status,
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Wrong buffer dtype: expected F32 but got S32")));
}

TEST(FfiTest, WrongNumberOfArguments) {
  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("foo", 42);
  attrs.Insert("bar", 43);

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto handler =
      Ffi::Bind().Attr<int>("foo").To([](int foo) { return Error::Success(); });
  auto status = Call(*handler, call_frame);

  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("Wrong number of attributes")));
  EXPECT_THAT(status.message(), HasSubstr("foo"));
  EXPECT_THAT(status.message(), HasSubstr("bar"));
}

TEST(FfiTest, TokenArgument) {
  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/0);
  builder.AddBufferArg(se::DeviceMemoryBase(), PrimitiveType::TOKEN,
                       /*dims=*/{});
  auto call_frame = builder.Build();

  auto fn = [&](Token tok) {
    EXPECT_EQ(tok.typed_data(), nullptr);
    EXPECT_EQ(tok.dimensions().size(), 0);
    return Error::Success();
  };

  auto handler = Ffi::Bind().Arg<Token>().To(fn);
  auto status = Call(*handler, call_frame);
  TF_ASSERT_OK(status);
}

TEST(FfiTest, RemainingArgs) {
  std::vector<float> storage(4, 0.0f);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(float));

  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/0);
  builder.AddBufferArg(memory, PrimitiveType::F32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto fn = [&](RemainingArgs args) {
    EXPECT_EQ(args.size(), 1);

    ErrorOr<AnyBuffer> arg0 = args.get<AnyBuffer>(0);
    ErrorOr<AnyBuffer> arg1 = args.get<AnyBuffer>(1);

    EXPECT_TRUE(arg0.has_value());
    EXPECT_FALSE(arg1.has_value());

    return Error::Success();
  };

  auto handler = Ffi::Bind().RemainingArgs().To(fn);
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, RemainingRets) {
  std::vector<float> storage(4, 0.0f);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(float));

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/2);
  builder.AddBufferRet(memory, PrimitiveType::F32, /*dims=*/{2, 2});
  builder.AddBufferRet(memory, PrimitiveType::F32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  auto fn = [&](Result<AnyBuffer> ret, RemainingRets rets) {
    EXPECT_EQ(rets.size(), 1);

    ErrorOr<Result<AnyBuffer>> ret0 = rets.get<AnyBuffer>(0);
    ErrorOr<Result<AnyBuffer>> ret1 = rets.get<AnyBuffer>(1);

    EXPECT_TRUE(ret0.has_value());
    EXPECT_FALSE(ret1.has_value());

    return Error::Success();
  };

  auto handler = Ffi::Bind().Ret<AnyBuffer>().RemainingRets().To(fn);
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, OptionalArgs) {
  std::vector<float> storage(4, 0.0f);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(float));

  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/0);
  builder.AddBufferArg(memory, PrimitiveType::F32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  {  // Single optional argument.
    auto fn = [&](std::optional<AnyBuffer> arg0) {
      EXPECT_TRUE(arg0.has_value());
      return Error::Success();
    };

    auto handler = Ffi::Bind().OptionalArg<AnyBuffer>().To(fn);
    auto status = Call(*handler, call_frame);

    TF_ASSERT_OK(status);
  }

  {  // Two optional arguments.
    auto fn = [&](std::optional<AnyBuffer> arg0,
                  std::optional<AnyBuffer> arg1) {
      EXPECT_TRUE(arg0.has_value());
      EXPECT_FALSE(arg1.has_value());
      return Error::Success();
    };

    auto handler =
        Ffi::Bind().OptionalArg<AnyBuffer>().OptionalArg<AnyBuffer>().To(fn);
    auto status = Call(*handler, call_frame);

    TF_ASSERT_OK(status);
  }

  {  // Optional argument after a regular one.
    auto fn = [&](AnyBuffer arg0, std::optional<AnyBuffer> arg1) {
      EXPECT_FALSE(arg1.has_value());
      return Error::Success();
    };

    auto handler = Ffi::Bind().Arg<AnyBuffer>().OptionalArg<AnyBuffer>().To(fn);
    auto status = Call(*handler, call_frame);

    TF_ASSERT_OK(status);
  }

  {  // Remaining arguments after optional one.
    auto fn = [&](std::optional<AnyBuffer> arg0, RemainingArgs args) {
      EXPECT_TRUE(arg0.has_value());
      EXPECT_EQ(args.size(), 0);
      return Error::Success();
    };

    auto handler = Ffi::Bind().OptionalArg<AnyBuffer>().RemainingArgs().To(fn);
    auto status = Call(*handler, call_frame);

    TF_ASSERT_OK(status);
  }
}

TEST(FfiTest, OptionalRets) {
  std::vector<float> storage(4, 0.0f);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(float));

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/1);
  builder.AddBufferRet(memory, PrimitiveType::F32, /*dims=*/{2, 2});
  auto call_frame = builder.Build();

  {  // Single optional result.
    auto fn = [&](std::optional<Result<AnyBuffer>> ret0) {
      EXPECT_TRUE(ret0.has_value());
      return Error::Success();
    };

    auto handler = Ffi::Bind().OptionalRet<AnyBuffer>().To(fn);
    auto status = Call(*handler, call_frame);

    TF_ASSERT_OK(status);
  }

  {  // Two optional results.
    auto fn = [&](std::optional<Result<AnyBuffer>> ret0,
                  std::optional<Result<AnyBuffer>> ret1) {
      EXPECT_TRUE(ret0.has_value());
      EXPECT_FALSE(ret1.has_value());
      return Error::Success();
    };

    auto handler =
        Ffi::Bind().OptionalRet<AnyBuffer>().OptionalRet<AnyBuffer>().To(fn);
    auto status = Call(*handler, call_frame);

    TF_ASSERT_OK(status);
  }

  {  // Optional result after a regular one.
    auto fn = [&](Result<AnyBuffer> ret0,
                  std::optional<Result<AnyBuffer>> ret1) {
      EXPECT_FALSE(ret1.has_value());
      return Error::Success();
    };

    auto handler = Ffi::Bind().Ret<AnyBuffer>().OptionalRet<AnyBuffer>().To(fn);
    auto status = Call(*handler, call_frame);

    TF_ASSERT_OK(status);
  }

  {  // Remaining results after optional one.
    auto fn = [&](std::optional<Result<AnyBuffer>> ret0, RemainingRets rets) {
      EXPECT_TRUE(ret0.has_value());
      EXPECT_EQ(rets.size(), 0);
      return Error::Success();
    };

    auto handler = Ffi::Bind().OptionalRet<AnyBuffer>().RemainingRets().To(fn);
    auto status = Call(*handler, call_frame);

    TF_ASSERT_OK(status);
  }
}

TEST(FfiTest, AutoBinding) {
  static constexpr char kI32[] = "i32";

  auto handler = Ffi::BindTo(+[](AnyBuffer buffer, Attr<int32_t, kI32> foo) {
    EXPECT_EQ(*foo, 42);
    return Error::Success();
  });

  std::vector<float> storage(4, 0.0f);
  se::DeviceMemoryBase memory(storage.data(), 4 * sizeof(float));

  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert(kI32, 42);

  CallFrameBuilder builder(/*num_args=*/1, /*num_rets=*/0);
  builder.AddBufferArg(memory, PrimitiveType::F32, /*dims=*/{2, 2});
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto status = Call(*handler, call_frame);
  TF_ASSERT_OK(status);
}

TEST(FfiTest, AutoBindingResult) {
  auto handler =
      Ffi::BindTo(+[](Result<AnyBuffer> buffer) { return Error::Success(); });

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/1);
  builder.AddBufferRet(se::DeviceMemoryBase(), PrimitiveType::F32, /*dims=*/{});
  auto call_frame = builder.Build();

  auto status = Call(*handler, call_frame);
  TF_ASSERT_OK(status);
}

TEST(FfiTest, AutoBindingStructs) {
  auto handler = Ffi::BindTo(+[](PairOfI32AndF32 attrs) {
    EXPECT_EQ(attrs.i32, 42);
    EXPECT_EQ(attrs.f32, 42.0f);
    return Error::Success();
  });

  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("i32", 42);
  attrs.Insert("f32", 42.0f);

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
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

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
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
  attrs.Insert("arr0", std::vector<int8_t>({1, 2, 3, 4}));
  attrs.Insert("arr1", std::vector<int16_t>({1, 2, 3, 4}));
  attrs.Insert("arr2", std::vector<int32_t>({1, 2, 3, 4}));
  attrs.Insert("arr3", std::vector<int64_t>({1, 2, 3, 4}));
  attrs.Insert("arr4", std::vector<uint8_t>({1, 2, 3, 4}));
  attrs.Insert("arr5", std::vector<uint16_t>({1, 2, 3, 4}));
  attrs.Insert("arr6", std::vector<uint32_t>({1, 2, 3, 4}));
  attrs.Insert("arr7", std::vector<uint64_t>({1, 2, 3, 4}));
  attrs.Insert("arr8", std::vector<float>({1, 2, 3, 4}));
  attrs.Insert("arr9", std::vector<double>({1, 2, 3, 4}));

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [&](auto arr0, auto arr1, auto arr2, auto arr3, auto arr4,
                auto arr5, auto arr6, auto arr7, auto arr8, auto arr9) {
    EXPECT_EQ(arr0, Span<const int8_t>({1, 2, 3, 4}));
    EXPECT_EQ(arr1, Span<const int16_t>({1, 2, 3, 4}));
    EXPECT_EQ(arr2, Span<const int32_t>({1, 2, 3, 4}));
    EXPECT_EQ(arr3, Span<const int64_t>({1, 2, 3, 4}));
    EXPECT_EQ(arr4, Span<const uint8_t>({1, 2, 3, 4}));
    EXPECT_EQ(arr5, Span<const uint16_t>({1, 2, 3, 4}));
    EXPECT_EQ(arr6, Span<const uint32_t>({1, 2, 3, 4}));
    EXPECT_EQ(arr7, Span<const uint64_t>({1, 2, 3, 4}));
    EXPECT_EQ(arr8, Span<const float>({1, 2, 3, 4}));
    EXPECT_EQ(arr9, Span<const double>({1, 2, 3, 4}));
    return Error::Success();
  };

  auto handler = Ffi::Bind()
                     .Attr<Span<const int8_t>>("arr0")
                     .Attr<Span<const int16_t>>("arr1")
                     .Attr<Span<const int32_t>>("arr2")
                     .Attr<Span<const int64_t>>("arr3")
                     .Attr<Span<const uint8_t>>("arr4")
                     .Attr<Span<const uint16_t>>("arr5")
                     .Attr<Span<const uint32_t>>("arr6")
                     .Attr<Span<const uint64_t>>("arr7")
                     .Attr<Span<const float>>("arr8")
                     .Attr<Span<const double>>("arr9")
                     .To(fn);
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, AttrsAsDictionary) {
  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("i32", 42);
  attrs.Insert("f32", 42.0f);
  attrs.Insert("str", "foo");

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [&](Dictionary dict) {
    EXPECT_EQ(dict.size(), 3);

    EXPECT_TRUE(dict.contains("i32"));
    EXPECT_TRUE(dict.contains("f32"));
    EXPECT_TRUE(dict.contains("str"));

    ErrorOr<int32_t> i32 = dict.get<int32_t>("i32");
    ErrorOr<float> f32 = dict.get<float>("f32");
    ErrorOr<std::string_view> str = dict.get<std::string_view>("str");

    EXPECT_TRUE(i32.has_value());
    EXPECT_TRUE(f32.has_value());
    EXPECT_TRUE(str.has_value());

    if (i32.has_value()) EXPECT_EQ(*i32, 42);
    if (f32.has_value()) EXPECT_EQ(*f32, 42.0f);
    if (str.has_value()) EXPECT_EQ(*str, "foo");

    EXPECT_FALSE(dict.contains("i64"));
    EXPECT_FALSE(dict.get<int64_t>("i32").has_value());
    EXPECT_FALSE(dict.get<int64_t>("i64").has_value());

    return Error::Success();
  };

  auto handler = Ffi::Bind().Attrs().To(fn);
  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, DictionaryAttr) {
  CallFrameBuilder::AttributesMap dict0;
  dict0.try_emplace("i32", 42);

  CallFrameBuilder::AttributesMap dict1;
  dict1.try_emplace("f32", 42.0f);

  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("dict0", dict0);
  attrs.Insert("dict1", dict1);

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [&](Dictionary dict0, Dictionary dict1) {
    EXPECT_EQ(dict0.size(), 1);
    EXPECT_EQ(dict1.size(), 1);

    EXPECT_TRUE(dict0.contains("i32"));
    EXPECT_TRUE(dict1.contains("f32"));

    ErrorOr<int32_t> i32 = dict0.get<int32_t>("i32");
    ErrorOr<float> f32 = dict1.get<float>("f32");

    EXPECT_TRUE(i32.has_value());
    EXPECT_TRUE(f32.has_value());

    if (i32.has_value()) EXPECT_EQ(*i32, 42);
    if (f32.has_value()) EXPECT_EQ(*f32, 42.0f);

    return Error::Success();
  };

  auto handler =
      Ffi::Bind().Attr<Dictionary>("dict0").Attr<Dictionary>("dict1").To(fn);

  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, StructAttr) {
  CallFrameBuilder::AttributesMap dict;
  dict.try_emplace("i32", 42);
  dict.try_emplace("f32", 42.0f);

  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("str", "foo");
  attrs.Insert("i32_and_f32", dict);

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [&](std::string_view str, PairOfI32AndF32 i32_and_f32) {
    EXPECT_EQ(str, "foo");
    EXPECT_EQ(i32_and_f32.i32, 42);
    EXPECT_EQ(i32_and_f32.f32, 42.0f);
    return Error::Success();
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

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [&](PairOfI32AndF32 i32_and_f32) {
    EXPECT_EQ(i32_and_f32.i32, 42);
    EXPECT_EQ(i32_and_f32.f32, 42.0f);
    return Error::Success();
  };

  auto handler = Ffi::Bind().Attrs<PairOfI32AndF32>().To(fn);
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

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
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

TEST(FfiTest, EnumAttr) {
  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("i32_one", static_cast<std::underlying_type_t<Int32BasedEnum>>(
                              Int32BasedEnum::kOne));
  attrs.Insert("i32_two", static_cast<std::underlying_type_t<Int32BasedEnum>>(
                              Int32BasedEnum::kTwo));
  attrs.Insert("i64_one", static_cast<std::underlying_type_t<Int64BasedEnum>>(
                              Int64BasedEnum::kOne));
  attrs.Insert("i64_two", static_cast<std::underlying_type_t<Int64BasedEnum>>(
                              Int64BasedEnum::kTwo));

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [&](Int32BasedEnum i32_one, Int32BasedEnum i32_two,
                Int64BasedEnum i64_one, Int64BasedEnum i64_two) {
    EXPECT_EQ(i32_one, Int32BasedEnum::kOne);
    EXPECT_EQ(i32_two, Int32BasedEnum::kTwo);
    EXPECT_EQ(i64_one, Int64BasedEnum::kOne);
    EXPECT_EQ(i64_two, Int64BasedEnum::kTwo);
    return Error::Success();
  };

  auto handler = Ffi::Bind()
                     .Attr<Int32BasedEnum>("i32_one")
                     .Attr<Int32BasedEnum>("i32_two")
                     .Attr<Int64BasedEnum>("i64_one")
                     .Attr<Int64BasedEnum>("i64_two")
                     .To(fn);

  auto status = Call(*handler, call_frame);

  TF_ASSERT_OK(status);
}

TEST(FfiTest, WrongEnumAttrType) {
  CallFrameBuilder::AttributesMap dict;
  dict.try_emplace("i32", 42);

  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("i32_enum1", dict);
  attrs.Insert("i32_enum0", 42u);

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  builder.AddAttributes(attrs.Build());
  auto call_frame = builder.Build();

  auto fn = [](Int32BasedEnum, Int32BasedEnum) { return Error::Success(); };

  auto handler = Ffi::Bind()
                     .Attr<Int32BasedEnum>("i32_enum0")
                     .Attr<Int32BasedEnum>("i32_enum1")
                     .To(fn);

  auto status = Call(*handler, call_frame);

  EXPECT_TRUE(absl::StrContains(
      status.message(),
      "Failed to decode all FFI handler operands (bad operands at: 0, 1)"))
      << "status.message():\n"
      << status.message() << "\n";

  EXPECT_TRUE(absl::StrContains(status.message(),
                                "Wrong scalar data type: expected S32 but got"))
      << "status.message():\n"
      << status.message() << "\n";

  EXPECT_TRUE(absl::StrContains(
      status.message(),
      "Wrong attribute type: expected scalar but got dictionary"))
      << "status.message():\n"
      << status.message() << "\n";
}

struct MyData {
  static TypeId id;
  std::string str;
};

TypeId MyData::id = {};  // zero-initialize type id
XLA_FFI_REGISTER_TYPE(GetXlaFfiApi(), "my_data", &MyData::id);

TEST(FfiTest, UserData) {
  MyData data{"foo"};

  ExecutionContext execution_context;
  TF_ASSERT_OK(execution_context.Insert(
      TypeIdRegistry::TypeId(MyData::id.type_id), &data));

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  auto call_frame = builder.Build();

  auto fn = [&](MyData* data) {
    EXPECT_EQ(data->str, "foo");
    return Error::Success();
  };

  auto handler = Ffi::Bind().Ctx<UserData<MyData>>().To(fn);

  CallOptions options;
  options.execution_context = &execution_context;

  auto status = Call(*handler, call_frame, options);

  TF_ASSERT_OK(status);
}

struct MyState {
  static TypeId id;

  explicit MyState(int32_t value) : value(value) {}
  int32_t value;
};

TypeId MyState::id = {};  // zero-initialize type id
XLA_FFI_REGISTER_TYPE(GetXlaFfiApi(), "state", &MyState::id);

TEST(FfiTest, StatefulHandler) {
  ExecutionState execution_state;

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
  auto call_frame = builder.Build();

  CallOptions options;
  options.execution_state = &execution_state;

  // FFI instantiation handler that creates a state for FFI handler.
  auto instantiate =
      Ffi::BindInstantiate().To([]() -> ErrorOr<std::unique_ptr<MyState>> {
        return std::make_unique<MyState>(42);
      });

  // FFI execute handler that uses state created by the instantiation handler.
  auto execute = Ffi::Bind().Ctx<State<MyState>>().To([](MyState* state) {
    EXPECT_EQ(state->value, 42);
    return Error::Success();
  });

  // Create `State` and store it in the execution state.
  TF_ASSERT_OK(
      Call(*instantiate, call_frame, options, ExecutionStage::kInstantiate));

  // Check that state was created and forwarded to the execute handler.
  TF_ASSERT_OK(Call(*execute, call_frame, options));
}

TEST(FfiTest, ScratchAllocator) {
  static void* kAddr = reinterpret_cast<void*>(0xDEADBEEF);

  // A test only memory allocator that returns a fixed memory address.
  struct TestDeviceMemoryAllocator final : public se::DeviceMemoryAllocator {
    size_t count;

    TestDeviceMemoryAllocator()
        : se::DeviceMemoryAllocator(nullptr), count(0) {}

    absl::StatusOr<se::OwningDeviceMemory> Allocate(int, uint64_t size, bool,
                                                    int64_t) final {
      count++;
      return se::OwningDeviceMemory(se::DeviceMemoryBase(kAddr, size), 0, this);
    }

    absl::Status Deallocate(int, se::DeviceMemoryBase mem) final {
      count--;
      EXPECT_EQ(mem.opaque(), kAddr);
      return absl::OkStatus();
    }

    absl::StatusOr<se::Stream*> GetStream(int) final {
      return absl::UnimplementedError("Not implemented");
    }
  };

  auto fn = [&](ScratchAllocator scratch_allocator) {
    auto mem = scratch_allocator.Allocate(1024);
    EXPECT_EQ(*mem, kAddr);
    return Error::Success();
  };

  TestDeviceMemoryAllocator allocator;

  auto handler = Ffi::Bind().Ctx<ScratchAllocator>().To(fn);

  CallFrame call_frame =
      CallFrameBuilder(/*num_args=*/0, /*num_rets=*/0).Build();

  CallOptions options;
  options.backend_options = CallOptions::GpuOptions{nullptr, &allocator};

  auto status = Call(*handler, call_frame, options);

  TF_ASSERT_OK(status);
  EXPECT_EQ(allocator.count, 0);
}

TEST(FfiTest, ScratchAllocatorUnimplemented) {
  auto fn = [&](ScratchAllocator scratch_allocator) {
    auto mem = scratch_allocator.Allocate(1024);
    EXPECT_FALSE(mem.has_value());
    return Error::Success();
  };
  auto handler = Ffi::Bind().Ctx<ScratchAllocator>().To(fn);
  CallFrame call_frame =
      CallFrameBuilder(/*num_args=*/0, /*num_rets=*/0).Build();
  auto status = Call(*handler, call_frame);
  TF_ASSERT_OK(status);
}

TEST(FfiTest, ThreadPool) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "ffi-test", 2);
  Eigen::ThreadPoolDevice device(pool.AsEigenThreadPool(), pool.NumThreads());

  auto fn = [&](ThreadPool thread_pool) {
    // Check that we can get the size of the underlying thread pool.
    if (thread_pool.num_threads() != 2) {
      return Error::Internal("Wrong number of threads");
    }

    // Use a pair of blocking counters to check that scheduled task was executed
    // on a thread pool (it would deadlock if executed inline).
    absl::BlockingCounter prepare(1);
    absl::BlockingCounter execute(1);

    thread_pool.Schedule([&] {
      prepare.Wait();
      execute.DecrementCount();
    });

    prepare.DecrementCount();
    execute.Wait();

    return Error::Success();
  };

  auto handler = Ffi::Bind().Ctx<ThreadPool>().To(fn);
  CallFrame call_frame =
      CallFrameBuilder(/*num_args=*/0, /*num_rets=*/0).Build();

  CallOptions options;
  options.backend_options = CallOptions::CpuOptions{&device};

  auto status = Call(*handler, call_frame, options);
  TF_ASSERT_OK(status);
}

TEST(FfiTest, AsyncHandler) {
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "ffi-test", 2);
  Eigen::ThreadPoolDevice device(pool.AsEigenThreadPool(), pool.NumThreads());

  int32_t value = 0;

  // Handler completes execution asynchronously on a given thread pool.
  auto fn = [&](ThreadPool thread_pool) -> Future {
    Promise promise;
    Future future(promise);

    thread_pool.Schedule([&, promise = std::move(promise)]() mutable {
      value = 42;
      promise.SetAvailable();
    });

    return future;
  };

  auto handler = Ffi::Bind().Ctx<ThreadPool>().To(fn);
  CallFrame call_frame =
      CallFrameBuilder(/*num_args=*/0, /*num_rets=*/0).Build();

  CallOptions options;
  options.backend_options = CallOptions::CpuOptions{&device};

  {  // Synchronous call.
    absl::Status status = Call(*handler, call_frame, options);
    TF_ASSERT_OK(status);
    EXPECT_EQ(value, 42);
  }

  value = 0;  // reset value between calls

  {  // Asynchronous call.
    tsl::AsyncValueRef<tsl::Chain> async_value =
        CallAsync(*handler, call_frame, options);
    tsl::BlockUntilReady(async_value);
    ASSERT_TRUE(async_value.IsConcrete());
    EXPECT_EQ(value, 42);
  }
}

TEST(FfiTest, Metadata) {
  auto api = GetXlaFfiApi();
  auto handler = Ffi::BindTo([]() { return Error::Success(); });
  auto maybe_metadata = GetMetadata(*handler);
  EXPECT_TRUE(maybe_metadata.ok());
  auto metadata = maybe_metadata.value();
  EXPECT_EQ(metadata.api_version.major_version, api->api_version.major_version);
  EXPECT_EQ(metadata.api_version.minor_version, api->api_version.minor_version);
  EXPECT_EQ(metadata.traits, 0);
}

TEST(FfiTest, MetadataTraits) {
  auto handler = Ffi::BindTo([]() { return Error::Success(); },
                             {Traits::kCmdBufferCompatible});
  auto maybe_metadata = GetMetadata(*handler);
  EXPECT_TRUE(maybe_metadata.ok());
  auto metadata = maybe_metadata.value();
  EXPECT_EQ(metadata.api_version.major_version, XLA_FFI_API_MAJOR);
  EXPECT_EQ(metadata.api_version.minor_version, XLA_FFI_API_MINOR);
  EXPECT_EQ(metadata.traits, XLA_FFI_HANDLER_TRAITS_COMMAND_BUFFER_COMPATIBLE);
}

//===----------------------------------------------------------------------===//
// Performance benchmarks are below.
//===----------------------------------------------------------------------===//

static CallFrameBuilder WithBufferArgs(size_t num_args, size_t rank = 4) {
  se::DeviceMemoryBase memory;
  std::vector<int64_t> dims(4, 1);

  CallFrameBuilder builder(/*num_args=*/num_args, /*num_rets=*/0);
  for (size_t i = 0; i < num_args; ++i) {
    builder.AddBufferArg(memory, PrimitiveType::F32, dims);
  }
  return builder;
}

//===----------------------------------------------------------------------===//
// BM_AnyBufferArgX1
//===----------------------------------------------------------------------===//

void BM_AnyBufferArgX1(benchmark::State& state) {
  auto call_frame = WithBufferArgs(1).Build();

  auto handler = Ffi::Bind().Arg<AnyBuffer>().To([](auto buffer) {
    benchmark::DoNotOptimize(buffer);
    return Error::Success();
  });

  for (auto _ : state) {
    CHECK_OK(Call(*handler, call_frame));
  }
}

BENCHMARK(BM_AnyBufferArgX1);

//===----------------------------------------------------------------------===//
// BM_AnyBufferArgX4
//===----------------------------------------------------------------------===//

void BM_AnyBufferArgX4(benchmark::State& state) {
  auto call_frame = WithBufferArgs(4).Build();

  auto handler = Ffi::Bind()
                     .Arg<AnyBuffer>()
                     .Arg<AnyBuffer>()
                     .Arg<AnyBuffer>()
                     .Arg<AnyBuffer>()
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

BENCHMARK(BM_AnyBufferArgX4);

//===----------------------------------------------------------------------===//
// BM_AsyncAnyBufferArgX1
//===----------------------------------------------------------------------===//

void BM_AsyncAnyBufferArgX1(benchmark::State& state) {
  auto call_frame = WithBufferArgs(1).Build();

  auto handler = Ffi::Bind().Arg<AnyBuffer>().To([](auto buffer) {
    benchmark::DoNotOptimize(buffer);
    Promise promise;
    promise.SetAvailable();
    return Future(promise);
  });

  for (auto _ : state) {
    CHECK_OK(Call(*handler, call_frame));
  }
}

BENCHMARK(BM_AsyncAnyBufferArgX1);

//===----------------------------------------------------------------------===//
// BM_BufferArgX1
//===----------------------------------------------------------------------===//

void BM_BufferArgX1(benchmark::State& state) {
  auto call_frame = WithBufferArgs(1).Build();

  auto handler = Ffi::Bind().Arg<BufferR4<F32>>().To([](auto buffer) {
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
                     .Arg<BufferR4<F32>>()
                     .Arg<BufferR4<F32>>()
                     .Arg<BufferR4<F32>>()
                     .Arg<BufferR4<F32>>()
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
// BM_BufferArgX8
//===----------------------------------------------------------------------===//

void BM_BufferArgX8(benchmark::State& state) {
  auto call_frame = WithBufferArgs(8).Build();

  auto handler = Ffi::Bind()
                     .Arg<BufferR4<F32>>()
                     .Arg<BufferR4<F32>>()
                     .Arg<BufferR4<F32>>()
                     .Arg<BufferR4<F32>>()
                     .Arg<BufferR4<F32>>()
                     .Arg<BufferR4<F32>>()
                     .Arg<BufferR4<F32>>()
                     .Arg<BufferR4<F32>>()
                     .To([](auto b0, auto b1, auto b2, auto b3, auto b4,
                            auto b5, auto b6, auto b7) {
                       benchmark::DoNotOptimize(b0);
                       benchmark::DoNotOptimize(b1);
                       benchmark::DoNotOptimize(b2);
                       benchmark::DoNotOptimize(b3);
                       benchmark::DoNotOptimize(b4);
                       benchmark::DoNotOptimize(b5);
                       benchmark::DoNotOptimize(b6);
                       benchmark::DoNotOptimize(b7);
                       return Error::Success();
                     });

  for (auto _ : state) {
    CHECK_OK(Call(*handler, call_frame));
  }
}

BENCHMARK(BM_BufferArgX8);

//===----------------------------------------------------------------------===//
// BM_TupleOfI32Attrs
//===----------------------------------------------------------------------===//

void BM_TupleOfI32Attrs(benchmark::State& state) {
  CallFrameBuilder::AttributesBuilder attrs;
  attrs.Insert("i32_0", 1);
  attrs.Insert("i32_1", 2);
  attrs.Insert("i32_2", 3);
  attrs.Insert("i32_3", 4);

  CallFrameBuilder builder(/*num_args=*/0, /*num_rets=*/0);
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
