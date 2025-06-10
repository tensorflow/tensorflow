/* Copyright 2017 The OpenXLA Authors.

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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/dynamic_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/types/span.h"
#include "xla/array2d.h"
#include "xla/array3d.h"
#include "xla/client/client_library.h"
#include "xla/client/local_client.h"
#include "xla/executable_run_options.h"
#include "xla/ffi/execution_context.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/platform_util.h"
#include "xla/service/service.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_macros.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

static const char* PLATFORM = "Host";

enum class BinaryOp : int8_t { kAdd, kMul };
enum class InitMethod : int { kZero, kOne };

XLA_FFI_REGISTER_ENUM_ATTR_DECODING(BinaryOp);
XLA_FFI_REGISTER_ENUM_ATTR_DECODING(InitMethod);

namespace xla {
namespace {

absl::Status R0F32Add2(ffi::Result<ffi::Buffer<PrimitiveType::F32>> out,
                       ffi::Buffer<PrimitiveType::F32> in) {
  *out->typed_data() = *in.typed_data() + 2.0f;
  return absl::OkStatus();
}

absl::Status R0F32Add2InPlace(ffi::Result<ffi::Buffer<PrimitiveType::F32>> out,
                              ffi::Buffer<PrimitiveType::F32> in) {
  *in.typed_data() = *in.typed_data() + 2.0f;
  return absl::OkStatus();
}

absl::Status R2F32ReduceSum(ffi::Result<ffi::Buffer<PrimitiveType::F32>> out,
                            ffi::Buffer<PrimitiveType::F32> in) {
  float* array = in.typed_data();
  *out->typed_data() = array[0] + array[1] + array[2] + array[3];
  return absl::OkStatus();
}

absl::Status Add1ToValues(ffi::Result<ffi::Buffer<PrimitiveType::F32>> out,
                          ffi::Buffer<PrimitiveType::F32> in) {
  float* array = in.typed_data();
  float* out_data = out->typed_data();
  out_data[0] = array[0] + 1;
  out_data[1] = array[1] + 1;
  out_data[2] = array[2] + 1;
  out_data[3] = array[3] + 1;
  return absl::OkStatus();
}

void R0F32Add2Succeed(float* out, float** in, XlaCustomCallStatus*) {
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(in, sizeof(float*));
  *out = **in + 2.0f;
  // Default state of 'status' is success.
}

void CustomCallFail(float*, float** in, XlaCustomCallStatus* status) {
  auto msg = absl::StrFormat("Failed: %.1f", in[0][0]);
  XlaCustomCallStatusSetFailure(status, msg.data(), msg.length());
}

void CustomCallFailWithBackendConfigStr(float*, float**, const char* opaque,
                                        size_t opaque_len,
                                        XlaCustomCallStatus* status) {
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(opaque, opaque_len);
  auto msg = absl::StrFormat("Fail with raw backend config str: %s.",
                             absl::string_view(opaque, opaque_len));
  XlaCustomCallStatusSetFailure(status, msg.data(), msg.length());
}

XLA_FFI_DEFINE_HANDLER(kR0F32Add2, R0F32Add2,
                       ffi::Ffi::Bind()
                           .Ret<ffi::Buffer<PrimitiveType::F32>>()
                           .Arg<ffi::Buffer<PrimitiveType::F32>>());

XLA_FFI_DEFINE_HANDLER(kR0F32Add2InPlace, R0F32Add2InPlace,
                       ffi::Ffi::Bind()
                           .Ret<ffi::Buffer<PrimitiveType::F32>>()
                           .Arg<ffi::Buffer<PrimitiveType::F32>>());

XLA_FFI_DEFINE_HANDLER(kR2F32ReduceSum, R2F32ReduceSum,
                       ffi::Ffi::Bind()
                           .Ret<ffi::Buffer<PrimitiveType::F32>>()
                           .Arg<ffi::Buffer<PrimitiveType::F32>>());

XLA_FFI_DEFINE_HANDLER(kAdd1ToValues, Add1ToValues,
                       ffi::Ffi::Bind()
                           .Ret<ffi::Buffer<PrimitiveType::F32>>()
                           .Arg<ffi::Buffer<PrimitiveType::F32>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "R0F32Add2", PLATFORM,
                         kR0F32Add2);

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "R0F32Add2InPlace", PLATFORM,
                         kR0F32Add2InPlace);

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "R2F32ReduceSum", PLATFORM,
                         kR2F32ReduceSum);

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "Add1ToValues", PLATFORM,
                         kAdd1ToValues);

XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(R0F32Add2Succeed);
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(CustomCallFail);
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(CustomCallFailWithBackendConfigStr);

std::ostream& operator<<(std::ostream& os, BinaryOp op) {
  switch (op) {
    case BinaryOp::kAdd:
      return os << "add";
    case BinaryOp::kMul:
      return os << "mul";
  }
}

std::ostream& operator<<(std::ostream& os, InitMethod op) {
  switch (op) {
    case InitMethod::kZero:
      return os << "zero";
    case InitMethod::kOne:
      return os << "one";
  }
}

using ::testing::HasSubstr;

class CustomCallTest : public HloPjRtTestBase {
 protected:
  Shape r0f32_ = ShapeUtil::MakeShape(F32, {});
  Shape r2f32_ = ShapeUtil::MakeShape(F32, {2, 2});
};

TEST_F(CustomCallTest, CustomCallR0F32Add2) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r0f32_, {constant}, "R0F32Add2", "",
      CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {}));
  LiteralTestUtil::ExpectR0Near<float>(44.0f, result, kDefaultErrorSpec);
}

TEST_F(CustomCallTest, CustomCallR0F32Add2Aliased) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));

  builder
      .AddInstruction(HloInstruction::CreateCustomCall(
          r0f32_, {constant}, "R0F32Add2InPlace", "",
          CustomCallApiVersion::API_VERSION_TYPED_FFI))
      ->set_output_to_operand_aliasing({{{}, {0, {}}}});

  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {}));
  LiteralTestUtil::ExpectR0Near<float>(44.0f, result, kDefaultErrorSpec);
}

TEST_F(CustomCallTest, CustomCallR2F32Reduce) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  Array2D<float> array(2, 2);
  array(0, 0) = 1.0f;
  array(0, 1) = 2.0f;
  array(1, 0) = 3.0f;
  array(1, 1) = 4.0f;

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2FromArray2D(array)));
  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r0f32_, {constant}, "R2F32ReduceSum", "",
      CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {}));
  LiteralTestUtil::ExpectR0Near<float>(10.0f, result, kDefaultErrorSpec);
}

TEST_F(CustomCallTest, ReportsSuccess) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r0f32_, {constant}, "R0F32Add2Succeed",
      /*opaque=*/"", CustomCallApiVersion::API_VERSION_STATUS_RETURNING));

  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {}));
  LiteralTestUtil::ExpectR0Near<float>(44.0f, result, kDefaultErrorSpec);
}

TEST_F(CustomCallTest, ReportsFailure) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  builder.AddInstruction(HloInstruction::CreateCustomCall(
      ShapeUtil::MakeShape(F32, {}), {constant}, "CustomCallFail",
      /*opaque=*/"", CustomCallApiVersion::API_VERSION_STATUS_RETURNING));

  module->AddEntryComputation(builder.Build());

  auto status = Execute(std::move(module), {}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(), ::testing::HasSubstr("Failed: 42.0"));
}

TEST_F(CustomCallTest, ReportsFirstFailure) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto constant_1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  auto res_1 = builder.AddInstruction(HloInstruction::CreateCustomCall(
      ShapeUtil::MakeShape(F32, {}), {constant_1}, "CustomCallFail",
      /*opaque=*/"", CustomCallApiVersion::API_VERSION_STATUS_RETURNING));
  auto res_2 = builder.AddInstruction(HloInstruction::CreateCustomCall(
      ShapeUtil::MakeShape(F32, {}), {res_1}, "CustomCallFail",
      /*opaque=*/"", CustomCallApiVersion::API_VERSION_STATUS_RETURNING));
  builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {}), HloOpcode::kAdd, res_1, res_2));

  module->AddEntryComputation(builder.Build());

  auto status = Execute(std::move(module), {}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(), ::testing::HasSubstr("Failed: 1.0"));
}

TEST_F(CustomCallTest, TransitiveCustomCallReportsFirstFailure) {
  const char* const kModuleStr = R"(
    HloModule m
    sub {
      p0 = f32[] parameter(0)
      ROOT custom-call = f32[] custom-call(f32[] %p0), custom_call_target="CustomCallFail", api_version=API_VERSION_STATUS_RETURNING
    }
    ENTRY test {
      c0 = f32[] constant(1.0)
      call0 = f32[] call(f32[] %c0), to_apply=sub
      call1 = f32[] call(f32[] %call0), to_apply=sub
      ROOT sum = f32[] add(%call0, %call1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  auto status = Execute(std::move(module), {}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(), HasSubstr("Failed: 1.0"));
}

TEST_F(CustomCallTest, FillStatusMsgWithBackendConfigStr) {
  const char* const kModuleStr = R"(
    HloModule m
    ENTRY test {
      c0 = f32[] constant(1.0)
      ROOT dummy-result = f32[] custom-call(f32[] %c0),
                                custom_call_target="CustomCallFailWithBackendConfigStr",
                                backend_config="foo",
                                api_version=API_VERSION_STATUS_RETURNING_UNIFIED
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  auto status = Execute(std::move(module), {}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(),
              HasSubstr("Fail with raw backend config str: foo"));
}

class CustomCallClientAPITest
    : public ClientLibraryTestRunnerMixin<
          HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>> {};

// When using the client API, CustomCall targets can't begin with '$' -- these
// are reserved for internal use.
TEST_F(CustomCallClientAPITest, IllegalCustomCallTarget) {
  XlaBuilder builder(TestName());
  CustomCall(&builder, "$illegal", /*operands=*/{},
             ShapeUtil::MakeShape(F32, {1}));

  EXPECT_IS_NOT_OK(ExecuteAndTransfer(&builder, /*arguments=*/{}).status());
}

//===----------------------------------------------------------------------===//
// XLA runtime custom call provides type-safe custom call API
//===----------------------------------------------------------------------===//

namespace {

// TODO(abanas): The following three usings are a workaround, delete when
// ResultBuffer is implemented as its own class
using ResultBufferBase = ffi::Result<ffi::AnyBuffer>;
template <PrimitiveType dtype, size_t rank = xla::ffi::internal::kDynamicRank>
using ResultBuffer = ffi::Result<ffi::Buffer<dtype, rank>>;
template <PrimitiveType dtype>
using ResultBufferR0 = ResultBuffer<dtype, 0>;

using R0F32Buffer = typename ffi::BufferR0<PrimitiveType::F32>;
using F32Buffer = typename ffi::Buffer<PrimitiveType::F32>;
using R0F32ResultBuffer = ResultBufferR0<PrimitiveType::F32>;
using F32ResultBuffer = ResultBuffer<PrimitiveType::F32>;
using AnyBuffer = ffi::AnyBuffer;

// Custom kernels definitions and registrations
static absl::Status AlwaysSucceed(ResultBufferBase) { return absl::OkStatus(); }

XLA_FFI_DEFINE_HANDLER(kAlwaysSucceed, AlwaysSucceed,
                       ffi::Ffi::Bind().Ret<AnyBuffer>()  // unused out buffer
);

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$always_succeed",
                         PLATFORM, kAlwaysSucceed);

static absl::Status AlwaysFail(ResultBufferBase, int32_t value) {
  return absl::InternalError(absl::StrCat("Failed: ", value));
}

XLA_FFI_DEFINE_HANDLER(kAlwaysFail, AlwaysFail,
                       ffi::Ffi::Bind()
                           .Ret<AnyBuffer>()        // unused out buffer
                           .Attr<int32_t>("value")  // value
);

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$always_fail",
                         PLATFORM, kAlwaysFail);

static absl::Status Tokens(ffi::Token, ffi::Result<AnyBuffer>,
                           ffi::Result<ffi::Token>) {
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(
    kTokens, Tokens,
    ffi::Ffi::Bind().Arg<ffi::Token>().Ret<AnyBuffer>().Ret<ffi::Token>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$tokens", PLATFORM,
                         kTokens);

static absl::Status FfiR0F32Add2(R0F32Buffer in, R0F32ResultBuffer out) {
  auto in_data = in.typed_data();
  auto out_data = out->typed_data();
  *out_data = *in_data + 2.0f;
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kFfiR0F32Add2, FfiR0F32Add2,
                       ffi::Ffi::Bind()
                           .Arg<R0F32Buffer>()  // in
                           .Ret<R0F32Buffer>()  // out
);

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$FfiR0F32Add2",
                         PLATFORM, kFfiR0F32Add2);

template <PrimitiveType dtype>
static absl::Status R0FAdd2(AnyBuffer in, ResultBufferBase out) {
  using NativeType =
      typename ::xla::primitive_util::PrimitiveTypeToNative<dtype>::type;

  auto in_data = reinterpret_cast<const NativeType*>(in.untyped_data());
  auto out_data = reinterpret_cast<NativeType*>(out->untyped_data());
  *out_data = *in_data + 2.0f;

  return absl::OkStatus();
}

// This represents a kernel that is valid only for F32 and F64 types
static absl::Status FfiR0FAdd2BufferBase(AnyBuffer in, ResultBufferBase out) {
  if (in.element_type() != out->element_type()) {
    return absl::InternalError("Input and output dtypes mismatch");
  }

  switch (in.element_type()) {
    case PrimitiveType::F32:
      return R0FAdd2<PrimitiveType::F32>(in, out);
    case PrimitiveType::F64:
      return R0FAdd2<PrimitiveType::F64>(in, out);
    default:
      return absl::InternalError("Incorrect type");
  }
}

XLA_FFI_DEFINE_HANDLER(kFfiR0FAdd2BufferBase, FfiR0FAdd2BufferBase,
                       ffi::Ffi::Bind()
                           .Arg<AnyBuffer>()  // in
                           .Ret<AnyBuffer>()  // out
);

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
                         "__xla_test$$FfiR0FAdd2BufferBase", PLATFORM,
                         kFfiR0FAdd2BufferBase);

static absl::Status FfiR0F32AddN(R0F32Buffer in, R0F32ResultBuffer out,
                                 float n) {
  auto in_data = in.typed_data();
  auto out_data = out->typed_data();
  *out_data = *in_data + n;
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kFfiR0F32AddN, FfiR0F32AddN,
                       ffi::Ffi::Bind()
                           .Arg<R0F32Buffer>()  // in
                           .Ret<R0F32Buffer>()  // out
                           .Attr<float>("n"));

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$FfiR0F32AddN",
                         PLATFORM, kFfiR0F32AddN);

static absl::Status FfiR0F32AddNPointer(R0F32Buffer in, R0F32ResultBuffer out,
                                        float* n) {
  auto in_data = in.typed_data();
  auto out_data = out->typed_data();
  *out_data = *in_data + *n;
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kFfiR0F32AddNPointer, FfiR0F32AddNPointer,
                       ffi::Ffi::Bind()
                           .Arg<R0F32Buffer>()  // in
                           .Ret<R0F32Buffer>()  // out
                           .Attr<ffi::Pointer<float>>("n"));

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$FfiR0F32AddNPointer",
                         PLATFORM, kFfiR0F32AddNPointer);

static absl::Status FfiF32ReduceSum(F32Buffer in, R0F32ResultBuffer out) {
  auto in_data = in.typed_data();
  auto out_data = out->typed_data();
  auto size = in.element_count();

  // Calculate the sum of the vector
  *out_data = absl::c_accumulate(absl::MakeSpan(in_data, size), 0.0f);

  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kFfiF32ReduceSum, FfiF32ReduceSum,
                       ffi::Ffi::Bind()
                           .Arg<F32Buffer>()    // in
                           .Ret<R0F32Buffer>()  // out
);

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$FfiF32ReduceSum",
                         PLATFORM, kFfiF32ReduceSum);

static absl::Status FfiF32Accumulate(F32Buffer in, InitMethod init,
                                     R0F32ResultBuffer out,
                                     BinaryOp binary_op) {
  auto in_data = in.typed_data();
  auto out_data = out->typed_data();

  // Init method is an artificial enum to demonstrate handling enums with
  // different underlying types. Normally it would be just a float scalar.
  float init_value = (init == InitMethod::kZero) ? 0.0f : 1.0f;

  // Calculate the total size of the vector
  auto size = in.element_count();

  // Calculate the sum or the product of the vector, based on binary_op value.
  switch (binary_op) {
    case BinaryOp::kAdd:
      *out_data = absl::c_accumulate(absl::MakeSpan(in_data, size), init_value);
      break;
    case BinaryOp::kMul:
      *out_data = absl::c_accumulate(absl::MakeSpan(in_data, size), init_value,
                                     std::multiplies<float>());
      break;
  }

  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kFfiF32Accumulate, FfiF32Accumulate,
                       ffi::Ffi::Bind()
                           .Arg<F32Buffer>(/*in*/)
                           .Attr<InitMethod>("init")
                           .Ret<R0F32Buffer>(/*out*/)
                           .Attr<BinaryOp>("binary_op"));

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$FfiF32Accumulate",
                         PLATFORM, kFfiF32Accumulate);

static absl::Status FfiF32Add1ToValues(F32Buffer in, F32ResultBuffer out) {
  auto in_data = in.typed_data();
  auto out_data = out->typed_data();

  // Calculate and verify the total size of the vector
  const auto in_size = in.element_count();
  const auto out_size = out->element_count();
  if (in_size != out_size) {
    return absl::InternalError("Input and output sizes mismatch");
  }

  // Actual computations
  std::transform(in_data, in_data + in_size, out_data,
                 [](float x) { return x + 1; });

  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kFfiF32Add1ToValues, FfiF32Add1ToValues,
                       ffi::Ffi::Bind()
                           .Arg<F32Buffer>()  // in
                           .Ret<F32Buffer>()  // out
);

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$FfiF32Add1ToValues",
                         PLATFORM, kFfiF32Add1ToValues);

static absl::Status FfiF32TupleSwap(R0F32Buffer in0, R0F32Buffer in1,
                                    R0F32ResultBuffer out0,
                                    R0F32ResultBuffer out1) {
  auto in_data0 = in0.typed_data();
  auto in_data1 = in1.typed_data();
  auto out_data0 = out0->typed_data();
  auto out_data1 = out1->typed_data();
  *out_data0 = *in_data1;
  *out_data1 = *in_data0;
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kFfiF32TupleSwap, FfiF32TupleSwap,
                       ffi::Ffi::Bind()
                           .Arg<R0F32Buffer>()  // in0
                           .Arg<R0F32Buffer>()  // in1
                           .Ret<R0F32Buffer>()  // out0
                           .Ret<R0F32Buffer>()  // out1
);

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$FfiF32TupleSwap",
                         PLATFORM, kFfiF32TupleSwap);

static absl::Status FfiTupleRotate(R0F32Buffer in0, R0F32Buffer in1,
                                   R0F32Buffer in2, R0F32Buffer in3,
                                   R0F32ResultBuffer out0,
                                   R0F32ResultBuffer out1,
                                   R0F32ResultBuffer out2,
                                   R0F32ResultBuffer out3) {
  auto in_data0 = in0.typed_data();
  auto in_data1 = in1.typed_data();
  auto in_data2 = in2.typed_data();
  auto in_data3 = in3.typed_data();
  auto out_data0 = out0->typed_data();
  auto out_data1 = out1->typed_data();
  auto out_data2 = out2->typed_data();
  auto out_data3 = out3->typed_data();
  *out_data0 = *in_data1;
  *out_data1 = *in_data2;
  *out_data2 = *in_data3;
  *out_data3 = *in_data0;
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kFfiTupleRotate, FfiTupleRotate,
                       ffi::Ffi::Bind()
                           .Arg<R0F32Buffer>()  // in0
                           .Arg<R0F32Buffer>()  // in1
                           .Arg<R0F32Buffer>()  // in2
                           .Arg<R0F32Buffer>()  // in3
                           .Ret<R0F32Buffer>()  // out0
                           .Ret<R0F32Buffer>()  // out1
                           .Ret<R0F32Buffer>()  // out2
                           .Ret<R0F32Buffer>()  // out3
);

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$FfiTupleRotate",
                         PLATFORM, kFfiTupleRotate);

static absl::Status VerifyR2Dimensions(ffi::AnyBuffer in, int32_t rows,
                                       int32_t cols) {
  std::string message;
  if (in.dimensions().size() != 2) {
    message += absl::StrFormat("dimensions.size() != 2 because %d != 2\n",
                               in.dimensions().size());
  }
  if (in.dimensions().front() != rows) {
    message += absl::StrFormat("dimensions.front() != rows because %d != %d\n",
                               in.dimensions().front(), rows);
  }
  if (in.dimensions().back() != cols) {
    message += absl::StrFormat("dimensions.back() != cols because %d != %d\n",
                               in.dimensions().back(), cols);
  }
  if (!message.empty()) {
    return absl::Status(absl::StatusCode::kFailedPrecondition,
                        std::move(message));
  }
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kVerifyR2Dimensions, VerifyR2Dimensions,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()  // in
                           .Attr<int32_t>("rows")
                           .Attr<int32_t>("cols"));

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$VerifyR2Dimensions",
                         PLATFORM, kVerifyR2Dimensions);

static absl::Status SwapTupleAnyBuffersToS16U32(ffi::AnyBuffer in_1,
                                                ffi::AnyBuffer in_2,
                                                ResultBufferR0<S16> out_1,
                                                ResultBufferR0<U32> out_2) {
  auto tuple_elem_1 = reinterpret_cast<uint32_t*>(in_1.untyped_data());
  auto tuple_elem_2 = reinterpret_cast<int16_t*>(in_2.untyped_data());
  out_1->typed_data()[0] = tuple_elem_2[0];
  out_2->typed_data()[0] = tuple_elem_1[0];
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kSwapTupleAnyBuffersToS16U32,
                       SwapTupleAnyBuffersToS16U32,
                       ffi::Ffi::Bind()
                           .Arg<ffi::AnyBuffer>()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::BufferR0<S16>>()
                           .Ret<ffi::BufferR0<U32>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
                         "__xla_test$$SwapTupleAnyBuffersToS16U32", PLATFORM,
                         kSwapTupleAnyBuffersToS16U32);

static absl::Status SwapTupleU32S16ToS16U32(ffi::BufferR0<U32> in_1,
                                            ffi::BufferR0<S16> in_2,
                                            ResultBufferR0<S16> out_1,
                                            ResultBufferR0<U32> out_2) {
  auto tuple_elem_1 = in_1.typed_data();
  auto tuple_elem_2 = in_2.typed_data();
  out_1->typed_data()[0] = tuple_elem_2[0];
  out_2->typed_data()[0] = tuple_elem_1[0];
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kSwapTupleU32S16ToS16U32, SwapTupleU32S16ToS16U32,
                       (ffi::Ffi::Bind()
                            .Arg<ffi::BufferR0<U32>>()
                            .Arg<ffi::BufferR0<S16>>()
                            .Ret<ffi::BufferR0<S16>>()
                            .Ret<ffi::BufferR0<U32>>()));

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
                         "__xla_test$$SwapTupleU32S16ToS16U32", PLATFORM,
                         kSwapTupleU32S16ToS16U32);

static absl::Status HandleTupleDifferentRanks(ffi::BufferR0<U32> x_1,
                                              ffi::BufferR1<S16> x_2,
                                              ffi::BufferR2<F32> y_1,
                                              ffi::BufferR3<F32> y_2,
                                              ResultBuffer<S32, 1> x_out,
                                              ResultBuffer<F32, 3> y_out) {
  if (x_2.element_count() != x_out->element_count()) {
    return absl::FailedPreconditionError(
        "`x_2` parameter should have the same number of elements as `x_out`");
  }
  if (y_1.dimensions() != y_out->dimensions().subspan(1) ||
      y_2.dimensions().front() + 1 != y_out->dimensions().front()) {
    return absl::FailedPreconditionError(
        "Cannot concatenate `y_1` and `y_2` due to dimensions mismatch. "
        "`y_2` dimensions should represent a batched `y_1`");
  }
  // Multiply R1 vector by R0 scalar
  const auto factor = x_1.typed_data()[0];
  for (int i = 0; i < x_2.element_count(); ++i) {
    x_out->typed_data()[i] = factor * x_2.typed_data()[i];
  }
  // Append R2 buffer to R3 buffer
  auto last_pos =
      std::copy_n(y_2.typed_data(), y_2.element_count(), y_out->typed_data());
  std::copy_n(y_1.typed_data(), y_1.element_count(), last_pos);
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kHandleTupleDifferentRanks, HandleTupleDifferentRanks,
                       ffi::Ffi::Bind()
                           .Arg<ffi::BufferR0<U32>>()
                           .Arg<ffi::BufferR1<S16>>()
                           .Arg<ffi::BufferR2<F32>>()
                           .Arg<ffi::BufferR3<F32>>()
                           .Ret<ffi::BufferR1<S32>>()
                           .Ret<ffi::BufferR3<F32>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
                         "__xla_test$$HandleTupleDifferentRanks", PLATFORM,
                         kHandleTupleDifferentRanks);

static absl::Status CustomCallWithIntraOpThreadPool(
    ffi::Result<ffi::AnyBuffer>,
    const Eigen::ThreadPoolDevice* intra_op_thread_pool) {
  // We use two blocking counters to ensure that the task is actually running
  // inside a thread pool.
  absl::BlockingCounter counter0(1);
  absl::BlockingCounter counter1(1);

  intra_op_thread_pool->getPool()->Schedule([&]() {
    counter0.Wait();
    counter1.DecrementCount();
  });

  // Unblock submitted task.
  counter0.DecrementCount();

  // TODO(b/356389210): It is unsafe to wait for the completion of a task
  // submitted into an intra-op thread pool as we might be running on a thread
  // inside the same thread pool, and this can lead to deadlocks. Custom calls
  // should return `AsyncValue` to signal completion of all submitted tasks.
  counter1.Wait();

  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kIntraOpThreadPool, CustomCallWithIntraOpThreadPool,
                       ffi::Ffi::Bind()
                           .Ret<AnyBuffer>()  // unused out buffer
                           .Ctx<ffi::IntraOpThreadPool>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
                         "__xla_test$$intra_op_thread_pool", PLATFORM,
                         kIntraOpThreadPool);

}  // namespace

// __xla_test$$ConcatVectors

static absl::Status Concat3Vectors(ffi::BufferR2<F32> vec_1,
                                   ffi::BufferR2<F32> vec_2,
                                   ffi::BufferR2<F32> vec_3,
                                   ResultBuffer<F32, 2> out) {
  if (out->dimensions().back() != 3) {
    return absl::FailedPreconditionError("output dimension 0 expected to be 3");
  }
  float* out_data = out->typed_data();

  ffi::BufferR2<F32>* vecs[3] = {&vec_1, &vec_2, &vec_3};
  for (int elem_idx = 0; elem_idx < out->dimensions().front(); ++elem_idx) {
    for (int vec_idx = 0; vec_idx < 3; ++vec_idx) {
      // {{vec_0[0], vec_1[0], vec_2[0]},
      //  {vec_0[1], vec_1[1], vec_2[1]},
      //  ...}
      const auto out_idx = elem_idx * out->dimensions().back() + vec_idx;
      out_data[out_idx] = vecs[vec_idx]->typed_data()[elem_idx];
    }
  }
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kConcat3Vectors, Concat3Vectors,
                       ffi::Ffi::Bind()
                           .Arg<ffi::BufferR2<F32>>()
                           .Arg<ffi::BufferR2<F32>>()
                           .Arg<ffi::BufferR2<F32>>()
                           .Ret<ffi::BufferR2<F32>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$Concat3Vectors",
                         PLATFORM, kConcat3Vectors);

using FfiCustomCallTest = CustomCallTest;

TEST_F(FfiCustomCallTest, FfiReportsSuccess) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r0f32_, {}, "__xla_test$$always_succeed", "",
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  auto status = Execute(std::move(module), {}).status();
  EXPECT_EQ(status, absl::OkStatus());
}

TEST_F(FfiCustomCallTest, Tokens) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  std::vector<Shape> ret = {ShapeUtil::MakeShape(F32, {}),
                            ShapeUtil::MakeTokenShape()};

  auto* token = builder.AddInstruction(HloInstruction::CreateToken());
  builder.AddInstruction(HloInstruction::CreateCustomCall(
      ShapeUtil::MakeTupleShape(ret), {token}, "__xla_test$$tokens", "",
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  TF_EXPECT_OK(Execute(std::move(module), {}).status());
}

TEST_F(FfiCustomCallTest, FfiUnknownTarget) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r0f32_, {}, "__xla_test$$unknown_target", "",
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  auto status = Execute(std::move(module), {}).status();
  EXPECT_THAT(status.message(), HasSubstr("No FFI handler registered for"));
}

TEST_F(FfiCustomCallTest, FfiReportsFailure) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r0f32_, {}, "__xla_test$$always_fail",
      /*opaque=*/"{value = 42 : i32}",
      CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  auto status = Execute(std::move(module), {}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(), ::testing::HasSubstr("Failed: 42"));
}

TEST_F(FfiCustomCallTest, FfiReportsOneOfFailures) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto res_1 = builder.AddInstruction(HloInstruction::CreateCustomCall(
      r0f32_, {}, "__xla_test$$always_fail",
      /*opaque=*/"{value = 1 : i32}",
      CustomCallApiVersion::API_VERSION_TYPED_FFI));
  auto res_2 = builder.AddInstruction(HloInstruction::CreateCustomCall(
      r0f32_, {}, "__xla_test$$always_fail",
      /*opaque=*/"{value = 2 : i32}",
      CustomCallApiVersion::API_VERSION_TYPED_FFI));
  builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32_, HloOpcode::kAdd, res_1, res_2));

  module->AddEntryComputation(builder.Build());

  // Execution order is undefined because custom calls do not have data
  // dependency, and we check that the error message contains one of the two
  // error codes.
  auto status = Execute(std::move(module), {}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(), HasSubstr("Failed:"));
}

TEST_F(FfiCustomCallTest, FfiTransitiveCustomCallReportsOneOfFailures) {
  const char* const kModuleStr = R"(
    HloModule m
    sub_2 {
      ROOT custom-call = f32[] custom-call(), custom_call_target="__xla_test$$always_fail", api_version=API_VERSION_TYPED_FFI, backend_config="{value = 2 : i32}"
    }
    sub_3 {
      ROOT custom-call = f32[] custom-call(), custom_call_target="__xla_test$$always_fail", api_version=API_VERSION_TYPED_FFI, backend_config="{value = 3 : i32}"
    }
    ENTRY test {
      call0 = f32[] call(), to_apply=sub_2
      call1 = f32[] call(), to_apply=sub_3
      ROOT sum = f32[] add(%call0, %call1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  // Execution order is undefined because custom calls do not have data
  // dependency, and we check that the error message contains one of the two
  // error codes.
  auto status = Execute(std::move(module), {}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(), HasSubstr("Failed:"));
}

TEST_F(FfiCustomCallTest, FfiWrongNumberOfArguments) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r0f32_, {}, "__xla_test$$FfiR0F32Add2", "",
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  auto status = Execute(std::move(module), {}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("Wrong number of arguments"));
}

TEST_F(FfiCustomCallTest, FfiWrongRankOfArgument) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  Array2D<float> array(2, 2);
  array(0, 0) = 1.0f;
  array(0, 1) = 2.0f;
  array(1, 0) = 3.0f;
  array(1, 1) = 4.0f;

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2FromArray2D(array)));
  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r2f32_, {constant}, "__xla_test$$FfiR0F32Add2", "",
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  auto status = Execute(std::move(module), {}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("Wrong buffer rank"));
}

TEST_F(FfiCustomCallTest, FfiWrongDTypeOfArgument) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int>(42)));
  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r2f32_, {constant}, "__xla_test$$FfiR0F32Add2", "",
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  auto status = Execute(std::move(module), {}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("Wrong buffer dtype"));
}

TEST_F(FfiCustomCallTest, FfiHandleTypedBuffers) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r0f32_, {constant}, "__xla_test$$FfiR0F32Add2", "",
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {}));
  LiteralTestUtil::ExpectR0Near<float>(44.0f, result, kDefaultErrorSpec);
}

TEST_F(FfiCustomCallTest, FfiHandleInputAsParameters) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto constant =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, "p"));
  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r0f32_, {constant}, "__xla_test$$FfiR0F32Add2", "",
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  Literal argument = LiteralUtil::CreateR0<float>(42.0f);

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {&argument}));
  LiteralTestUtil::ExpectR0Near<float>(44.0f, result, kDefaultErrorSpec);
}

TEST_F(FfiCustomCallTest, FfiHandleBufferBaseFloat) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r0f32_, {constant}, "__xla_test$$FfiR0FAdd2BufferBase", "",
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {}));
  LiteralTestUtil::ExpectR0Near<float>(44.0f, result, kDefaultErrorSpec);
}

TEST_F(FfiCustomCallTest, FfiHandleBufferBaseDouble) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<double>(42.0f)));
  builder.AddInstruction(HloInstruction::CreateCustomCall(
      ShapeUtil::MakeShape(F64, {}), {constant},
      "__xla_test$$FfiR0FAdd2BufferBase", "",
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {}));
  LiteralTestUtil::ExpectR0Near<double>(44.0f, result, kDefaultErrorSpec);
}

TEST_F(FfiCustomCallTest, FfiHandleAttr) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r0f32_, {constant}, "__xla_test$$FfiR0F32AddN",
      /*opaque=*/"{n = 3.0 : f32}",
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {}));
  LiteralTestUtil::ExpectR0Near<float>(45.0f, result, kDefaultErrorSpec);
}

TEST_F(FfiCustomCallTest, FfiHandleAttrPointer) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  auto n = 4.0f;
  auto ptr = reinterpret_cast<uintptr_t>(&n);
  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r0f32_, {constant}, "__xla_test$$FfiR0F32AddNPointer",
      /*opaque=*/absl::StrFormat("{n = %d : i64}", ptr),
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {}));
  LiteralTestUtil::ExpectR0Near<float>(46.0f, result, kDefaultErrorSpec);
}

TEST_F(FfiCustomCallTest, FfiHandleR2Vector) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  Array2D<float> array(2, 2);
  array(0, 0) = 1.0f;
  array(0, 1) = 2.0f;
  array(1, 0) = 3.0f;
  array(1, 1) = 4.0f;

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2FromArray2D(array)));
  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r0f32_, {constant}, "__xla_test$$FfiF32ReduceSum",
      /*opaque=*/"",
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {}));
  LiteralTestUtil::ExpectR0Near<float>(10.0f, result, kDefaultErrorSpec);
}

TEST_F(FfiCustomCallTest, FfiWrongEnumType) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  Array2D<float> array(2, 2);
  array(0, 0) = 1.0f;
  array(0, 1) = 2.0f;
  array(1, 0) = 3.0f;
  array(1, 1) = 4.0f;

  auto input = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2FromArray2D(array)));

  auto op = BinaryOp::kAdd;
  auto init_method = InitMethod::kZero;

  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r0f32_, {input}, "__xla_test$$FfiF32Accumulate",
      /*opaque=*/
      absl::StrFormat("{binary_op = %d : i16, init = %d : i16}", op,
                      init_method),
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  auto status = Execute(std::move(module), {}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_THAT(status.message(), HasSubstr("Wrong scalar data type"));
}

class FfiCustomCallEnumTest
    : public FfiCustomCallTest,
      public ::testing::WithParamInterface<std::tuple<BinaryOp, InitMethod>> {};

XLA_TEST_P(FfiCustomCallEnumTest, FfiHandleEnumAttr) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  Array2D<float> array(2, 2);
  array(0, 0) = 1.0f;
  array(0, 1) = 2.0f;
  array(1, 0) = 3.0f;
  array(1, 1) = 4.0f;

  auto input = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2FromArray2D(array)));

  auto op = std::get<0>(GetParam());
  auto init_method = std::get<1>(GetParam());

  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r0f32_, {input}, "__xla_test$$FfiF32Accumulate",
      /*opaque=*/
      absl::StrFormat("{binary_op = %d : i8, init = %d : i32}", op,
                      init_method),
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {}));

  // Init method is an artificial enum to demonstrate handling enums with
  // different underlying types. Normally it would be just a float scalar.
  float expected = (init_method == InitMethod::kZero) ? 0.0f : 1.0f;
  switch (op) {
    case BinaryOp::kAdd:
      expected += 10.0f;  // Sum of input array elements
      break;
    case BinaryOp::kMul:
      expected *= 24.0f;  // Product of input array elements
      break;
  }

  LiteralTestUtil::ExpectR0Near<float>(expected, result, kDefaultErrorSpec);
}

INSTANTIATE_TEST_SUITE_P(
    FfiEnum, FfiCustomCallEnumTest,
    ::testing::Combine(::testing::Values(BinaryOp::kAdd, BinaryOp::kMul),
                       ::testing::Values(InitMethod::kZero, InitMethod::kOne)));

TEST_F(FfiCustomCallTest, FfiUsedInOtherComputations) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto input = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2FromArray2D(
          Array2D<float>{{1.0f, 2.0f}, {3.0f, 4.0f}})));
  auto incremented = builder.AddInstruction(HloInstruction::CreateCustomCall(
      ShapeUtil::MakeShape(F32, {1, 2, 2}), {input},
      "__xla_test$$FfiF32Add1ToValues",
      /*opaque=*/"",
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));
  auto incremented_again =
      builder.AddInstruction(HloInstruction::CreateCustomCall(
          ShapeUtil::MakeShape(F32, {1, 2, 2}), {incremented},
          "__xla_test$$FfiF32Add1ToValues",
          /*opaque=*/"",
          /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));

  // Concatenate the values along first dim.
  builder.AddInstruction(
      HloInstruction::CreateConcatenate(ShapeUtil::MakeShape(F32, {2, 2, 2}),
                                        {incremented, incremented_again}, 0));

  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {}));
  LiteralTestUtil::ExpectR3EqualArray3D<float>(
      Array3D<float>{{{2, 3}, {4, 5}}, {{3, 4}, {5, 6}}}, result);
}

TEST_F(FfiCustomCallTest, FfiInputAndOutputLayoutDiffer) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto input =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r2f32_, "p"));

  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r2f32_, {input}, "__xla_test$$FfiF32Add1ToValues", /*opaque=*/"",
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());
  ForceParameterLayout(module.get(), 0, LayoutUtil::MakeLayout({1, 0}));
  ForceResultLayout(module.get(), LayoutUtil::MakeLayout({0, 1}));

  Literal argument = LiteralUtil::CreateR2<float>({{1.f, 2.f}, {3.f, 4.f}});

  // Note, the expected result is transposed! This is because the input and
  // output layouts of the custom call differ and the called function just
  // blindly adds one to each element.
  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {&argument}));
  LiteralTestUtil::ExpectR2Equal<float>({{2.f, 4.f}, {3.f, 5.f}}, result);
}

TEST_F(FfiCustomCallTest, FfiLayoutConstrained) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  // The argument and result of the computation are set to different layouts,
  // but the custom call is layout constrained to a fixed operand and result
  // layout, so the correct result should be produced.
  auto input =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r2f32_, "p"));

  const Shape& r2f32_dim0_major =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 2}, {1, 0});
  auto custom_call = builder.AddInstruction(HloInstruction::CreateCustomCall(
      r2f32_dim0_major, {input}, "__xla_test$$FfiF32Add1ToValues",
      /*operand_shapes_with_layout=*/{r2f32_dim0_major},
      /*opaque=*/"",
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));
  builder.AddInstruction(
      custom_call->CloneWithNewOperands(r2f32_dim0_major, {custom_call}));

  module->AddEntryComputation(builder.Build());
  ForceParameterLayout(module.get(), 0, LayoutUtil::MakeLayout({1, 0}));
  ForceResultLayout(module.get(), LayoutUtil::MakeLayout({0, 1}));

  Literal argument = LiteralUtil::CreateR2<float>({{1.f, 2.f}, {3.f, 4.f}});

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {&argument}));
  LiteralTestUtil::ExpectR2Equal<float>({{3.f, 4.f}, {5.f, 6.f}}, result);
}

TEST_F(FfiCustomCallTest, FfiTupleOutput) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto input0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, r0f32_, "p0"));
  auto input1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, r0f32_, "p1"));
  builder.AddInstruction(HloInstruction::CreateCustomCall(
      ShapeUtil::MakeTupleShape({r0f32_, r0f32_}), {input0, input1},
      "__xla_test$$FfiF32TupleSwap", /*opaque=*/"",
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  Literal arg0 = LiteralUtil::CreateR0<float>(7.f);
  Literal arg1 = LiteralUtil::CreateR0<float>(42.f);

  Literal expected = LiteralUtil::MakeTuple({&arg1, &arg0});
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          Execute(std::move(module), {&arg0, &arg1}));
  EXPECT_EQ(result, expected);
}

TEST_F(FfiCustomCallTest, FfiNestedTupleOutput) {
  const char* const kModuleStr = R"(
    HloModule m

    ENTRY test {
      c0 = f32[] constant(7.0)
      c1 = f32[] constant(42.0)
      c2 = f32[] constant(8.0)
      c3 = f32[] constant(43.0)
      custom-call = ((f32[], f32[]), (f32[], f32[])) custom-call(c0, c1, c2, c3), custom_call_target="__xla_test$$FfiTupleRotate", api_version=API_VERSION_TYPED_FFI
      t0x = (f32[], f32[]) get-tuple-element(custom-call), index=0
      t00 = f32[] get-tuple-element(t0x), index=0
      t01 = f32[] get-tuple-element(t0x), index=1
      t1x = (f32[], f32[]) get-tuple-element(custom-call), index=1
      t10 = f32[] get-tuple-element(t1x), index=0
      t11 = f32[] get-tuple-element(t1x), index=1
      ROOT tuple = (f32[], f32[], f32[], f32[]) tuple(t00, t01, t10, t11)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  const Literal arg0 = LiteralUtil::CreateR0<float>(7.f);
  const Literal arg1 = LiteralUtil::CreateR0<float>(42.f);
  const Literal arg2 = LiteralUtil::CreateR0<float>(8.f);
  const Literal arg3 = LiteralUtil::CreateR0<float>(43.f);

  const Literal expected = LiteralUtil::MakeTuple({&arg1, &arg2, &arg3, &arg0});
  TF_ASSERT_OK_AND_ASSIGN(const Literal result, Execute(std::move(module), {}));
  EXPECT_EQ(result, expected);
}

TEST_F(FfiCustomCallTest, FfiTupleInput) {
  const char* const kModuleStr = R"(
    HloModule m

    ENTRY test {
      c0 = (f32[], f32[]) constant((7.0, 42.0))
      ROOT custom-call = (f32[], f32[]) custom-call(c0), custom_call_target="__xla_test$$FfiF32TupleSwap", api_version=API_VERSION_TYPED_FFI
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  Literal arg0 = LiteralUtil::CreateR0<float>(7.f);
  Literal arg1 = LiteralUtil::CreateR0<float>(42.f);

  Literal expected = LiteralUtil::MakeTuple({&arg1, &arg0});
  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {}));
  EXPECT_EQ(result, expected);
}

TEST_F(FfiCustomCallTest, FfiNestedTupleInput) {
  const char* const kModuleStr = R"(
    HloModule m

    ENTRY test {
      c0 = ((f32[], f32[]), (f32[], f32[])) constant(((7.0, 42.0), (8.0, 43.0)))
      ROOT custom-call = (f32[], f32[], f32[], f32[]) custom-call(c0), custom_call_target="__xla_test$$FfiTupleRotate", api_version=API_VERSION_TYPED_FFI
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  Literal arg0 = LiteralUtil::CreateR0<float>(7.f);
  Literal arg1 = LiteralUtil::CreateR0<float>(42.f);
  Literal arg2 = LiteralUtil::CreateR0<float>(8.f);
  Literal arg3 = LiteralUtil::CreateR0<float>(43.f);

  Literal expected = LiteralUtil::MakeTuple({&arg1, &arg2, &arg3, &arg0});
  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {}));
  EXPECT_EQ(result, expected);
}

TEST_F(FfiCustomCallTest, SwapTupleAnyBuffersToS16U32) {
  const char* const kModuleStr = R"(
    HloModule m

    ENTRY test {
      p0 = (u32[], s16[]) parameter(0)
      ROOT custom-call = (s16[], u32[]) custom-call(p0), custom_call_target="__xla_test$$SwapTupleAnyBuffersToS16U32", api_version=API_VERSION_TYPED_FFI
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  Literal arg0 = LiteralUtil::CreateR0<uint32_t>(0xDEADC0DE);
  Literal arg1 = LiteralUtil::CreateR0<int16_t>(29);
  Literal argument = LiteralUtil::MakeTuple({&arg0, &arg1});
  Literal expected = LiteralUtil::MakeTuple({&arg1, &arg0});

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {&argument}));
  EXPECT_EQ(result, expected);
}

TEST_F(FfiCustomCallTest, IgnoresEmptyTupleParameter) {
  const char* const kModuleStr = R"(
    HloModule m

    ENTRY test {
      t0 = u32[] parameter(0)
      t1 = s16[] parameter(1)
      t2 = () tuple()
      t3 = ((), ()) tuple(t2, t2)
      p0 = (u32[], s16[], ((), ())) tuple(t0, t1, t3)
      ROOT custom-call = (s16[], u32[]) custom-call(p0), custom_call_target="__xla_test$$SwapTupleAnyBuffersToS16U32", api_version=API_VERSION_TYPED_FFI
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  Literal arg0 = LiteralUtil::CreateR0<uint32_t>(0xDEADC0DE);
  Literal arg1 = LiteralUtil::CreateR0<int16_t>(29);
  const Literal expected = LiteralUtil::MakeTuple({&arg1, &arg0});

  TF_ASSERT_OK_AND_ASSIGN(const Literal result,
                          Execute(std::move(module), {&arg0, &arg1}));
  EXPECT_EQ(result, expected);
}

TEST_F(FfiCustomCallTest, SwapTupleU32S16ToS16U32) {
  const char* const kModuleStr = R"(
    HloModule m

    ENTRY test {
      p0 = (u32[], s16[]) parameter(0)
      ROOT custom-call = (s16[], u32[]) custom-call(p0), custom_call_target="__xla_test$$SwapTupleU32S16ToS16U32", api_version=API_VERSION_TYPED_FFI
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  Literal arg0 = LiteralUtil::CreateR0<uint32_t>(0xDEADC0DE);
  Literal arg1 = LiteralUtil::CreateR0<int16_t>(29);
  Literal argument = LiteralUtil::MakeTuple({&arg0, &arg1});
  Literal expected = LiteralUtil::MakeTuple({&arg1, &arg0});

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {&argument}));
  EXPECT_EQ(result, expected);
}

TEST_F(FfiCustomCallTest, HandleR2Tuple) {
  const char* const kModuleStr = R"(
    HloModule m

    ENTRY test {
      p0 = (f32[2, 1], f32[2, 1], f32[2, 1]) parameter(0)
      ROOT custom-call = f32[2, 3] custom-call(p0), custom_call_target="__xla_test$$Concat3Vectors", api_version=API_VERSION_TYPED_FFI
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  Literal arg_0 = LiteralUtil::CreateR2<float>({{1.f}, {2.f}});
  Literal arg_1 = LiteralUtil::CreateR2<float>({{3.f}, {4.f}});
  Literal arg_2 = LiteralUtil::CreateR2<float>({{5.f}, {6.f}});
  Literal tuple_arg = LiteralUtil::MakeTuple({&arg_0, &arg_1, &arg_2});

  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          Execute(std::move(module), {&tuple_arg}));

  LiteralTestUtil::ExpectR2Equal<float>({{1.f, 3.f, 5.f},   //
                                         {2.f, 4.f, 6.f}},  //
                                        result);
}

TEST_F(FfiCustomCallTest, HandleTupleDifferentRanks) {
  const char* const kModuleStr = R"(
    HloModule m

    ENTRY test {
      t00 = u32[] parameter(0)
      t01 = s16[5] parameter(1)
      t0x = (u32[], s16[5]) tuple(t00, t01)
      t10 = f32[2, 2] parameter(2)
      t11 = f32[4, 2, 2] parameter(3)
      t1x = (f32[2, 2], f32[4, 2, 2]) tuple(t10, t11)
      p0 = ((u32[], s16[5]), (f32[2, 2], f32[4, 2, 2])) tuple(t0x, t1x)
      ROOT custom-call = (s32[5], f32[5, 2, 2]) custom-call(p0), custom_call_target="__xla_test$$HandleTupleDifferentRanks", api_version=API_VERSION_TYPED_FFI
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  Literal arg_0 = LiteralUtil::CreateR0<uint32_t>(100);
  Literal arg_1 = LiteralUtil::CreateR1<int16_t>({29, 30, 31, 32, 33});
  Literal arg_2 = LiteralUtil::CreateR2<float>({{17.f, 18.f}, {19.f, 20.f}});
  Literal arg_3 = LiteralUtil::CreateR3<float>({{{1.f, 2.f}, {3.f, 4.f}},
                                                {{5.f, 6.f}, {7.f, 8.f}},
                                                {{9.f, 10.f}, {11.f, 12.f}},
                                                {{13.f, 14.f}, {15.f, 16.f}}});

  TF_ASSERT_OK_AND_ASSIGN(
      const Literal result,
      Execute(std::move(module), {&arg_0, &arg_1, &arg_2, &arg_3}));

  Literal expected_0 =
      LiteralUtil::CreateR1<int32_t>({2900, 3000, 3100, 3200, 3300});
  Literal expected_1 =
      LiteralUtil::CreateR3<float>({{{1.f, 2.f}, {3.f, 4.f}},
                                    {{5.f, 6.f}, {7.f, 8.f}},
                                    {{9.f, 10.f}, {11.f, 12.f}},
                                    {{13.f, 14.f}, {15.f, 16.f}},
                                    {{17.f, 18.f}, {19.f, 20.f}}});

  const Literal expected_tuple =
      LiteralUtil::MakeTuple({&expected_0, &expected_1});
  EXPECT_EQ(result, expected_tuple);
}

TEST_F(FfiCustomCallTest, FfiNestedTupleInputAndOutput) {
  const char* const kModuleStr = R"(
    HloModule m

    ENTRY test {
      c0 = ((f32[], f32[]), (f32[], f32[])) constant(((7.0, 42.0), (8.0, 43.0)))
      custom-call = (f32[], (f32[], f32[]), f32[]) custom-call(c0), custom_call_target="__xla_test$$FfiTupleRotate", api_version=API_VERSION_TYPED_FFI
      t00 = f32[] get-tuple-element(custom-call), index=0
      t1x = (f32[], f32[]) get-tuple-element(custom-call), index=1
      t10 = f32[] get-tuple-element(t1x), index=0
      t11 = f32[] get-tuple-element(t1x), index=1
      t20 = f32[] get-tuple-element(custom-call), index=2
      ROOT result = (f32[], f32[], f32[], f32[]) tuple(t00, t10, t11, t20)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  Literal arg0 = LiteralUtil::CreateR0<float>(7.f);
  Literal arg1 = LiteralUtil::CreateR0<float>(42.f);
  Literal arg2 = LiteralUtil::CreateR0<float>(8.f);
  Literal arg3 = LiteralUtil::CreateR0<float>(43.f);

  const Literal expected = LiteralUtil::MakeTuple({&arg1, &arg2, &arg3, &arg0});
  TF_ASSERT_OK_AND_ASSIGN(const Literal result, Execute(std::move(module), {}));
  EXPECT_EQ(result, expected);
}

TEST_F(FfiCustomCallTest, IntraOpThreadPool) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r0f32_, {}, "__xla_test$$intra_op_thread_pool", "",
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  auto status = Execute(std::move(module), {}).status();
  EXPECT_EQ(status, absl::OkStatus());
}

//===----------------------------------------------------------------------===//
// Stateful XLA:FFI handler
//===----------------------------------------------------------------------===//

struct SomeState {
  explicit SomeState(float value) : value(value) {}
  float value = 0;
};

int instantiate_called_counter = 0;

// Every time custom call HLO operation is instantiated as a CPU runtime Thunk,
// XLA calls instantiate callback to create a new instance of the handler state,
// that will be passed to all other FFI handler calls.
static absl::StatusOr<std::unique_ptr<SomeState>> InstantiateState() {
  ++instantiate_called_counter;
  return std::make_unique<SomeState>(42.f);
}

// At run time we can access the state created by the instantiate callback.
static absl::Status IncrementState(R0F32ResultBuffer out, SomeState* state) {
  state->value += 1.f;
  auto out_data = out->typed_data();
  *out_data = state->value;
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kInstantiateState, InstantiateState,
                       ffi::Ffi::BindInstantiate());

XLA_FFI_DEFINE_HANDLER(
    kIncrementState, IncrementState,
    ffi::Ffi::Bind().Ret<R0F32Buffer>().Ctx<ffi::State<SomeState>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$ffi_execution_state",
                         PLATFORM,
                         {
                             /*instantiate=*/kInstantiateState,
                             /*prepare=*/nullptr,
                             /*initialize=*/nullptr,
                             /*execute=*/kIncrementState,
                         });

float last_value = 0.f;

// Similar to InstantiateState above, but takes initial value as an attribute.
static absl::StatusOr<std::unique_ptr<SomeState>> InstantiateStateWithAttribute(
    float initial_value) {
  last_value = initial_value;
  return std::make_unique<SomeState>(initial_value);
}

// Similar to IncrementState above, but with attributes. No attribute is used
// here, but still their type and number must match the instantiate callback.
static absl::Status IncrementStateWithAttribute(
    R0F32ResultBuffer out, SomeState* state,
    [[maybe_unused]] float initial_value) {
  return IncrementState(out, state);
}

XLA_FFI_DEFINE_HANDLER(
    kInstantiateStateWithAttribute, InstantiateStateWithAttribute,
    ffi::Ffi::BindInstantiate().Attr<float>("initial_value"));

XLA_FFI_DEFINE_HANDLER(kIncrementStateWithAttribute,
                       IncrementStateWithAttribute,
                       ffi::Ffi::Bind()
                           .Ret<R0F32Buffer>()
                           .Ctx<ffi::State<SomeState>>()
                           .Attr<float>("initial_value"));

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
                         "__xla_test$$ffi_execution_state_with_attrs", "Host",
                         {
                             /*instantiate=*/kInstantiateStateWithAttribute,
                             /*prepare=*/nullptr,
                             /*initialize=*/nullptr,
                             /*execute=*/kIncrementStateWithAttribute,
                         });

// This test doesn't care about execution results, its intent is just to test if
// instantiate function was called.
TEST_F(CustomCallTest, FfiExecutionStateInstantiate) {
  const char* const kModuleStr = R"(
    HloModule m
    ENTRY test {
      ROOT result = f32[] custom-call(), custom_call_target=
        "__xla_test$$ffi_execution_state", api_version=API_VERSION_TYPED_FFI
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  // Execute the module, but don't verify the results.
  instantiate_called_counter = 0;
  auto result = Execute(std::move(module), {});

  // Check that instantiate callback was called. Even though we don't care about
  // the result in this test, log it in case of failure to help debugging.
  EXPECT_EQ(instantiate_called_counter, 1) << result.status();
}

TEST_F(CustomCallTest, FfiExecutionStateExecute) {
  // Module that calls custom call in a loop two times.
  const char* const kModuleStr = R"(
    HloModule m
    lt2 (arg: (s32[], f32[])) -> pred[] {
      arg = (s32[], f32[]) parameter(0)
      i =  s32[] get-tuple-element(arg), index=0
      two = s32[] constant(2)
      ROOT result = pred[] compare(i, two), direction=LT
    }

    incr_i_and_call_custom_call (arg: (s32[], f32[])) -> (s32[], f32[]) {
      arg = (s32[], f32[]) parameter(0)
      i =  s32[] get-tuple-element(arg), index=0
      one = s32[] constant(1)
      i_incr = s32[] add(i, one)
      custom_call = f32[] custom-call(), custom_call_target=
        "__xla_test$$ffi_execution_state", api_version=API_VERSION_TYPED_FFI
      ROOT result = (s32[], f32[]) tuple(i_incr, custom_call)
    }

    ENTRY test {
      i = s32[] constant(0)
      placeholder = f32[] constant(0.0)
      tuple = (s32[], f32[]) tuple(i, placeholder)
      while = (s32[], f32[]) while(tuple), body=incr_i_and_call_custom_call,
        condition=lt2
      ROOT result = f32[] get-tuple-element(while), index=1
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  // Custom call called twice, starting value is hardcoded in the instantiate
  // callback as 42.0, so we expect 44.0 as a result.
  Literal expected = LiteralUtil::CreateR0<float>(44.f);

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {}));
  EXPECT_EQ(result, expected);
}

// Similarly to FfiExecutionStateInstantiate, this test doesn't care about
// execution results, its intent is just to test if instantiate function was
// called (with correct attributes).
TEST_F(CustomCallTest, FfiExecutionStateInstantiateWithAttribute) {
  const char* const kModuleStr = R"(
    HloModule m
    ENTRY test {
      ROOT result = f32[] custom-call(), custom_call_target=
        "__xla_test$$ffi_execution_state_with_attrs",
        api_version=API_VERSION_TYPED_FFI,
        backend_config="{initial_value = 43.0 : f32}"
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  // Execute the module, but don't verify the results.
  last_value = 0;
  auto result = Execute(std::move(module), {});

  // Check that the correct instantiate callback was called. Even though we
  // don't care about the result in this test, log it in case of failure to help
  // debugging.
  EXPECT_EQ(last_value, 43.f) << result.status();
}

TEST_F(CustomCallTest, FfiExecutionStateExecuteWithAttribute) {
  // Module that calls custom call in a loop three times, with initial value set
  // to 43.0.
  const char* const kModuleStr = R"(
    HloModule m
    lt3 (arg: (s32[], f32[])) -> pred[] {
      arg = (s32[], f32[]) parameter(0)
      i =  s32[] get-tuple-element(arg), index=0
      three = s32[] constant(3)
      ROOT result = pred[] compare(i, three), direction=LT
    }

    incr_i_and_call_custom_call (arg: (s32[], f32[])) -> (s32[], f32[]) {
      arg = (s32[], f32[]) parameter(0)
      i =  s32[] get-tuple-element(arg), index=0
      one = s32[] constant(1)
      i_incr = s32[] add(i, one)
      custom_call = f32[] custom-call(), custom_call_target=
        "__xla_test$$ffi_execution_state_with_attrs",
        api_version=API_VERSION_TYPED_FFI,
        backend_config="{initial_value = 43.0 : f32}"
      ROOT result = (s32[], f32[]) tuple(i_incr, custom_call)
    }

    ENTRY test {
      i = s32[] constant(0)
      placeholder = f32[] constant(0.0)
      tuple = (s32[], f32[]) tuple(i, placeholder)
      while = (s32[], f32[]) while(tuple), body=incr_i_and_call_custom_call,
        condition=lt3
      ROOT result = f32[] get-tuple-element(while), index=1
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  // Custom call called three times, with initial value set to 43.0. So we
  // expect 46.0 as a result.
  Literal expected = LiteralUtil::CreateR0<float>(46.f);

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {}));
  EXPECT_EQ(result, expected);
}

//===----------------------------------------------------------------------===//
// XLA:FFI handler with execution context
//===----------------------------------------------------------------------===//

// Arbitrary user-defined context passed via the execution context side channel
// to a custom call handlers.
struct SomeExtraContext {
  explicit SomeExtraContext(int32_t value) : value(value) {}
  int32_t value;
  bool executed = false;
};

template <ffi::ExecutionStage stage>
static absl::Status ExecutionContext(ffi::Result<ffi::AnyBuffer>,
                                     SomeExtraContext* ctx) {
  if (ctx->value != 42) return absl::InternalError("Unexpected value");
  if constexpr (stage == ffi::ExecutionStage::kExecute) {
    ctx->executed = true;
  }

  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kExecutionContextExecute,
                       ExecutionContext<ffi::ExecutionStage::kExecute>,
                       ffi::Ffi::Bind<ffi::ExecutionStage::kExecute>()
                           .Ret<ffi::AnyBuffer>()
                           .Ctx<ffi::UserData<SomeExtraContext>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla.cpu.ffi_execution_context",
                         PLATFORM,
                         {
                             /*instantiate=*/nullptr,
                             /*prepare=*/nullptr,
                             /*initialize=*/nullptr,
                             /*execute=*/kExecutionContextExecute,
                         });

static absl::StatusOr<LocalClient*> CreateClient() {
  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetPlatform(PLATFORM));
  LocalClientOptions client_options(platform, 1, 1, std::nullopt);
  return xla::ClientLibrary::GetOrCreateLocalClient(client_options);
}

TEST_F(CustomCallClientAPITest, FfiExecutionContext) {
  XlaBuilder b(TestName());
  const Shape shape = ShapeUtil::MakeShape(F32, {});
  CustomCall(&b, "xla.cpu.ffi_execution_context", /*operands=*/{}, shape,
             /*opaque=*/"",
             /*has_side_effect=*/false,
             /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
             /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
             /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);

  TF_ASSERT_OK_AND_ASSIGN(auto local_client, CreateClient());
  EXPECT_NE(local_client->device_count(), 0);

  TF_ASSERT_OK_AND_ASSIGN(auto computation, b.Build());
  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      local_client->Compile(computation, /*argument_layouts=*/{},
                            /*options=*/{}));

  ffi::ExecutionContext execution_context;
  TF_ASSERT_OK(execution_context.Emplace<SomeExtraContext>(42));

  ExecutableRunOptions run_options;
  run_options.set_allocator(local_client->backend().memory_allocator());
  run_options.set_ffi_execution_context(&execution_context);

  std::vector<const xla::ShapedBuffer*> args;
  TF_ASSERT_OK_AND_ASSIGN(auto result, executable[0]->Run(args, run_options));
  TF_ASSERT_OK_AND_ASSIGN(auto* user_context,
                          execution_context.Lookup<SomeExtraContext>());
  EXPECT_TRUE(user_context->executed);
}

}  // namespace
}  // namespace xla
