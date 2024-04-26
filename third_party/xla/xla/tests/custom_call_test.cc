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
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/base/dynamic_annotations.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/shape_util.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_macros.h"
#include "xla/tests/test_utils.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace {
void R0F32Add2(float* out, float** in) {
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(in, sizeof(float*));
  *out = **in + 2.0f;
}

void R0F32Add2InPlace(float* out, float** in) {
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(in, sizeof(float*));
  **in = **in + 2.0f;
}

void R2F32ReduceSum(float* out, float** in) {
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(in, sizeof(float) * 4);
  float* array = in[0];
  *out = array[0] + array[1] + array[2] + array[3];
}

void Add1ToValues(float* out, float** in) {
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(in, sizeof(float) * 4);
  float* array = in[0];
  out[0] = array[0] + 1;
  out[1] = array[1] + 1;
  out[2] = array[2] + 1;
  out[3] = array[3] + 1;
}

void F32TupleSwap(float** out, float** in) {
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(in[0], sizeof(float));
  ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(in[1], sizeof(float));
  *out[0] = *in[1];
  *out[1] = *in[0];
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

}  // namespace

XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(R0F32Add2);
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(R0F32Add2InPlace);
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(R2F32ReduceSum);
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(Add1ToValues);
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(F32TupleSwap);
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(R0F32Add2Succeed);
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(CustomCallFail);
XLA_CPU_REGISTER_CUSTOM_CALL_TARGET(CustomCallFailWithBackendConfigStr);

namespace xla {
namespace {

using ::testing::HasSubstr;

class CustomCallTest : public HloTestBase {
 protected:
  Shape r0f32_ = ShapeUtil::MakeShape(F32, {});
  Shape r2f32_ = ShapeUtil::MakeShape(F32, {2, 2});
};

XLA_TEST_F(CustomCallTest, CustomCallR0F32Add2) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  builder.AddInstruction(
      HloInstruction::CreateCustomCall(r0f32_, {constant}, "R0F32Add2"));

  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {}));
  LiteralTestUtil::ExpectR0Near<float>(44.0f, result, error_spec_);
}

XLA_TEST_F(CustomCallTest, CustomCallR0F32Add2Aliased) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));

  builder
      .AddInstruction(HloInstruction::CreateCustomCall(r0f32_, {constant},
                                                       "R0F32Add2InPlace"))
      ->set_output_to_operand_aliasing({{{}, {0, {}}}});

  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {}));
  LiteralTestUtil::ExpectR0Near<float>(44.0f, result, error_spec_);
}

XLA_TEST_F(CustomCallTest, CustomCallR2F32Reduce) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  Array2D<float> array(2, 2);
  array(0, 0) = 1.0f;
  array(0, 1) = 2.0f;
  array(1, 0) = 3.0f;
  array(1, 1) = 4.0f;

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2FromArray2D(array)));
  builder.AddInstruction(
      HloInstruction::CreateCustomCall(r0f32_, {constant}, "R2F32ReduceSum"));

  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {}));
  LiteralTestUtil::ExpectR0Near<float>(10.0f, result, error_spec_);
}

XLA_TEST_F(CustomCallTest, UsedInOtherComputations) {
  auto module = CreateNewVerifiedModule();
  auto b = HloComputation::Builder(TestName());

  auto input = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR2FromArray2D(
          Array2D<float>{{1.0f, 2.0f}, {3.0f, 4.0f}})));
  auto incremented = b.AddInstruction(HloInstruction::CreateCustomCall(
      ShapeUtil::MakeShape(F32, {1, 2, 2}), {input}, "Add1ToValues"));
  auto incremented_again = b.AddInstruction(HloInstruction::CreateCustomCall(
      ShapeUtil::MakeShape(F32, {1, 2, 2}), {incremented}, "Add1ToValues"));

  // Concatenate the values along first dim.
  b.AddInstruction(
      HloInstruction::CreateConcatenate(ShapeUtil::MakeShape(F32, {2, 2, 2}),
                                        {incremented, incremented_again}, 0));

  module->AddEntryComputation(b.Build());

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {}));
  LiteralTestUtil::ExpectR3EqualArray3D<float>(
      Array3D<float>{{{2, 3}, {4, 5}}, {{3, 4}, {5, 6}}}, result);
}

XLA_TEST_F(CustomCallTest, InputAndOutputLayoutDiffer) {
  if (IsMlirLoweringEnabled()) {
    // The MLIR pipeline does /not/ transpose the output here, and there's no
    // obvious reason why it should.
    GTEST_SKIP() << "Appears to test an XLA current implementation detail";
  }

  auto module = CreateNewVerifiedModule();
  auto b = HloComputation::Builder(TestName());

  auto input =
      b.AddInstruction(HloInstruction::CreateParameter(0, r2f32_, "p"));
  b.AddInstruction(
      HloInstruction::CreateCustomCall(r2f32_, {input}, "Add1ToValues"));

  module->AddEntryComputation(b.Build());
  ForceParameterLayout(module.get(), 0, LayoutUtil::MakeLayout({1, 0}));
  ForceResultLayout(module.get(), LayoutUtil::MakeLayout({0, 1}));

  Literal argument = LiteralUtil::CreateR2<float>({{1.f, 2.f}, {3.f, 4.f}});

  // Note, the expected result is transposed! This is because the input and
  // output layouts of the custom call differ and the called function just
  // blindly adds one to each element.
  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {&argument}));
  LiteralTestUtil::ExpectR2Equal<float>({{2.f, 4.f}, {3.f, 5.f}}, result);
}

XLA_TEST_F(CustomCallTest, LayoutConstrained) {
  // The argument and result of the computation are set to different layouts,
  // but the custom call is layout constrained to a fixed operand and result
  // layout, so the correct result should be produced.
  auto module = CreateNewVerifiedModule();
  auto b = HloComputation::Builder(TestName());

  auto input =
      b.AddInstruction(HloInstruction::CreateParameter(0, r2f32_, "p"));

  const Shape& r2f32_dim0_major =
      ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 2}, {1, 0});
  auto custom_call = b.AddInstruction(HloInstruction::CreateCustomCall(
      r2f32_dim0_major, {input}, "Add1ToValues", {r2f32_dim0_major}));
  b.AddInstruction(
      custom_call->CloneWithNewOperands(r2f32_dim0_major, {custom_call}));

  module->AddEntryComputation(b.Build());
  ForceParameterLayout(module.get(), 0, LayoutUtil::MakeLayout({1, 0}));
  ForceResultLayout(module.get(), LayoutUtil::MakeLayout({0, 1}));

  Literal argument = LiteralUtil::CreateR2<float>({{1.f, 2.f}, {3.f, 4.f}});

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {&argument}));
  LiteralTestUtil::ExpectR2Equal<float>({{3.f, 4.f}, {5.f, 6.f}}, result);
}

XLA_TEST_F(CustomCallTest, TupleOutput) {
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = f32[] parameter(0)
      p1 = f32[] parameter(1)
      ROOT %custom-call = (f32[], f32[]) custom-call(f32[] %p0, f32[] %p1), custom_call_target="F32TupleSwap", operand_layout_constraints={f32[], f32[]}
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  Literal arg0 = LiteralUtil::CreateR0<float>(7.f);
  Literal arg1 = LiteralUtil::CreateR0<float>(42.f);

  Literal expected = LiteralUtil::MakeTuple({&arg1, &arg0});
  TF_ASSERT_OK_AND_ASSIGN(auto result,
                          Execute(std::move(module), {&arg0, &arg1}));
  EXPECT_EQ(result, expected);
}

XLA_TEST_F(CustomCallTest, ReportsSuccess) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r0f32_, {constant}, "R0F32Add2Succeed",
      /*opaque=*/"", CustomCallApiVersion::API_VERSION_STATUS_RETURNING));

  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {}));
  LiteralTestUtil::ExpectR0Near<float>(44.0f, result, error_spec_);
}

XLA_TEST_F(CustomCallTest, ReportsFailure) {
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

XLA_TEST_F(CustomCallTest, ReportsFirstFailure) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto constant_1 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  auto constant_2 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(2.0f)));
  auto res_1 = builder.AddInstruction(HloInstruction::CreateCustomCall(
      ShapeUtil::MakeShape(F32, {}), {constant_1}, "CustomCallFail",
      /*opaque=*/"", CustomCallApiVersion::API_VERSION_STATUS_RETURNING));
  auto res_2 = builder.AddInstruction(HloInstruction::CreateCustomCall(
      ShapeUtil::MakeShape(F32, {}), {constant_2}, "CustomCallFail",
      /*opaque=*/"", CustomCallApiVersion::API_VERSION_STATUS_RETURNING));
  builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {}), HloOpcode::kAdd, res_1, res_2));

  module->AddEntryComputation(builder.Build());

  auto status = Execute(std::move(module), {}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(), ::testing::HasSubstr("Failed: 1.0"));
}

XLA_TEST_F(CustomCallTest, TransitiveCustomCallReportsFirstFailure) {
  const char* const kModuleStr = R"(
    HloModule m
    sub {
      p0 = f32[] parameter(0)
      ROOT custom-call = f32[] custom-call(f32[] %p0), custom_call_target="CustomCallFail", api_version=API_VERSION_STATUS_RETURNING
    }
    ENTRY test {
      c0 = f32[] constant(1.0)
      c1 = f32[] constant(2.0)
      call0 = f32[] call(f32[] %c0), to_apply=sub
      call1 = f32[] call(f32[] %c1), to_apply=sub
      ROOT sum = f32[] add(%call0, %call1)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  auto status = Execute(std::move(module), {}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(), HasSubstr("Failed: 1.0"));
}

XLA_TEST_F(CustomCallTest, FillStatusMsgWithBackendConfigStr) {
  if (IsMlirLoweringEnabled()) {
    GTEST_SKIP() << "Invalid values unsupported by MLIR";
  }

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

class CustomCallClientAPITest : public ClientLibraryTestBase {};

// When using the client API, CustomCall targets can't begin with '$' -- these
// are reserved for internal use.
XLA_TEST_F(CustomCallClientAPITest, IllegalCustomCallTarget) {
  XlaBuilder builder(TestName());
  CustomCall(&builder, "$illegal", /*operands=*/{},
             ShapeUtil::MakeShape(F32, {1}));

  absl::StatusOr<std::unique_ptr<GlobalData>> result =
      Execute(&builder, /*arguments=*/{});
  EXPECT_FALSE(result.ok());
}

//===----------------------------------------------------------------------===//
// XLA runtime custom call provides type-safe custom call API
//===----------------------------------------------------------------------===//

namespace {
// Helper functions to get data pointers from buffers
template <typename NativeType, typename BufferType>
static NativeType* DataPointer(BufferType& buffer) {
  return reinterpret_cast<NativeType*>(buffer.data.opaque());
}
template <typename NativeType, typename BufferType>
static NativeType* DataPointer(ffi::Result<BufferType>& buffer) {
  return reinterpret_cast<NativeType*>(buffer->data.opaque());
}

// TODO(abanas): The following three usings are a workaround, delete when
// ResultBuffer is implemented as its own class
using ResultBufferBase = ffi::Result<ffi::BufferBase>;
template <PrimitiveType dtype, size_t rank = xla::ffi::internal::kDynamicRank>
using ResultBuffer = ffi::Result<ffi::Buffer<dtype, rank>>;
template <PrimitiveType dtype>
using ResultBufferR0 = ResultBuffer<dtype, 0>;

using R0F32Buffer = typename ffi::BufferR0<PrimitiveType::F32>;
using F32Buffer = typename ffi::Buffer<PrimitiveType::F32>;
using R0F32ResultBuffer = ResultBufferR0<PrimitiveType::F32>;
using F32ResultBuffer = ResultBuffer<PrimitiveType::F32>;
using BufferBase = ffi::BufferBase;

// Custom kernels definitions and registrations
static absl::Status AlwaysSucceed(ResultBufferBase) { return absl::OkStatus(); }

XLA_FFI_DEFINE_HANDLER(kAlwaysSucceed, AlwaysSucceed,
                       ffi::Ffi::Bind().Ret<BufferBase>()  // unused out buffer
);

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$always_succeed",
                         "Host", kAlwaysSucceed);

static absl::Status AlwaysFail(ResultBufferBase, int32_t value) {
  return absl::InternalError(absl::StrCat("Failed: ", value));
}

XLA_FFI_DEFINE_HANDLER(kAlwaysFail, AlwaysFail,
                       ffi::Ffi::Bind()
                           .Ret<BufferBase>()       // unused out buffer
                           .Attr<int32_t>("value")  // value
);

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$always_fail", "Host",
                         kAlwaysFail);

static absl::Status FfiR0F32Add2(R0F32Buffer in, R0F32ResultBuffer out) {
  auto in_data = DataPointer<float>(in);
  auto out_data = DataPointer<float>(out);
  *out_data = *in_data + 2.0f;
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kFfiR0F32Add2, FfiR0F32Add2,
                       ffi::Ffi::Bind()
                           .Arg<R0F32Buffer>()  // in
                           .Ret<R0F32Buffer>()  // out
);

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$FfiR0F32Add2",
                         "Host", kFfiR0F32Add2);

// This represents a kernel that is valid only for F32 and F64 types
static absl::Status FfiR0FAdd2BufferBase(BufferBase in, ResultBufferBase out) {
  if (in.dtype != out->dtype) {
    return absl::InternalError("Input and output dtypes mismatch");
  }

  switch (in.dtype) {
    case PrimitiveType::F32: {
      auto in_data = DataPointer<float>(in);
      auto out_data = DataPointer<float>(out);
      *out_data = *in_data + 2.0f;
      break;
    }
    case PrimitiveType::F64: {
      auto in_data = DataPointer<double>(in);
      auto out_data = DataPointer<double>(out);
      *out_data = *in_data + 2.0f;
      break;
    }
    default:
      return absl::InternalError("Incorrect type");
  }

  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kFfiR0FAdd2BufferBase, FfiR0FAdd2BufferBase,
                       ffi::Ffi::Bind()
                           .Arg<BufferBase>()  // in
                           .Ret<BufferBase>()  // out
);

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
                         "__xla_test$$FfiR0FAdd2BufferBase", "Host",
                         kFfiR0FAdd2BufferBase);

static absl::Status FfiR0F32AddN(R0F32Buffer in, R0F32ResultBuffer out,
                                 float n) {
  auto in_data = DataPointer<float>(in);
  auto out_data = DataPointer<float>(out);
  *out_data = *in_data + n;
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kFfiR0F32AddN, FfiR0F32AddN,
                       ffi::Ffi::Bind()
                           .Arg<R0F32Buffer>()  // in
                           .Ret<R0F32Buffer>()  // out
                           .Attr<float>("n"));

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$FfiR0F32AddN",
                         "Host", kFfiR0F32AddN);

static absl::Status FfiR0F32AddNPointer(R0F32Buffer in, R0F32ResultBuffer out,
                                        float* n) {
  auto in_data = DataPointer<float>(in);
  auto out_data = DataPointer<float>(out);
  *out_data = *in_data + *n;
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kFfiR0F32AddNPointer, FfiR0F32AddNPointer,
                       ffi::Ffi::Bind()
                           .Arg<R0F32Buffer>()  // in
                           .Ret<R0F32Buffer>()  // out
                           .Attr<ffi::Pointer<float>>("n"));

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$FfiR0F32AddNPointer",
                         "Host", kFfiR0F32AddNPointer);

static absl::Status FfiF32ReduceSum(F32Buffer in, R0F32ResultBuffer out) {
  auto in_data = DataPointer<float>(in);
  auto out_data = DataPointer<float>(out);

  // Calculate the total size of the vector
  const auto size =
      absl::c_accumulate(in.dimensions, 1, std::multiplies<int>());

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
                         "Host", kFfiF32ReduceSum);

static absl::Status FfiF32Add1ToValues(F32Buffer in, F32ResultBuffer out) {
  auto in_data = DataPointer<float>(in);
  auto out_data = DataPointer<float>(out);

  // Calculate and verify the total size of the vector
  const auto in_size =
      absl::c_accumulate(in.dimensions, 1, std::multiplies<int>());
  const auto out_size =
      absl::c_accumulate(out->dimensions, 1, std::multiplies<int>());
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
                         "Host", kFfiF32Add1ToValues);

static absl::Status FfiF32TupleSwap(R0F32Buffer in0, R0F32Buffer in1,
                                    R0F32ResultBuffer out0,
                                    R0F32ResultBuffer out1) {
  auto in_data0 = DataPointer<float>(in0);
  auto in_data1 = DataPointer<float>(in1);
  auto out_data0 = DataPointer<float>(out0);
  auto out_data1 = DataPointer<float>(out1);
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
                         "Host", kFfiF32TupleSwap);

static absl::Status FfiTupleRotate(BufferBase in0, BufferBase in1,
                                   BufferBase in2, BufferBase in3,
                                   ResultBufferBase out0, ResultBufferBase out1,
                                   ResultBufferBase out2,
                                   ResultBufferBase out3) {
  auto in_data0 = DataPointer<float>(in0);
  auto in_data1 = DataPointer<float>(in1);
  auto in_data2 = DataPointer<float>(in2);
  auto in_data3 = DataPointer<float>(in3);
  auto out_data0 = DataPointer<float>(out0);
  auto out_data1 = DataPointer<float>(out1);
  auto out_data2 = DataPointer<float>(out2);
  auto out_data3 = DataPointer<float>(out3);
  *out_data0 = *in_data1;
  *out_data1 = *in_data2;
  *out_data2 = *in_data3;
  *out_data3 = *in_data0;
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kFfiTupleRotate, FfiTupleRotate,
                       ffi::Ffi::Bind()
                           .Arg<BufferBase>()  // in0
                           .Arg<BufferBase>()  // in1
                           .Arg<BufferBase>()  // in2
                           .Arg<BufferBase>()  // in3
                           .Ret<BufferBase>()  // out0
                           .Ret<BufferBase>()  // out1
                           .Ret<BufferBase>()  // out2
                           .Ret<BufferBase>()  // out3
);

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$FfiTupleRotate",
                         "Host", kFfiTupleRotate);

}  // namespace

// TODO(abanas): When #10056 (typed FFI support) is ready, this class can be
// replaced by a simple 'using FfiCustomCallTest = CustomCallTest;'
class FfiCustomCallTest : public CustomCallTest {
 protected:
  void SetUp() override {
    GTEST_SKIP() << "Typed FFI is not supported yet on CPU";
  }
};

XLA_TEST_F(FfiCustomCallTest, FfiReportsSuccess) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r0f32_, {}, "__xla_test$$always_succeed", "",
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  auto status = Execute(std::move(module), {}).status();
  EXPECT_EQ(status, absl::OkStatus());
}

XLA_TEST_F(FfiCustomCallTest, FfiUnknownTarget) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r0f32_, {}, "__xla_test$$unknown_target", "",
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  auto status = Execute(std::move(module), {}).status();
  // NOTE: In the current CPU implementation, the 'kInternal' status code is
  // returned when the target is not found. This behavior differs from that of
  // the GPU, which returns 'kUnimplemented' in such case. When the CPU adopts
  // the thunks runtime, the status code will be unified across both backends.
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(), HasSubstr("No registered implementation"));
}

XLA_TEST_F(FfiCustomCallTest, FfiReportsFailure) {
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

XLA_TEST_F(FfiCustomCallTest, FfiReportsFirstFailure) {
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

  auto status = Execute(std::move(module), {}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(), ::testing::HasSubstr("Failed: 1"));
}

XLA_TEST_F(FfiCustomCallTest, FfiTransitiveCustomCallReportsFirstFailure) {
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

  auto status = Execute(std::move(module), {}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(), HasSubstr("Failed: 2"));
}

XLA_TEST_F(FfiCustomCallTest, FfiWrongNumberOfArguments) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r0f32_, {}, "__xla_test$$FfiR0F32Add2", "",
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  auto status = Execute(std::move(module), {}).status();
  // NOTE: In the current CPU implementation, the 'kInternal' status code is
  // returned when the argument is invalid. This behavior differs from that of
  // the GPU, which returns 'kInvalidArgument' in such case. When the CPU adopts
  // the thunks runtime, the status code will be unified across both backends.
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(), HasSubstr("Wrong number of arguments"));
}

XLA_TEST_F(FfiCustomCallTest, FfiWrongRankOfArgument) {
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
  // NOTE: In the current CPU implementation, the 'kInternal' status code is
  // returned when the argument is invalid. This behavior differs from that of
  // the GPU, which returns 'kInvalidArgument' in such case. When the CPU adopts
  // the thunks runtime, the status code will be unified across both backends.
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(), HasSubstr("Wrong buffer rank"));
}

XLA_TEST_F(FfiCustomCallTest, FfiWrongDTypeOfArgument) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int>(42)));
  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r2f32_, {constant}, "__xla_test$$FfiR0F32Add2", "",
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  auto status = Execute(std::move(module), {}).status();
  // NOTE: In the current CPU implementation, the 'kInternal' status code is
  // returned when the argument is invalid. This behavior differs from that of
  // the GPU, which returns 'kInvalidArgument' in such case. When the CPU adopts
  // the thunks runtime, the status code will be unified across both backends.
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(), HasSubstr("Wrong buffer dtype"));
}

XLA_TEST_F(FfiCustomCallTest, FfiHandleTypedBuffers) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r0f32_, {constant}, "__xla_test$$FfiR0F32Add2", "",
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {}));
  LiteralTestUtil::ExpectR0Near<float>(44.0f, result, error_spec_);
}

XLA_TEST_F(FfiCustomCallTest, FfiHandleInputAsParameters) {
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
  LiteralTestUtil::ExpectR0Near<float>(44.0f, result, error_spec_);
}

XLA_TEST_F(FfiCustomCallTest, FfiHandleBufferBaseFloat) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r0f32_, {constant}, "__xla_test$$FfiR0FAdd2BufferBase", "",
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {}));
  LiteralTestUtil::ExpectR0Near<float>(44.0f, result, error_spec_);
}

XLA_TEST_F(FfiCustomCallTest, FfiHandleBufferBaseDouble) {
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
  LiteralTestUtil::ExpectR0Near<double>(44.0f, result, error_spec_);
}

XLA_TEST_F(FfiCustomCallTest, FfiHandleAttr) {
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
  LiteralTestUtil::ExpectR0Near<float>(45.0f, result, error_spec_);
}

XLA_TEST_F(FfiCustomCallTest, FfiHandleAttrPointer) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  auto constant = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(42.0f)));
  auto n = 4.0f;
  auto ptr = reinterpret_cast<uintptr_t>(&n);
  builder.AddInstruction(HloInstruction::CreateCustomCall(
      r0f32_, {constant}, "__xla_test$$FfiR0F32AddN",
      /*opaque=*/absl::StrFormat("{n = %d : i64}", ptr),
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI));

  module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {}));
  LiteralTestUtil::ExpectR0Near<float>(46.0f, result, error_spec_);
}

XLA_TEST_F(FfiCustomCallTest, FfiHandleR2Vector) {
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
  LiteralTestUtil::ExpectR0Near<float>(10.0f, result, error_spec_);
}

XLA_TEST_F(FfiCustomCallTest, FfiUsedInOtherComputations) {
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

XLA_TEST_F(FfiCustomCallTest, FfiInputAndOutputLayoutDiffer) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());

  if (IsMlirLoweringEnabled()) {
    // The MLIR pipeline does /not/ transpose the output here, and there's no
    // obvious reason why it should.
    GTEST_SKIP() << "Appears to test an XLA current implementation detail";
  }

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

XLA_TEST_F(FfiCustomCallTest, FfiLayoutConstrained) {
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

XLA_TEST_F(FfiCustomCallTest, FfiTupleOutput) {
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

XLA_TEST_F(FfiCustomCallTest, FfiNestedTupleOutput) {
  GTEST_SKIP() << "Nested tuple outputs not yet implemented.";
  const char* const kModuleStr = R"(
    HloModule m

    ENTRY test {
      c0 = f32[] constant(7.0)
      c1 = f32[] constant(42.0)
      c2 = f32[] constant(8.0)
      c3 = f32[] constant(43.0)
      ROOT custom-call = ((f32[], f32[]), (f32[], f32[])) custom-call(c0, c1, c2, c3), custom_call_target="__xla_test$$FfiTupleRotate", api_version=API_VERSION_TYPED_FFI
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));

  Literal arg0 = LiteralUtil::CreateR0<float>(7.f);
  Literal arg1 = LiteralUtil::CreateR0<float>(42.f);
  Literal arg2 = LiteralUtil::CreateR0<float>(8.f);
  Literal arg3 = LiteralUtil::CreateR0<float>(43.f);

  Literal tuple0 = LiteralUtil::MakeTuple({&arg1, &arg2});
  Literal tuple1 = LiteralUtil::MakeTuple({&arg3, &arg0});

  Literal expected = LiteralUtil::MakeTuple({&tuple0, &tuple1});
  TF_ASSERT_OK_AND_ASSIGN(auto result, Execute(std::move(module), {}));
  EXPECT_EQ(result, expected);
}

XLA_TEST_F(FfiCustomCallTest, FfiTupleInput) {
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

XLA_TEST_F(FfiCustomCallTest, FfiNestedTupleInput) {
  GTEST_SKIP() << "Nested tuple inputs not yet implemented.";
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

}  // namespace
}  // namespace xla
