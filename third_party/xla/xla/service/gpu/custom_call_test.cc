/* Copyright 2019 The OpenXLA Authors.

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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/backends/gpu/ffi.h"
#include "xla/ffi/execution_context.h"
#include "xla/ffi/ffi.h"
#include "xla/ffi/ffi_api.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {

struct Range {
  int64_t lo;
  int64_t hi;
};

}  // namespace xla

// Register struct types with XLA:FFI to enable automatic decoding from
// dictionary attributes to structs.
XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(::xla::Range, StructMember<int64_t>("lo"),
                                      StructMember<int64_t>("hi"));

namespace xla {
namespace {
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

class CustomCallTest : public ClientLibraryTestRunnerMixin<
                           HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>> {
 public:
  std::string PlatformName() {
    if (test_runner().HasProperty(HloRunnerPropertyTag::kUsingGpuCuda)) {
      return "CUDA";
    }
    if (test_runner().HasProperty(HloRunnerPropertyTag::kUsingGpuRocm)) {
      return "ROCM";
    }
    LOG(FATAL) << TestName() << " was executed on an unsupported platform.";
  }
};

// The test case for custom call with tokens encodes the arguments and result
// type using a string with A(=Array), T(=Token) and {} for Tuples. It also
// encodes the check that the callback has to do in terms of a string of A and T
// where all the As need to be non-null and all the Ts need to be null. This is
// passed to the custom call as its opaque data.
//
// As an example, "ATTA" for an input encodes 4 inputs to custom call,
// "{A{A}T}" for output encodes a custom call with return type containing a
// single tuple, with another tuple as the 2nd element. For outputs, it is
// either a single element or a tuple. Note, no error checking is performed.

struct TokenTestCase {
  std::string input;
  std::string output;
  std::string opaque;
};

std::vector<TokenTestCase> GetTokenTestCases() {
  return {{"{AT}{AT}", "{AATA}", "ATATAATA"},  // tokens in input and output
          {"{A}", "T", "AT"},                  // single token as output
          {"{{T}}", "A", "TA"},                // single token as input
          {"AA", "{TA}", "AATA"},
          {"TA{TA{TA}}", "{AA}", "TATATAAA"}};
}

class CustomCallTokensTest
    : public ::testing::WithParamInterface<TokenTestCase>,
      public CustomCallTest {
 public:
  static std::vector<XlaOp> BuildInputs(XlaBuilder& b,
                                        std::istringstream& str) {
    std::vector<XlaOp> values;
    while (!str.eof()) {
      int ch = str.get();
      if (ch == 'A') {
        values.push_back(Broadcast(ConstantR0WithType(&b, F32, 1), {128}));
      } else if (ch == 'T') {
        values.push_back(CreateToken(&b));
      } else if (ch == '{') {
        // build a tuple of values. This will eat the } as well.
        std::vector<XlaOp> tuple_elements = BuildInputs(b, str);
        values.push_back(Tuple(&b, tuple_elements));
      } else if (ch == '}') {
        break;
      }
    }
    return values;
  }

  static std::vector<Shape> BuildOutputType(std::istringstream& str) {
    std::vector<Shape> shapes;
    while (!str.eof()) {
      int ch = str.get();
      if (ch == 'A') {
        shapes.push_back(ShapeUtil::MakeShape(F32, {8}));
      } else if (ch == 'T') {
        shapes.push_back(ShapeUtil::MakeTokenShape());
      } else if (ch == '{') {
        // build a tuple shape. This will eat the } as well.
        std::vector<Shape> tuple_elements = BuildOutputType(str);
        shapes.push_back(ShapeUtil::MakeTupleShape(tuple_elements));
      } else if (ch == '}') {
        break;
      }
    }
    return shapes;
  }
};

void Callback_WithStatusSucceeded(void* /*stream*/, void** /*buffers*/,
                                  const char* /*opaque*/, size_t /*opaque_len*/,
                                  XlaCustomCallStatus* status) {
  XlaCustomCallStatusSetSuccess(status);
}

void Callback_WithStatusFailed(void* /*stream*/, void** /*buffers*/,
                               const char* /*opaque*/, size_t /*opaque_len*/,
                               XlaCustomCallStatus* status) {
  XlaCustomCallStatusSetFailure(status, "Failed", 6);
}

TEST_F(CustomCallTest, WithStatusSucceeded) {
  CustomCallTargetRegistry::Global()->Register(
      "Callback_WithStatusSucceeded",
      reinterpret_cast<void*>(Callback_WithStatusSucceeded), PlatformName());

  XlaBuilder b(TestName());
  CustomCall(
      &b, "Callback_WithStatusSucceeded", /*operands=*/{},
      ShapeUtil::MakeShape(F32, {}), /*opaque=*/"",
      /*has_side_effect=*/false,
      /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
      /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
      /*api_version=*/CustomCallApiVersion::API_VERSION_STATUS_RETURNING);
  TF_ASSERT_OK(ExecuteAndTransfer(&b, {}).status());
}

TEST_F(CustomCallTest, WithStatusFailed) {
  CustomCallTargetRegistry::Global()->Register(
      "Callback_WithStatusFailed",
      reinterpret_cast<void*>(Callback_WithStatusFailed), PlatformName());

  XlaBuilder b(TestName());
  CustomCall(
      &b, "Callback_WithStatusFailed", /*operands=*/{},
      ShapeUtil::MakeShape(F32, {}), /*opaque=*/"",
      /*has_side_effect=*/false,
      /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
      /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
      /*api_version=*/CustomCallApiVersion::API_VERSION_STATUS_RETURNING);
  auto status = ExecuteAndTransfer(&b, {}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(), ::testing::HasSubstr("Failed"));
}

//===----------------------------------------------------------------------===//
// XLA runtime custom calls provides type-safe custom call API
//===----------------------------------------------------------------------===//

static absl::Status AlwaysFail(ffi::Result<ffi::AnyBuffer>, int32_t value) {
  return absl::InternalError(absl::StrCat("Uh oh, wrong value: ", value));
}

XLA_FFI_DEFINE_HANDLER(kAlwaysFail, AlwaysFail,
                       ffi::Ffi::Bind()
                           .Ret<ffi::AnyBuffer>()   //
                           .Attr<int32_t>("value")  // value
);

TEST_F(CustomCallTest, RuntimeCustomCallAlwaysFail) {
  xla::ffi::Ffi::RegisterStaticHandler(ffi::GetXlaFfiApi(),
                                       "__xla_test$$always_fail",
                                       PlatformName(), kAlwaysFail);

  XlaBuilder b(TestName());
  CustomCall(&b, "__xla_test$$always_fail", /*operands=*/{},
             ShapeUtil::MakeShape(F32, {}), /*opaque=*/"{value = 42 : i32}",
             /*has_side_effect=*/false,
             /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
             /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
             /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);
  auto status = ExecuteAndTransfer(&b, {}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(), ::testing::HasSubstr("Uh oh, wrong value: 42"));
}

// Same as the above test but just pass attribute through
// the backend config proto string instead.
TEST_F(CustomCallTest, PassAttributesByBackendConfig) {
  xla::ffi::Ffi::RegisterStaticHandler(ffi::GetXlaFfiApi(),
                                       "__xla_test$$always_fail",
                                       PlatformName(), kAlwaysFail);

  XlaBuilder b(TestName());
  CustomCall(
      &b, "__xla_test$$always_fail", /*operands=*/{},
      ShapeUtil::MakeShape(F32, {}), /*opaque=*/
      R"({"custom_call_backend_config": {"attributes": "{value = 42 : i32}"}})",
      /*has_side_effect=*/false,
      /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
      /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);
  auto status = ExecuteAndTransfer(&b, {}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(), ::testing::HasSubstr("Uh oh, wrong value: 42"));
}

static absl::Status Memcpy(se::Stream* stream, ffi::AnyBuffer src,
                           ffi::Result<ffi::AnyBuffer> dst) {
  se::DeviceAddressBase dst_mem = dst->device_memory();
  se::DeviceAddressBase src_mem = src.device_memory();
  return stream->MemcpyD2D(&dst_mem, src_mem, src_mem.size());
}

XLA_FFI_DEFINE_HANDLER(kMemcpy, Memcpy,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::AnyBuffer>()   // src
                           .Ret<ffi::AnyBuffer>(),  // dst
                       {ffi::Traits::kCmdBufferCompatible});

TEST_F(CustomCallTest, ExportedFfiMemcpy) {
  xla::ffi::Ffi::RegisterStaticHandler(
      ffi::GetXlaFfiApi(), "__xla_test$$memcpy", PlatformName(), kMemcpy);

  XlaBuilder b(TestName());
  CustomCall(&b, "__xla_test$$memcpy",
             /*operands=*/{Broadcast(ConstantR0WithType(&b, F32, 42.0), {128})},
             ShapeUtil::MakeShape(F32, {128}), /*opaque=*/"",
             /*has_side_effect=*/false,
             /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
             /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
             /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);
  TF_ASSERT_OK_AND_ASSIGN(auto result, ExecuteAndTransfer(&b, {}));
  EXPECT_THAT(result.data<float>(), ::testing::Each(42));
}

static absl::Status HandleUserPointer(ffi::Result<ffi::AnyBuffer>,
                                      const std::string* str) {
  return absl::InternalError(*str);
}

XLA_FFI_DEFINE_HANDLER(kHandleUserPointer, HandleUserPointer,
                       ffi::Ffi::Bind()
                           .Ret<ffi::AnyBuffer>()  // buffer for result
                           .Attr<ffi::Pointer<std::string>>("message"));

TEST_F(CustomCallTest, PassUserPointerWithAttrs) {
  xla::ffi::Ffi::RegisterStaticHandler(ffi::GetXlaFfiApi(),
                                       "__xla_test$$user_data", PlatformName(),
                                       kHandleUserPointer);

  std::string message = "User-defined message";
  auto ptr = reinterpret_cast<uintptr_t>(&message);

  XlaBuilder b(TestName());
  CustomCall(&b, "__xla_test$$user_data", /*operands=*/{},
             ShapeUtil::MakeShape(F32, {}),
             /*opaque=*/absl::StrFormat("{message = %d : i64}", ptr),
             /*has_side_effect=*/false,
             /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
             /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
             /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);
  auto status = ExecuteAndTransfer(&b, {}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(), ::testing::HasSubstr("User-defined message"));
}

bool is_ffi_invoked = false;
static absl::Status IsInvoked(ffi::Result<ffi::AnyBuffer>) {
  is_ffi_invoked = true;
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(
    kIsInvoked, IsInvoked,
    ffi::Ffi::Bind().Ret<ffi::AnyBuffer>());  // Buffer for result (unused).

TEST_F(CustomCallTest, ExportedFfiIsInvoked) {
  xla::ffi::Ffi::RegisterStaticHandler(
      ffi::GetXlaFfiApi(), "__xla_test$$isinvoked", PlatformName(), kIsInvoked);

  XlaBuilder b(TestName());
  CustomCall(&b, "__xla_test$$isinvoked", /*operands=*/{},
             ShapeUtil::MakeShape(F32, {}), /*opaque=*/"",
             /*has_side_effect=*/false,
             /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
             /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
             /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);
  TF_ASSERT_OK_AND_ASSIGN(auto result, ExecuteAndTransfer(&b, {}));
  EXPECT_TRUE(is_ffi_invoked);
}

TEST_F(CustomCallTest, ExportedFfiUnknownTarget) {
  XlaBuilder b(TestName());
  CustomCall(&b, "__xla_test$$unknown_target", /*operands=*/{},
             ShapeUtil::MakeShape(F32, {}), /*opaque=*/"",
             /*has_side_effect=*/false,
             /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
             /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
             /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);
  auto status = ExecuteAndTransfer(&b, {}).status();
  EXPECT_THAT(
      status,
      StatusIs(
          absl::StatusCode::kNotFound,
          HasSubstr(
              "No FFI handler registered for __xla_test$$unknown_target")));
}

// Memcpy and SubBuffers tests are already ported in
// fusions/address_computation_fusion_test.cc

std::string& kExpectedOpaque = *new std::string("abc\0def", 7);

static absl::Status Opaque(ffi::Result<ffi::AnyBuffer>,
                           const std::string* str) {
  std::string opaque(*str);
  if (opaque != kExpectedOpaque) {
    return absl::InternalError(absl::StrFormat(
        "Opaque string does not match. Expected `%s` but got `%s`",
        kExpectedOpaque, opaque));
  }
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kOpaque, Opaque,
                       ffi::Ffi::Bind()
                           .Ret<ffi::AnyBuffer>()  // Dummy result buffer.
                           .Attr<ffi::Pointer<std::string>>("opaque"));

TEST_F(CustomCallTest, ExportedFfiOpaque) {
  xla::ffi::Ffi::RegisterStaticHandler(
      ffi::GetXlaFfiApi(), "__xla_test$$opaque", PlatformName(), kOpaque);

  XlaBuilder b(TestName());
  const std::string opaque = absl::StrFormat(
      "{opaque = %d : i64}", reinterpret_cast<uintptr_t>(&kExpectedOpaque));
  CustomCall(&b, "__xla_test$$opaque", /*operands=*/{},
             ShapeUtil::MakeShape(F32, {}),
             /*opaque=*/opaque,
             /*has_side_effect=*/false,
             /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
             /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
             /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);
  TF_ASSERT_OK(ExecuteAndTransfer(&b, {}).status());
}

static absl::Status CheckTokens(std::vector<PrimitiveType> args,
                                absl::string_view pattern) {
  if (args.size() != pattern.size()) {
    return absl::InternalError("Incorrect number of arguments");
  }
  for (auto i = 0; i < pattern.size(); ++i) {
    char c = pattern[i];
    bool is_token = args[i] == PrimitiveType::TOKEN;
    if (c == 'T') {
      if (!is_token) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Expected token at position %d", i));
      }
    } else if (c == 'A') {
      if (is_token) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Unexpected token at position %d", i));
      }
    } else {
      return absl::InternalError(
          absl::StrFormat("Unexpected character %c at position %d", c, i));
    }
  }
  return absl::OkStatus();
}

static absl::Status FfiTokens(ffi::RemainingArgs inputs,
                              ffi::RemainingRets outputs,
                              absl::string_view pattern) {
  std::vector<PrimitiveType> types;
  for (auto i = 0; i < inputs.size(); ++i) {
    types.push_back(inputs.get<ffi::AnyBuffer>(i).value().element_type());
  }
  for (auto i = 0; i < outputs.size(); ++i) {
    types.push_back(outputs.get<ffi::AnyBuffer>(i).value()->element_type());
  }
  return CheckTokens(types, pattern);
}

XLA_FFI_DEFINE_HANDLER(
    kFfiTokens, FfiTokens,
    ffi::Ffi::Bind().RemainingArgs().RemainingRets().Attr<absl::string_view>(
        "pattern"));

TEST_P(CustomCallTokensTest, ExportedTokensTest) {
  xla::ffi::Ffi::RegisterStaticHandler(
      ffi::GetXlaFfiApi(), "__xla_test$$tokens", PlatformName(), kFfiTokens);

  const TokenTestCase& tc = GetParam();
  XlaBuilder b(TestName());
  std::istringstream input(tc.input);
  std::istringstream output(tc.output);
  std::vector<XlaOp> call_inputs = BuildInputs(b, input);
  std::vector<Shape> call_output = BuildOutputType(output);
  ASSERT_GE(call_inputs.size(), 1);
  ASSERT_LE(call_inputs.size(), 3);
  ASSERT_EQ(call_output.size(), 1);

  const std::string custom_call_name = "__xla_test$$tokens";
  const std::string opaque = absl::StrFormat("{pattern = \"%s\"}", tc.opaque);
  CustomCall(&b, custom_call_name, /*operands=*/call_inputs,
             call_output.front(),
             /*opaque=*/opaque,
             /*has_side_effect=*/false,
             /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
             /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
             /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);

  TF_ASSERT_OK(ExecuteAndTransfer(&b, {}).status());
}

INSTANTIATE_TEST_SUITE_P(CustomCallTokensTest, CustomCallTokensTest,
                         ::testing::ValuesIn(GetTokenTestCases()));

static absl::Status AlwaysSucceed(ffi::Result<ffi::AnyBuffer>) {
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kAlwaysSucceed, AlwaysSucceed,
                       ffi::Ffi::Bind().Ret<ffi::AnyBuffer>());

TEST_F(CustomCallTest, ExportedFfiWithStatusSucceeded) {
  xla::ffi::Ffi::RegisterStaticHandler(ffi::GetXlaFfiApi(),
                                       "__xla_test$$always_succeed",
                                       PlatformName(), kAlwaysSucceed);

  XlaBuilder b(TestName());
  CustomCall(&b, "__xla_test$$always_succeed", /*operands=*/{},
             ShapeUtil::MakeShape(F32, {}), /*opaque=*/"",
             /*has_side_effect=*/false,
             /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
             /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
             /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);
  TF_ASSERT_OK(ExecuteAndTransfer(&b, {}).status());
}

//===----------------------------------------------------------------------===//
// XLA:FFI handler for testing attributes decoding
//===----------------------------------------------------------------------===//

static absl::Status FfiAttributes(ffi::Result<ffi::AnyBuffer>,
                                  absl::Span<const int32_t> i32_arr,
                                  Range range) {
  if (i32_arr.size() != 4) {
    return absl::InternalError("i32_arr size does not match");
  }

  if (i32_arr[0] != 1 || i32_arr[1] != 2 || i32_arr[2] != 3 ||
      i32_arr[3] != 4) {
    return absl::InternalError("i32_arr values do not match");
  }

  if (range.lo != 0 || range.hi != 42) {
    return absl::InternalError("range values do not match");
  }

  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kFfiAttributes, FfiAttributes,
                       ffi::Ffi::Bind()
                           .Ret<ffi::AnyBuffer>()
                           .Attr<absl::Span<const int32_t>>("i32_arr")
                           .Attr<Range>("range"));

TEST_F(CustomCallTest, FfiAttributes) {
  xla::ffi::Ffi::RegisterStaticHandler(ffi::GetXlaFfiApi(),
                                       "xla.gpu.ffi_attributes", PlatformName(),
                                       kFfiAttributes);

  XlaBuilder b(TestName());
  CustomCall(&b, "xla.gpu.ffi_attributes", /*operands=*/{},
             ShapeUtil::MakeShape(F32, {}),
             /*opaque=*/
             "{ i32_arr = array<i32: 1, 2, 3, 4>,"
             "  range = { lo = 0 : i64, hi = 42 : i64 } }",
             /*has_side_effect=*/false,
             /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
             /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
             /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);
  TF_ASSERT_OK(ExecuteAndTransfer(&b, {}).status());
}

//===----------------------------------------------------------------------===//
// XLA:FFI handler with attached HloComputation
//===----------------------------------------------------------------------===//

static absl::Status MemcpyWithCalledComputation(
    se::Stream* stream, int32_t device_ordinal,
    se::OwningScratchAllocator<> scratch_allocator, ffi::AnyBuffer src,
    ffi::Result<ffi::AnyBuffer> dst, const HloComputation* called_computation) {
  if (called_computation == nullptr) {
    return absl::InternalError("Called computation is not defined");
  }

  if (called_computation->instruction_count() != 1) {
    return absl::InternalError("Unexpected number of instructions");
  }

  if (!DynCast<HloParameterInstruction>(
          called_computation->root_instruction())) {
    return absl::InternalError("ROOT must be a paremeter");
  }

  // Check that scratch allocator is working.
  auto scratch = scratch_allocator.AllocateBytes(1024);
  if (!scratch.ok()) {
    return scratch.status();
  }

  return Memcpy(stream, src, dst);
}

XLA_FFI_DEFINE_HANDLER(kMemcpyWithCalledComputation,
                       MemcpyWithCalledComputation,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Ctx<ffi::DeviceOrdinal>()     // device_ordinal
                           .Ctx<ffi::ScratchAllocator>()  // scratch
                           .Arg<ffi::AnyBuffer>()         // src
                           .Ret<ffi::AnyBuffer>()         // dst
                           .Ctx<ffi::CalledComputation>());

TEST_F(CustomCallTest, WithCalledComputation) {
  xla::ffi::Ffi::RegisterStaticHandler(
      ffi::GetXlaFfiApi(), "xla.gpu.ext.memcpy_with_called_computation",
      PlatformName(), kMemcpyWithCalledComputation);

  auto shape = ShapeUtil::MakeShape(F32, {128});

  // Build a called computation which is just a copy instruction.
  XlaBuilder copy("copy");
  auto p0 = Parameter(&copy, 0, shape, "l_val");
  Copy(p0);
  auto copy_computation = copy.Build().value();

  XlaBuilder b(TestName());
  CustomCallWithComputation(
      &b, "xla.gpu.ext.memcpy_with_called_computation",
      /*operands=*/{Broadcast(ConstantR0WithType(&b, F32, 42.0), {128})},
      copy_computation, shape, /*opaque=*/"",
      /*has_side_effect=*/false,
      /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
      /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);
  TF_ASSERT_OK_AND_ASSIGN(auto result, ExecuteAndTransfer(&b, {}));
  EXPECT_THAT(result.data<float>(), ::testing::Each(42));
}

TEST_F(CustomCallTest, WithCalledComputationAndLayouts) {
  xla::ffi::Ffi::RegisterStaticHandler(
      ffi::GetXlaFfiApi(), "xla.gpu.ext.memcpy_with_called_computation",
      PlatformName(), kMemcpyWithCalledComputation);

  auto shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {128, 128}, {0, 1});
  // Build a called computation which is just a copy instruction.
  XlaBuilder copy("copy");
  auto p0 = Parameter(&copy, 0, shape, "l_val");
  Copy(p0);
  auto copy_computation = copy.Build().value();

  XlaBuilder b(TestName());
  CustomCallWithComputationAndLayouts(
      &b, "xla.gpu.ext.memcpy_with_called_computation",
      /*operands=*/{Broadcast(ConstantR0WithType(&b, F32, 42.0), {128, 128})},
      b.AddSubComputation(copy_computation), shape, {shape}, /*opaque=*/"",
      /*has_side_effect=*/false, /*output_operand_aliasing=*/{},
      /*literal=*/nullptr, /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);
  TF_ASSERT_OK_AND_ASSIGN(auto result, ExecuteAndTransfer(&b, {}, &shape));
  EXPECT_THAT(result.data<float>(), ::testing::Each(42));
}

//===----------------------------------------------------------------------===//
// XLA:FFI handler with execution context
//===----------------------------------------------------------------------===//

// HloRunnerPjRt doesn't offer a way to provide the execution context for the
// execution. Therefore we use a global static variable to pass the execution
// context to the custom call handler.
absl::Mutex execution_context_mutex(absl::kConstInit);
ffi::ExecutionContext* global_execution_context
    ABSL_GUARDED_BY(execution_context_mutex) = nullptr;
absl::NoDestructor<std::optional<ffi::internal::ScopedExecutionContext>>
    scoped_execution_context;

template <ffi::ExecutionStage stage>
absl::Status ExecutionContextRegister(ffi::Result<ffi::AnyBuffer>) {
  if constexpr (stage != ffi::ExecutionStage::kPrepare) {
    return absl::OkStatus();
  }

  absl::MutexLock lock(execution_context_mutex);
  // ScopedExecutionContext needs to be constructed on the same thread as the
  // execution context is used. Therefore we use the prepare callback to
  // create the execution context.
  scoped_execution_context->emplace(global_execution_context);
  return absl::OkStatus();
};

XLA_FFI_DEFINE_HANDLER(
    kExecutionContextRegisterPrepare,
    ExecutionContextRegister<ffi::ExecutionStage::kPrepare>,
    ffi::Ffi::Bind<ffi::ExecutionStage::kPrepare>().Ret<ffi::AnyBuffer>());
XLA_FFI_DEFINE_HANDLER(
    kExecutionContextRegisterExecute,
    ExecutionContextRegister<ffi::ExecutionStage::kExecute>,
    ffi::Ffi::Bind<ffi::ExecutionStage::kExecute>().Ret<ffi::AnyBuffer>());

// Arbitrary user-defined context passed via the execution context side channel
// to a custom call handlers.
struct SomeExtraContext {
  explicit SomeExtraContext(int32_t value) : value(value) {}
  int32_t value;
  bool prepared = false;
  bool initialized = false;
  bool executed = false;
};

template <ffi::ExecutionStage stage>
static absl::Status ExecutionContext(ffi::AnyBuffer,
                                     ffi::Result<ffi::AnyBuffer>,
                                     SomeExtraContext* ctx) {
  if (ctx->value != 42) {
    return absl::InternalError("Unexpected value");
  }
  if constexpr (stage == ffi::ExecutionStage::kPrepare) {
    ctx->prepared = true;
  } else if constexpr (stage == ffi::ExecutionStage::kInitialize) {
    ctx->initialized = true;
  } else if constexpr (stage == ffi::ExecutionStage::kExecute) {
    ctx->executed = true;
  } else {
    return absl::InternalError("Unexpected stage");
  }

  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kExecutionContextPrepare,
                       ExecutionContext<ffi::ExecutionStage::kPrepare>,
                       ffi::Ffi::Bind<ffi::ExecutionStage::kPrepare>()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ctx<ffi::UserData<SomeExtraContext>>());

XLA_FFI_DEFINE_HANDLER(kExecutionContextInitialize,
                       ExecutionContext<ffi::ExecutionStage::kInitialize>,
                       ffi::Ffi::Bind<ffi::ExecutionStage::kInitialize>()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ctx<ffi::UserData<SomeExtraContext>>());

XLA_FFI_DEFINE_HANDLER(kExecutionContextExecute,
                       ExecutionContext<ffi::ExecutionStage::kExecute>,
                       ffi::Ffi::Bind<ffi::ExecutionStage::kExecute>()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>()
                           .Ctx<ffi::UserData<SomeExtraContext>>());

TEST_F(CustomCallTest, FfiExecutionContext) {
  xla::ffi::Ffi::RegisterStaticHandler(
      ffi::GetXlaFfiApi(), "xla.gpu.register_ffi_execution_context",
      PlatformName(),
      {
          /*instantiate=*/nullptr,
          /*prepare=*/kExecutionContextRegisterPrepare,
          /*initialize=*/nullptr,
          /*execute=*/kExecutionContextRegisterExecute,
      });

  xla::ffi::Ffi::RegisterStaticHandler(
      ffi::GetXlaFfiApi(), "xla.gpu.ffi_execution_context", PlatformName(),
      {
          /*instantiate=*/nullptr,
          /*prepare=*/kExecutionContextPrepare,
          /*initialize=*/kExecutionContextInitialize,
          /*execute=*/kExecutionContextExecute,
      });

  XlaBuilder b(TestName());

  // This custom call users ScopedExecutionContext to register the execution
  // context for the duration of the current XLA computation.
  // Usually the execution context is passed in via ExecutionOptions, but that's
  // not supported in HloRunnerPjRt.
  XlaOp output =
      CustomCall(&b, "xla.gpu.register_ffi_execution_context",
                 /*operands=*/{}, ShapeUtil::MakeShape(F32, {}),
                 /*opaque=*/"",
                 /*has_side_effect=*/true,
                 /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
                 /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
                 /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);
  CustomCall(&b, "xla.gpu.ffi_execution_context", /*operands=*/{output},
             ShapeUtil::MakeShape(F32, {}),
             /*opaque=*/"",
             /*has_side_effect=*/false,
             /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
             /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
             /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);

  ffi::ExecutionContext execution_context;
  TF_ASSERT_OK(execution_context.Emplace<SomeExtraContext>(42));
  {
    absl::MutexLock lock(execution_context_mutex);
    global_execution_context = &execution_context;
  }

  TF_ASSERT_OK(ExecuteAndTransfer(&b, {}).status());

  // Check that FFI handler was called during initialization and execution.
  TF_ASSERT_OK_AND_ASSIGN(auto* user_context,
                          execution_context.Lookup<SomeExtraContext>());
  EXPECT_TRUE(user_context->prepared);
  EXPECT_TRUE(user_context->initialized);
  EXPECT_TRUE(user_context->executed);
}

//===----------------------------------------------------------------------===//
// Stateful XLA:FFI handler
//===----------------------------------------------------------------------===//

struct SomeState {
  explicit SomeState(int32_t value) : value(value) {}
  int32_t value = 0;
};

// Every time custom call HLO operation is instantiated as a GPU runtime Thunk,
// XLA calls instantiate callback to create a new instance of the handler state,
// that will be passed to all other FFI handler calls.
static absl::StatusOr<std::unique_ptr<SomeState>> InstantiateState() {
  return std::make_unique<SomeState>(42);
}

// At run time we can access the state created by the instantiate callback.
static absl::Status GetState(ffi::Result<ffi::AnyBuffer>, SomeState* state) {
  if (state->value != 42) {
    return absl::InternalError("Unexpected value");
  }
  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kInstantiateState, InstantiateState,
                       ffi::Ffi::BindInstantiate());

XLA_FFI_DEFINE_HANDLER(
    kGetState, GetState,
    ffi::Ffi::Bind().Ret<ffi::AnyBuffer>().Ctx<ffi::State<SomeState>>());

TEST_F(CustomCallTest, FfiExecutionState) {
  xla::ffi::Ffi::RegisterStaticHandler(
      ffi::GetXlaFfiApi(), "xla.gpu.ffi_execution_state", PlatformName(),
      {
          /*instantiate=*/kInstantiateState,
          /*prepare=*/nullptr,
          /*initialize=*/nullptr,
          /*execute=*/kGetState,
      });

  XlaBuilder b(TestName());
  CustomCall(&b, "xla.gpu.ffi_execution_state", /*operands=*/{},
             ShapeUtil::MakeShape(F32, {}),
             /*opaque=*/"",
             /*has_side_effect=*/false,
             /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
             /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
             /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);

  TF_ASSERT_OK(ExecuteAndTransfer(&b, {}).status());
}

//===----------------------------------------------------------------------===//
// Asynchronous custom calls example.
//===----------------------------------------------------------------------===//

// This is an example of how to implement an asynchronous custom call:
//
//   1. Start custom call initiates async operations and extends the lifetime of
//      the input buffer by aliasing it with the output.
//   2. Done custom call waits for the async operations to complete and returns
//      the result.
//
// Because HLO type system doesn't allow to express arbitrary values passed
// between operations, we rely on a "side channel" to communicate between
// start and done custom calls. In this example, this side channel is
// implemented as a global static map.
static absl::NoDestructor<absl::flat_hash_map<int32_t, void*>> async_work_map;

static absl::Status AsyncStartCustomCall(ffi::AnyBuffer arg,
                                         ffi::Result<ffi::AnyBuffer> ret,
                                         int32_t channel) {
  // Inside that start custom call we alias input with output and by doing that
  // extend the lifetime of the input buffer until the linked done custom call.
  EXPECT_EQ(arg.untyped_data(), ret->untyped_data());
  EXPECT_EQ(arg.element_type(), F32);
  EXPECT_EQ(ret->element_type(), F32);

  EXPECT_TRUE(async_work_map->empty());
  async_work_map->insert({channel, arg.untyped_data()});

  return absl::OkStatus();
}

static absl::Status AsyncDoneCustomCall(ffi::AnyBuffer arg,
                                        ffi::Result<ffi::AnyBuffer> ret,
                                        int32_t channel) {
  // In done custom call we "allocate" real result buffer.
  EXPECT_NE(arg.untyped_data(), ret->untyped_data());
  EXPECT_EQ(arg.element_type(), F32);

  // Chat that argument is the same as the one we put into a map earlier.
  EXPECT_EQ(async_work_map->at(channel), arg.untyped_data());

  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(
    kAsyncStartCustomCall, AsyncStartCustomCall,
    ffi::Ffi::Bind().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Attr<int32_t>(
        "channel"));

XLA_FFI_DEFINE_HANDLER(
    kAsyncDoneCustomCall, AsyncDoneCustomCall,
    ffi::Ffi::Bind().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Attr<int32_t>(
        "channel"));

TEST_F(CustomCallTest, AsyncCustomCalls) {
  xla::ffi::Ffi::RegisterStaticHandler(ffi::GetXlaFfiApi(),
                                       "xla.gpu.async_start_custom_call",
                                       PlatformName(), kAsyncStartCustomCall);
  xla::ffi::Ffi::RegisterStaticHandler(ffi::GetXlaFfiApi(),
                                       "xla.gpu.async_done_custom_call",
                                       PlatformName(), kAsyncDoneCustomCall);

  auto shape = ShapeUtil::MakeShape(F32, {});

  XlaBuilder b(TestName());
  auto p0 = Parameter(&b, 0, shape, "p0");

  auto start = CustomCall(
      &b, "xla.gpu.async_start_custom_call",
      /*operands=*/{Copy(p0)}, ShapeUtil::MakeShape(F32, {}),
      /*opaque=*/"{channel = 0 : i32}",
      /*has_side_effect=*/false,
      /*output_operand_aliasing=*/{{{}, {0, {}}}}, /*literal=*/nullptr,
      /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
      /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);

  CustomCall(&b, "xla.gpu.async_done_custom_call",
             /*operands=*/{start}, ShapeUtil::MakeShape(F32, {}),
             /*opaque=*/"{channel = 0 : i32}",
             /*has_side_effect=*/false,
             /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
             /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
             /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);

  Literal literal = LiteralUtil::CreateR0<float>(42.0f);
  TF_ASSERT_OK(ExecuteAndTransfer(&b, {&literal}).status());
}

//===----------------------------------------------------------------------===//
// Testing the use of buffers in custom calls.
//===----------------------------------------------------------------------===//

using CustomCallHloTest = CustomCallTest;

static absl::Status AddOne(se::Stream* stream, ffi::AnyBuffer src,
                           ffi::Result<ffi::AnyBuffer> ret) {
  if (src.untyped_data() != ret->untyped_data()) {
    return absl::InternalError("Input and output buffers must be the same.");
  }

  int32_t data[2];
  se::DeviceAddressBase buffer_mem = ret->device_memory();
  TF_RETURN_IF_ERROR(stream->Memcpy(data, buffer_mem, sizeof(data)));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

  data[0] += 1;
  data[1] += 1;

  TF_RETURN_IF_ERROR(stream->Memcpy(&buffer_mem, data, sizeof(data)));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

  return absl::OkStatus();
}

XLA_FFI_DEFINE_HANDLER(kAddOne, AddOne,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>());

TEST_F(CustomCallHloTest, HloBufferStraightLine) {
  xla::ffi::Ffi::RegisterStaticHandler(ffi::GetXlaFfiApi(), "xla.gpu.add_one",
                                       PlatformName(), kAddOne);

  const char* const kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    c1 = s32[] constant(1)
    init = s32[2] broadcast(c1), dimensions={}
    b0 = b(s32[2]) custom-call(init), custom_call_target="Pin",
      output_to_operand_aliasing={{}: (0, {})}
    b1 = b(s32[2]) custom-call(b0), custom_call_target="xla.gpu.add_one",
      output_to_operand_aliasing={{}: (0, {})},
      api_version=API_VERSION_TYPED_FFI
    b2 = b(s32[2]) custom-call(b1), custom_call_target="xla.gpu.add_one",
      output_to_operand_aliasing={{}: (0, {})},
      api_version=API_VERSION_TYPED_FFI
    ROOT v = s32[2] custom-call(b2), custom_call_target="Unpin",
      output_to_operand_aliasing={{}: (0, {})}
  })";

  const int64_t kNumReplicas = 1;
  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  auto module = ParseAndReturnUnverifiedModule(kModuleStr, config);
  EXPECT_TRUE(module.ok());
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module.value()), absl::Span<Literal* const>{},
                        kNumReplicas,
                        /*use_threads=*/true, /*run_hlo_passes=*/true));
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_THAT(results[0].data<int32_t>(), ::testing::Each(3));
}

TEST_F(CustomCallHloTest, HloBufferRotated) {
  xla::ffi::Ffi::RegisterStaticHandler(ffi::GetXlaFfiApi(), "xla.gpu.add_one",
                                       PlatformName(), kAddOne);

  const char* const kModuleStr = R"(

  HloModule test
  cond {
    param = (s32[], b(s32[2])) parameter(0)
    count = get-tuple-element(%param), index=0
    ub = s32[] constant(2)
    ROOT compare = pred[] compare(count, ub), direction=LT
  }

  body {
    param = (s32[], b(s32[2])) parameter(0)
    count = get-tuple-element(%param), index=0
    b3 = get-tuple-element(%param), index=1

    c1 = s32[] constant(1)
    new_count = s32[] add(count, c1)
    b4 = b(s32[2]) custom-call(b3), custom_call_target="xla.gpu.add_one",
      output_to_operand_aliasing={{}: (0, {})},
      api_version=API_VERSION_TYPED_FFI
    b5 = b(s32[2]) custom-call(b4), custom_call_target="xla.gpu.add_one",
      output_to_operand_aliasing={{}: (0, {})},
      api_version=API_VERSION_TYPED_FFI
    v0 = s32[2] custom-call(b5), custom_call_target="Unpin",
      output_to_operand_aliasing={{}: (0, {})}
    c1_broadcast = s32[2] broadcast(c1), dimensions={}
    v1 = s32[2] add(c1_broadcast, v0)

    b6 = b(s32[2]) custom-call(v1), custom_call_target="Pin",
      output_to_operand_aliasing={{}: (0, {})}
    ROOT result = (s32[], b(s32[2])) tuple(new_count, b6)
  }

  ENTRY test_computation {
    c0 = s32[] constant(0)
    c1 = s32[] constant(1)
    init = s32[2] broadcast(c1), dimensions={}
    b0 = b(s32[2]) custom-call(init), custom_call_target="Pin",
      output_to_operand_aliasing={{}: (0, {})}
    while_init = (s32[], b(s32[2])) tuple(c0, b0)
    while_result = (s32[], b(s32[2])) while(while_init), body=body, condition=cond
    b1 = b(s32[2]) get-tuple-element(while_result), index=1
    ROOT v = s32[2] custom-call(b1), custom_call_target="Unpin",
      output_to_operand_aliasing={{}: (0, {})}
  })";

  const int64_t kNumReplicas = 1;
  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  auto module = ParseAndReturnUnverifiedModule(kModuleStr, config);
  EXPECT_TRUE(module.ok());
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module.value()), absl::Span<Literal* const>{},
                        kNumReplicas,
                        /*use_threads=*/true, /*run_hlo_passes=*/true));
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_THAT(results[0].data<int32_t>(), ::testing::Each(7));
}

// Adds 1 to 2 elements with the given offset in the input buffer.
absl::Status UpadteBufferImpl(se::Stream* stream, ffi::AnyBuffer src,
                              ffi::Result<ffi::AnyBuffer> ret, int offset) {
  if (src.untyped_data() != ret->untyped_data()) {
    return absl::InternalError("Input and output buffers must be the same.");
  }
  if (offset < 0 || offset > 2) {
    return absl::InternalError("Offset must be in [0, 2].");
  }
  int32_t data[2];
  se::DeviceAddressBase buffer_mem = ret->device_memory();
  TF_RETURN_IF_ERROR(stream->Memcpy(data, buffer_mem, sizeof(data)));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

  data[offset] += 1;
  data[offset + 1] += 1;

  TF_RETURN_IF_ERROR(stream->Memcpy(&buffer_mem, data, sizeof(data)));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

  return absl::OkStatus();
}

// Adds 1 to the first 2 elements of the input buffer.
static absl::Status UpdateBuffer1(se::Stream* stream, ffi::AnyBuffer src,
                                  ffi::Result<ffi::AnyBuffer> ret) {
  return UpadteBufferImpl(stream, src, ret, /*offset=*/0);
}
// Adds 1 to the last 2 elements of the input buffer.
static absl::Status UpdateBuffer2(se::Stream* stream, ffi::AnyBuffer src,
                                  ffi::Result<ffi::AnyBuffer> ret) {
  return UpadteBufferImpl(stream, src, ret, /*offset=*/2);
}

XLA_FFI_DEFINE_HANDLER(kUpdateBuffer1, UpdateBuffer1,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>());
XLA_FFI_DEFINE_HANDLER(kUpdateBuffer2, UpdateBuffer2,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::AnyBuffer>()
                           .Ret<ffi::AnyBuffer>());

// TODO: Enable this test once the emitter failure is fixed.
TEST_F(CustomCallHloTest, DISABLED_CallConcurrentUpdateTwoBuffers) {
  xla::ffi::Ffi::RegisterStaticHandler(ffi::GetXlaFfiApi(),
                                       "xla.gpu.update_buffer1", PlatformName(),
                                       kUpdateBuffer1);
  xla::ffi::Ffi::RegisterStaticHandler(ffi::GetXlaFfiApi(),
                                       "xla.gpu.update_buffer2", PlatformName(),
                                       kUpdateBuffer2);
  const char* const kModuleStr = R"(

  HloModule test

  async_comp1 {
   pa1 = b(s32[4]) parameter(0)
   ROOT va1 = b(s32[4]) custom-call(pa1),
     custom_call_target="xla.gpu.update_buffer1",
     output_to_operand_aliasing={{}: (0, {})}, api_version=API_VERSION_TYPED_FFI
  }

  async_comp2 {
   pa2 = b(s32[4]) parameter(0)
   ROOT va2 = b(s32[4]) custom-call(pa2),
     custom_call_target="xla.gpu.update_buffer2",
     output_to_operand_aliasing={{}: (0, {})}, api_version=API_VERSION_TYPED_FFI
  }

  ENTRY test_computation {
    p0 = s32[4] parameter(0)
    p1 = s32[4] parameter(1)

    b1_0 = b(s32[4]) custom-call(p0), custom_call_target="Pin",
      output_to_operand_aliasing={{}: (0, {})}
    b2_0 = b(s32[4]) custom-call(p1), custom_call_target="Pin",
      output_to_operand_aliasing={{}: (0, {})}

    b1_1 = b(s32[4]) call(b1_0), to_apply=async_comp1,
      frontend_attributes={_xla_stream_annotation="1", inlineable="false"}
    b2_1 = b(s32[4]) call(b2_0), to_apply=async_comp2,
      frontend_attributes={_xla_stream_annotation="2", inlineable="false"}

    v_1 = s32[4] custom-call(b1_1), custom_call_target="Unpin",
      output_to_operand_aliasing={{}: (0, {})}
    v_2 = s32[4] custom-call(b2_1), custom_call_target="Unpin",
    output_to_operand_aliasing={{}: (0, {})}
    ROOT or = s32[4] or(v_1, v_2)
  })";

  const int64_t kNumReplicas = 1;
  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  config.mutable_debug_options().set_xla_gpu_experimental_stream_annotation(
      true);
  config.mutable_debug_options().clear_xla_gpu_enable_command_buffer();
  auto module = ParseAndReturnUnverifiedModule(kModuleStr, config);
  EXPECT_TRUE(module.ok());
  Array<int32_t> input1({4}), input2({4});
  input1.Fill(0);
  input2.Fill(0);
  Literal input_literal1 = LiteralUtil::CreateFromArray(input1);
  Literal input_literal2 = LiteralUtil::CreateFromArray(input2);

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module.value()),
                        {{&input_literal1, &input_literal2}}, kNumReplicas,
                        /*use_threads=*/true, /*run_hlo_passes=*/true));
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_THAT(results[0].data<int32_t>(), ::testing::Each(1));
}

// TODO: Enable this test once the runtime failure is fixed.
TEST_F(CustomCallHloTest, DISABLED_CustomCallConcurrentUpdateTwoBuffers) {
  xla::ffi::Ffi::RegisterStaticHandler(ffi::GetXlaFfiApi(),
                                       "xla.gpu.update_buffer1", PlatformName(),
                                       kUpdateBuffer1);
  xla::ffi::Ffi::RegisterStaticHandler(ffi::GetXlaFfiApi(),
                                       "xla.gpu.update_buffer2", PlatformName(),
                                       kUpdateBuffer2);
  const char* const kModuleStr = R"(

  HloModule test

  async_comp1 {
   pa1 = b(s32[4]) parameter(0)
   ROOT va1 = b(s32[4]) custom-call(pa1),
     custom_call_target="xla.gpu.update_buffer1",
     output_to_operand_aliasing={{}: (0, {})}, api_version=API_VERSION_TYPED_FFI
  }

  async_comp2 {
   pa2 = b(s32[4]) parameter(0)
   ROOT va2 = b(s32[4]) custom-call(pa2),
     custom_call_target="xla.gpu.update_buffer2",
     output_to_operand_aliasing={{}: (0, {})}, api_version=API_VERSION_TYPED_FFI
  }

  ENTRY test_computation {
    p0 = s32[4] parameter(0)
    p1 = s32[4] parameter(1)

    b1_0 = b(s32[4]) custom-call(p0), custom_call_target="Pin",
      output_to_operand_aliasing={{}: (0, {})}
    b2_0 = b(s32[4]) custom-call(p1), custom_call_target="Pin",
      output_to_operand_aliasing={{}: (0, {})}

    b1_1_start = ((b(s32[4])), b(s32[4])) async-start(b1_0), calls=async_comp1,
      frontend_attributes={_xla_stream_annotation="1"}
    b1_1 = b(s32[4]) async-done(b1_1_start),
      frontend_attributes={_xla_stream_annotation="1"},
      backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"force_earliest_schedule":false,"reification_cost":[],"device_type":"DEVICE_TYPE_INVALID"}
    b2_1_start = ((b(s32[4])), b(s32[4])) async-start(b2_0), calls=async_comp2,
      frontend_attributes={_xla_stream_annotation="2"}
    b2_1 = b(s32[4]) async-done(b2_1_start),
      frontend_attributes={_xla_stream_annotation="2"},
      backend_config={"operation_queue_id":"0","wait_on_operation_queues":[],"force_earliest_schedule":false,"reification_cost":[],"device_type":"DEVICE_TYPE_INVALID"}

    v_1 = s32[4] custom-call(b1_1), custom_call_target="Unpin",
      output_to_operand_aliasing={{}: (0, {})}
    v_2 = s32[4] custom-call(b2_1), custom_call_target="Unpin",
    output_to_operand_aliasing={{}: (0, {})}
    ROOT or = s32[4] or(v_1, v_2)
  })";

  const int64_t kNumReplicas = 1;
  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  config.mutable_debug_options().set_xla_gpu_experimental_stream_annotation(
      true);
  config.mutable_debug_options().clear_xla_gpu_enable_command_buffer();
  auto module = ParseAndReturnUnverifiedModule(kModuleStr, config);
  EXPECT_TRUE(module.ok());
  Array<int32_t> input1({4}), input2({4});
  input1.Fill(0);
  input2.Fill(0);
  Literal input_literal1 = LiteralUtil::CreateFromArray(input1);
  Literal input_literal2 = LiteralUtil::CreateFromArray(input2);

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> results,
      ExecuteReplicated(std::move(module.value()),
                        {{&input_literal1, &input_literal2}}, kNumReplicas,
                        /*use_threads=*/true, /*run_hlo_passes=*/true));
  ASSERT_EQ(results.size(), kNumReplicas);
  EXPECT_THAT(results[0].data<int32_t>(), ::testing::Each(1));
}

}  // anonymous namespace
}  // namespace xla
