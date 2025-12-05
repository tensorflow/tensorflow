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
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "xla/literal_util.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"  // IWYU pragma: keep
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#define PLATFORM "CUDA"
#elif TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_runtime.h"
#define PLATFORM "ROCM"
#endif

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
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
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/tests/client_library_test_runner_mixin.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

#if GOOGLE_CUDA
#define gpuSuccess cudaSuccess
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#elif TENSORFLOW_USE_ROCM
#define gpuSuccess hipSuccess
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemcpy hipMemcpy
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#endif

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

using CustomCallTest = ClientLibraryTestRunnerMixin<HloTestBase>;

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

void Callback_Tokens(se::gpu::GpuStreamHandle stream, void** buffers,
                     const char* opaque, size_t opaque_len) {
  for (int i = 0; i < opaque_len; ++i) {
    char c = opaque[i];
    ASSERT_TRUE(c == 'A' || c == 'T');
    if (c == 'A') {
      ASSERT_NE(buffers[i], nullptr);
    } else {
      ASSERT_EQ(buffers[i], nullptr);
    }
  }
}

XLA_REGISTER_CUSTOM_CALL_TARGET(Callback_Tokens, PLATFORM);

std::vector<TokenTestCase> GetTokenTestCases() {
  return {{"{AT}{AT}", "{A{AT}A}", "ATATAATA"},  // tokens in input and output
          {"{A}", "T", "AT"},                    // single token as output
          {"{{T}}", "A", "TA"},                  // single token as input
          {"AA", "{TA}", "AATA"},
          {"TA{TA{TA}}", "{AA}", "TATATAAA"}};
}

class CustomCallTokensTest
    : public ::testing::WithParamInterface<TokenTestCase>,
      public ClientLibraryTestRunnerMixin<HloTestBase> {
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

void Callback_WithStatusSucceeded(se::gpu::GpuStreamHandle /*stream*/,
                                  void** /*buffers*/, const char* /*opaque*/,
                                  size_t /*opaque_len*/,
                                  XlaCustomCallStatus* status) {
  XlaCustomCallStatusSetSuccess(status);
}
XLA_REGISTER_CUSTOM_CALL_TARGET(Callback_WithStatusSucceeded, PLATFORM);

TEST_F(CustomCallTest, WithStatusSucceeded) {
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

void Callback_WithStatusFailed(se::gpu::GpuStreamHandle /*stream*/,
                               void** /*buffers*/, const char* /*opaque*/,
                               size_t /*opaque_len*/,
                               XlaCustomCallStatus* status) {
  XlaCustomCallStatusSetFailure(status, "Failed", 6);
}
XLA_REGISTER_CUSTOM_CALL_TARGET(Callback_WithStatusFailed, PLATFORM);

TEST_F(CustomCallTest, WithStatusFailed) {
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
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$always_fail",
                         PLATFORM, kAlwaysFail);

TEST_F(CustomCallTest, RuntimeCustomCallAlwaysFail) {
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
  se::DeviceMemoryBase dst_mem = dst->device_memory();
  se::DeviceMemoryBase src_mem = src.device_memory();
  return stream->MemcpyD2D(&dst_mem, src_mem, src_mem.size());
}

XLA_FFI_DEFINE_HANDLER(kMemcpy, Memcpy,
                       ffi::Ffi::Bind()
                           .Ctx<ffi::Stream>()
                           .Arg<ffi::AnyBuffer>()   // src
                           .Ret<ffi::AnyBuffer>(),  // dst
                       {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$memcpy", PLATFORM,
                         kMemcpy);

TEST_F(CustomCallTest, ExportedFfiMemcpy) {
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

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$user_data", PLATFORM,
                         kHandleUserPointer);

TEST_F(CustomCallTest, PassUserPointerWithAttrs) {
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

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$isinvoked", PLATFORM,
                         kIsInvoked);

TEST_F(CustomCallTest, ExportedFfiIsInvoked) {
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

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$opaque", PLATFORM,
                         kOpaque);

TEST_F(CustomCallTest, ExportedFfiOpaque) {
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

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$tokens", PLATFORM,
                         kFfiTokens);

TEST_P(CustomCallTokensTest, ExportedTokensTest) {
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

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "__xla_test$$always_succeed",
                         PLATFORM, kAlwaysSucceed);

TEST_F(CustomCallTest, ExportedFfiWithStatusSucceeded) {
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

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla.gpu.ffi_attributes",
                         PLATFORM, kFfiAttributes);

TEST_F(CustomCallTest, FfiAttributes) {
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

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(),
                         "xla.gpu.ext.memcpy_with_called_computation", PLATFORM,
                         kMemcpyWithCalledComputation);

TEST_F(CustomCallTest, WithCalledComputation) {
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
static absl::Status ExecutionContext(ffi::Result<ffi::AnyBuffer>,
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
                           .Ret<ffi::AnyBuffer>()
                           .Ctx<ffi::UserData<SomeExtraContext>>());

XLA_FFI_DEFINE_HANDLER(kExecutionContextInitialize,
                       ExecutionContext<ffi::ExecutionStage::kInitialize>,
                       ffi::Ffi::Bind<ffi::ExecutionStage::kInitialize>()
                           .Ret<ffi::AnyBuffer>()
                           .Ctx<ffi::UserData<SomeExtraContext>>());

XLA_FFI_DEFINE_HANDLER(kExecutionContextExecute,
                       ExecutionContext<ffi::ExecutionStage::kExecute>,
                       ffi::Ffi::Bind<ffi::ExecutionStage::kExecute>()
                           .Ret<ffi::AnyBuffer>()
                           .Ctx<ffi::UserData<SomeExtraContext>>());

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla.gpu.ffi_execution_context",
                         PLATFORM,
                         {
                             /*instantiate=*/nullptr,
                             /*prepare=*/kExecutionContextPrepare,
                             /*initialize=*/kExecutionContextInitialize,
                             /*execute=*/kExecutionContextExecute,
                         });

TEST_F(CustomCallTest, FfiExecutionContext) {
  XlaBuilder b(TestName());
  CustomCall(&b, "xla.gpu.ffi_execution_context", /*operands=*/{},
             ShapeUtil::MakeShape(F32, {}),
             /*opaque=*/"",
             /*has_side_effect=*/false,
             /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
             /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
             /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);

  ffi::ExecutionContext execution_context;
  TF_ASSERT_OK(execution_context.Emplace<SomeExtraContext>(42));

  ffi::internal::ScopedExecutionContext scoped_execution_context(
      &execution_context);

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

XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla.gpu.ffi_execution_state",
                         PLATFORM,
                         {
                             /*instantiate=*/kInstantiateState,
                             /*prepare=*/nullptr,
                             /*initialize=*/nullptr,
                             /*execute=*/kGetState,
                         });

TEST_F(CustomCallTest, FfiExecutionState) {
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
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla.gpu.async_start_custom_call",
                         PLATFORM, kAsyncStartCustomCall);

XLA_FFI_DEFINE_HANDLER(
    kAsyncDoneCustomCall, AsyncDoneCustomCall,
    ffi::Ffi::Bind().Arg<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Attr<int32_t>(
        "channel"));
XLA_FFI_REGISTER_HANDLER(ffi::GetXlaFfiApi(), "xla.gpu.async_done_custom_call",
                         PLATFORM, kAsyncDoneCustomCall);

TEST_F(CustomCallTest, AsyncCustomCalls) {
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

class CustomCallHloTest : public HloTestBase {};

void CallBack_AddOne(se::gpu::GpuStreamHandle stream, void** buffers,
                     const char* /*opaque*/, size_t /*opaque_len*/) {
  // Expect that the input and output buffers are the same.
  if (buffers[0] != buffers[1]) {
    return;
  }
  int32_t dst[2];
  auto err = gpuMemcpy(dst, buffers[0], /*count=*/sizeof(int32_t) * 2,
                       gpuMemcpyDeviceToHost);
  ASSERT_EQ(err, gpuSuccess);
  dst[0] += 1;
  dst[1] += 1;
  err = gpuMemcpy(buffers[1], dst, /*count=*/sizeof(int32_t) * 2,
                  gpuMemcpyHostToDevice);
}
XLA_REGISTER_CUSTOM_CALL_TARGET(CallBack_AddOne, PLATFORM);

TEST_F(CustomCallHloTest, HloBufferStraightLine) {
  const char* const kModuleStr = R"(

  HloModule test
  ENTRY test_computation {
    c1 = s32[] constant(1)
    init = s32[2] broadcast(c1), dimensions={}
    b0 = b(s32[2]) custom-call(init), custom_call_target="Pin",
      output_to_operand_aliasing={{}: (0, {})}
    b1 = b(s32[2]) custom-call(b0), custom_call_target="CallBack_AddOne",
      output_to_operand_aliasing={{}: (0, {})},
      api_version=API_VERSION_STATUS_RETURNING
    b2 = b(s32[2]) custom-call(b1), custom_call_target="CallBack_AddOne",
      output_to_operand_aliasing={{}: (0, {})},
      api_version=API_VERSION_STATUS_RETURNING
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
    b4 = b(s32[2]) custom-call(b3), custom_call_target="CallBack_AddOne",
      output_to_operand_aliasing={{}: (0, {})},
      api_version=API_VERSION_STATUS_RETURNING
    b5 = b(s32[2]) custom-call(b4), custom_call_target="CallBack_AddOne",
      output_to_operand_aliasing={{}: (0, {})},
      api_version=API_VERSION_STATUS_RETURNING
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

}  // anonymous namespace
}  // namespace xla
