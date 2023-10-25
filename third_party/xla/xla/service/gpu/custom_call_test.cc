/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <sstream>
#include <string>

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#define PLATFORM "CUDA"
#elif TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_runtime.h"
#define PLATFORM "ROCM"
#endif

#include "xla/client/lib/constants.h"
#include "xla/client/xla_builder.h"
#include "xla/runtime/custom_call.h"
#include "xla/runtime/custom_call_registry.h"
#include "xla/runtime/executable.h"
#include "xla/runtime/memref_view.h"
#include "xla/runtime/module.h"
#include "xla/runtime/module_registry.h"
#include "xla/service/custom_call_status.h"
#include "xla/service/custom_call_target_registry.h"
#include "xla/service/gpu/runtime/custom_call_registry.h"
#include "xla/service/gpu/runtime/support.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/test_helpers.h"
#include "xla/tests/client_library_test_base.h"
#include "tsl/lib/core/status_test_util.h"

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
namespace {

class CustomCallTest : public ClientLibraryTestBase {};

bool is_invoked_called = false;
void Callback_IsInvoked(se::gpu::GpuStreamHandle /*stream*/, void** /*buffers*/,
                        const char* /*opaque*/, size_t /*opaque_len*/) {
  is_invoked_called = true;
}
XLA_REGISTER_CUSTOM_CALL_TARGET(Callback_IsInvoked, PLATFORM);

TEST_F(CustomCallTest, IsInvoked) {
  XlaBuilder b(TestName());
  CustomCall(&b, "Callback_IsInvoked", /*operands=*/{},
             ShapeUtil::MakeShape(F32, {}),
             /*opaque=*/"");
  EXPECT_FALSE(is_invoked_called);
  TF_ASSERT_OK(Execute(&b, {}).status());
  EXPECT_TRUE(is_invoked_called);
}

TEST_F(CustomCallTest, UnknownTarget) {
  XlaBuilder b(TestName());
  CustomCall(&b, "UnknownTarget", /*operands=*/{},
             ShapeUtil::MakeShape(F32, {}),
             /*opaque=*/"");
  ASSERT_FALSE(Execute(&b, {}).ok());
}
void Callback_Memcpy(se::gpu::GpuStreamHandle stream, void** buffers,
                     const char* /*opaque*/, size_t /*opaque_len*/) {
  void* src = buffers[0];
  void* dst = buffers[1];
  auto err = gpuMemcpyAsync(dst, src, /*count=*/sizeof(float) * 128,
                            gpuMemcpyDeviceToDevice, stream);
  ASSERT_EQ(err, gpuSuccess);
}
XLA_REGISTER_CUSTOM_CALL_TARGET(Callback_Memcpy, PLATFORM);
TEST_F(CustomCallTest, Memcpy) {
  XlaBuilder b(TestName());
  CustomCall(&b, "Callback_Memcpy",
             /*operands=*/{Broadcast(ConstantR0WithType(&b, F32, 42.0), {128})},
             ShapeUtil::MakeShape(F32, {128}), /*opaque=*/"");
  TF_ASSERT_OK_AND_ASSIGN(auto result, ExecuteAndTransfer(&b, {}));
  EXPECT_THAT(result.data<float>(), ::testing::Each(42));
}

// Check that opaque handles nulls within the string.
std::string& kExpectedOpaque = *new std::string("abc\0def", 7);
void Callback_Opaque(se::gpu::GpuStreamHandle /*stream*/, void** /*buffers*/,
                     const char* opaque, size_t opaque_len) {
  std::string opaque_str(opaque, opaque_len);
  ASSERT_EQ(opaque_str, kExpectedOpaque);
}
XLA_REGISTER_CUSTOM_CALL_TARGET(Callback_Opaque, PLATFORM);
TEST_F(CustomCallTest, Opaque) {
  XlaBuilder b(TestName());
  CustomCall(&b, "Callback_Opaque", /*operands=*/{},
             ShapeUtil::MakeShape(F32, {}), kExpectedOpaque);
  TF_ASSERT_OK(Execute(&b, {}).status());
}

void Callback_SubBuffers(se::gpu::GpuStreamHandle stream, void** buffers,
                         const char* /*opaque*/, size_t /*opaque_len*/) {
  // `buffers` is a flat array containing device pointers to the following.
  //
  //  0:  param 0 at tuple index {0}, shape f32[128]
  //  1:  param 0 at tuple index {1}, shape f32[256]
  //  2:  param 1 at tuple index {0}, shape f32[1024]
  //  3:  param 1 at tuple index {1}, shape f32[8]
  //  4:  result at tuple index {0}, shape f32[8]
  //  5:  result at tuple index {1, 0}, shape f32[128]
  //  6:  result at tuple index {1, 1}, shape f32[256]
  //  7:  result at tuple index {2}, shape f32[1024]
  //

  // Set output leaf buffers, copying data from the corresponding same-sized
  // inputs.
  auto err = gpuMemcpyAsync(buffers[4], buffers[3], 8 * sizeof(float),
                            gpuMemcpyDeviceToDevice, stream);
  ASSERT_EQ(err, gpuSuccess);
  err = gpuMemcpyAsync(buffers[5], buffers[0], 128 * sizeof(float),
                       gpuMemcpyDeviceToDevice, stream);
  ASSERT_EQ(err, gpuSuccess);
  err = gpuMemcpyAsync(buffers[6], buffers[1], 256 * sizeof(float),
                       gpuMemcpyDeviceToDevice, stream);
  ASSERT_EQ(err, gpuSuccess);
  err = gpuMemcpyAsync(buffers[7], buffers[2], 1024 * sizeof(float),
                       gpuMemcpyDeviceToDevice, stream);
  ASSERT_EQ(err, gpuSuccess);
}
XLA_REGISTER_CUSTOM_CALL_TARGET(Callback_SubBuffers, PLATFORM);
TEST_F(CustomCallTest, SubBuffers) {
  XlaBuilder b(TestName());
  CustomCall(&b, "Callback_SubBuffers", /*operands=*/
             {
                 Tuple(&b,
                       {
                           Broadcast(ConstantR0WithType(&b, F32, 1), {128}),
                           Broadcast(ConstantR0WithType(&b, F32, 2), {256}),
                       }),
                 Tuple(&b,
                       {
                           Broadcast(ConstantR0WithType(&b, F32, 3), {1024}),
                           Broadcast(ConstantR0WithType(&b, F32, 4), {8}),
                       }),
             },
             ShapeUtil::MakeTupleShape({
                 ShapeUtil::MakeShape(F32, {8}),
                 ShapeUtil::MakeTupleShape({
                     ShapeUtil::MakeShape(F32, {128}),
                     ShapeUtil::MakeShape(F32, {256}),
                 }),
                 ShapeUtil::MakeShape(F32, {1024}),
             }),
             /*opaque=*/"");
  TF_ASSERT_OK_AND_ASSIGN(auto result, ExecuteAndTransfer(&b, {}));
  EXPECT_THAT(result.data<float>({0}), ::testing::Each(4));
  EXPECT_THAT(result.data<float>({1, 0}), ::testing::Each(1));
  EXPECT_THAT(result.data<float>({1, 1}), ::testing::Each(2));
  EXPECT_THAT(result.data<float>({2}), ::testing::Each(3));
}

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

std::ostream& operator<<(std::ostream& s, const TokenTestCase& tc) {
  s << tc.input << "x" << tc.output << "x" << tc.opaque;
  return s;
}

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
      public ClientLibraryTestBase {
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

TEST_P(CustomCallTokensTest, TokensTest) {
  const TokenTestCase& tc = GetParam();

  XlaBuilder b("CustomCallTokens");

  std::istringstream input(tc.input);
  std::istringstream output(tc.output);
  std::vector<XlaOp> call_inputs = BuildInputs(b, input);
  std::vector<Shape> call_output = BuildOutputType(output);
  ASSERT_EQ(call_output.size(), 1);

  CustomCall(&b, "Callback_Tokens", call_inputs, call_output.front(),
             tc.opaque);
  TF_ASSERT_OK(Execute(&b, {}).status());
}

INSTANTIATE_TEST_CASE_P(CustomCallTokens, CustomCallTokensTest,
                        ::testing::ValuesIn(GetTokenTestCases()));

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
  TF_ASSERT_OK(Execute(&b, {}).status());
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
  auto status = Execute(&b, {}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  EXPECT_THAT(status.message(), ::testing::HasSubstr("Failed"));
}

//===----------------------------------------------------------------------===//
// Custom calls based on XLA runtime modules.
//===----------------------------------------------------------------------===//

struct TestModule : runtime::StatelessModule {
  TestModule() : StatelessModule("TestModule") {}

  // Check that we can use absl::Status to return errors back to the caller.
  static absl::Status AlwaysFail(runtime::StridedMemrefView arg) {
    return absl::InternalError("Uh oh, too bad");
  }

  // Check that we can get access to the stream and launch on device.
  static absl::Status Memcpy(const ServiceExecutableRunOptions* run_options,
                             runtime::FlatMemrefView src,
                             runtime::FlatMemrefView dst) {
    se::DeviceMemoryBase src_mem(src.data);
    se::DeviceMemoryBase dst_mem(dst.data);

    if (src.size_in_bytes != dst.size_in_bytes) {
      return absl::InternalError("Size in bytes must match");
    }

    run_options->stream()->ThenMemcpyD2D(&dst_mem, src_mem, src.size_in_bytes);
    return absl::OkStatus();
  }

  // Write bindings for custom calls and register with runtime.
  void Export(runtime::DynamicCustomCallRegistry& registry) const final {
    registry.Register(runtime::CustomCall::Bind("test.always_fail")
                          .Arg<runtime::StridedMemrefView>()
                          .To(AlwaysFail));

    registry.Register(runtime::CustomCall::Bind("test.memcpy")
                          .UserData<const ServiceExecutableRunOptions*>()
                          .Arg<runtime::FlatMemrefView>()
                          .Arg<runtime::FlatMemrefView>()
                          .To(Memcpy));
  }
};

XLA_REGISTER_RUNTIME_MODULE(std::make_unique<TestModule>());

TEST_F(CustomCallTest, ExportedAlwaysFail) {
  // TODO(ezhulenev): Remove once XLA runtime is enabled by default.
  mutable_debug_options()->set_xla_gpu_enable_xla_runtime_executable(true);

  XlaBuilder b(TestName());
  CustomCall(&b, "test.always_fail", /*operands=*/{},
             ShapeUtil::MakeShape(F32, {}), /*opaque=*/"",
             /*has_side_effect=*/false,
             /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
             /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
             /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);
  auto status = Execute(&b, {}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  VLOG(0) << status.message();
  EXPECT_THAT(status.message(), ::testing::HasSubstr("Uh oh, too bad"));
}

TEST_F(CustomCallTest, ExportedMemcpy) {
  // TODO(ezhulenev): Remove once XLA runtime is enabled by default.
  mutable_debug_options()->set_xla_gpu_enable_xla_runtime_executable(true);

  XlaBuilder b(TestName());
  CustomCall(&b, "test.memcpy",
             /*operands=*/{Broadcast(ConstantR0WithType(&b, F32, 42.0), {128})},
             ShapeUtil::MakeShape(F32, {128}), /*opaque=*/"",
             /*has_side_effect=*/false,
             /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
             /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
             /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);
  TF_ASSERT_OK_AND_ASSIGN(auto result, ExecuteAndTransfer(&b, {}));
  EXPECT_THAT(result.data<float>(), ::testing::Each(42));
}

//===----------------------------------------------------------------------===//
// XLA runtime custom calls provides type-safe custom call API
//===----------------------------------------------------------------------===//

// WARNING: We currently rely on a magic custom call prefix `__gpu$` to detect
// "internal" custom calls that linked statically into the binary. Without this
// prefix custom calls expected to be registered as XLA:FFI custom calls, and
// this is not yet fully supported.
//
// TODO(ezhulenev): Unify runtime custom calls and XLA:FFI.

// (1) Declare custom call implementations as static functions.

static absl::Status AlwaysFailImpl(runtime::StridedMemrefView arg) {
  return absl::InternalError("Uh oh, too bad");
}

static absl::Status MemcpyImpl(const ServiceExecutableRunOptions* run_options,
                               runtime::StridedMemrefView src,
                               runtime::StridedMemrefView dst) {
  auto src_mem = gpu::GetDeviceAddress(src);
  auto dst_mem = gpu::GetDeviceAddress(dst);
  run_options->stream()->ThenMemcpyD2D(&dst_mem, src_mem, src_mem.size());
  return absl::OkStatus();
}

// (2) Declare custom call binding signature. At compile time we check that
// declared signature matches function handlers, and at run time we check that
// passed arguments match the signature (number of arguments and their types).

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    AlwaysFail, AlwaysFailImpl, runtime::CustomCall::RuntimeChecks::kDefault,
    runtime::CustomCall::Bind("__gpu$xla.gpu.ext.always_fail")
        .Arg<runtime::StridedMemrefView>()  // arg
);

XLA_RUNTIME_DEFINE_CUSTOM_CALL(
    Memcpy, MemcpyImpl, runtime::CustomCall::RuntimeChecks::kDefault,
    runtime::CustomCall::Bind("__gpu$xla.gpu.ext.memcpy")
        .UserData<const ServiceExecutableRunOptions*>()
        .Arg<runtime::StridedMemrefView>()  // src
        .Arg<runtime::StridedMemrefView>()  // dst
);

static void RegisterCustomCalls(runtime::DirectCustomCallRegistry& registry) {
  registry.Register("__gpu$xla.gpu.ext.always_fail", AlwaysFail);
  registry.Register("__gpu$xla.gpu.ext.memcpy", Memcpy);
}

XLA_GPU_REGISTER_RUNTIME_CUSTOM_CALL(RegisterCustomCalls);

TEST_F(CustomCallTest, RuntimeCustomCallAlwaysFail) {
  // TODO(ezhulenev): Remove once XLA runtime is enabled by default.
  mutable_debug_options()->set_xla_gpu_enable_xla_runtime_executable(true);

  XlaBuilder b(TestName());
  CustomCall(&b, "__gpu$xla.gpu.ext.always_fail", /*operands=*/{},
             ShapeUtil::MakeShape(F32, {}), /*opaque=*/"",
             /*has_side_effect=*/false,
             /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
             /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
             /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);
  auto status = Execute(&b, {}).status();
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
  VLOG(0) << status.message();
  EXPECT_THAT(status.message(), ::testing::HasSubstr("Uh oh, too bad"));
}

TEST_F(CustomCallTest, ExportedFfiMemcpy) {
  // TODO(ezhulenev): Remove once XLA runtime is enabled by default.
  mutable_debug_options()->set_xla_gpu_enable_xla_runtime_executable(true);

  XlaBuilder b(TestName());
  CustomCall(&b, "__gpu$xla.gpu.ext.memcpy",
             /*operands=*/{Broadcast(ConstantR0WithType(&b, F32, 42.0), {128})},
             ShapeUtil::MakeShape(F32, {128}), /*opaque=*/"",
             /*has_side_effect=*/false,
             /*output_operand_aliasing=*/{}, /*literal=*/nullptr,
             /*schedule=*/CustomCallSchedule::SCHEDULE_NONE,
             /*api_version=*/CustomCallApiVersion::API_VERSION_TYPED_FFI);
  TF_ASSERT_OK_AND_ASSIGN(auto result, ExecuteAndTransfer(&b, {}));
  EXPECT_THAT(result.data<float>(), ::testing::Each(42));
}

}  // anonymous namespace
}  // namespace xla
