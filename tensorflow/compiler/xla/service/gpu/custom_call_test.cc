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

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#define PLATFORM "CUDA"
#elif TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_runtime.h"
#define PLATFORM "ROCM"
#endif
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/stream_executor/gpu/gpu_types.h"

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
  gpuMemcpyAsync(buffers[4], buffers[3], 8 * sizeof(float),
                 gpuMemcpyDeviceToDevice, stream);
  gpuMemcpyAsync(buffers[5], buffers[0], 128 * sizeof(float),
                 gpuMemcpyDeviceToDevice, stream);
  gpuMemcpyAsync(buffers[6], buffers[1], 256 * sizeof(float),
                 gpuMemcpyDeviceToDevice, stream);
  gpuMemcpyAsync(buffers[7], buffers[2], 1024 * sizeof(float),
                 gpuMemcpyDeviceToDevice, stream);
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

}  // anonymous namespace
}  // namespace xla
