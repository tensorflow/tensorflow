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

#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

class CustomCallTest : public ClientLibraryTestBase {};

bool is_invoked_called = false;
void Callback_IsInvoked(CUstream /*stream*/, void** /*buffers*/,
                        const char* /*opaque*/, size_t /*opaque_len*/) {
  is_invoked_called = true;
}
XLA_REGISTER_CUSTOM_CALL_TARGET(Callback_IsInvoked, "CUDA");

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

void Callback_Memcpy(CUstream stream, void** buffers, const char* /*opaque*/,
                     size_t /*opaque_len*/) {
  void* src = buffers[0];
  void* dst = buffers[1];
  auto err = cudaMemcpyAsync(dst, src, /*count=*/sizeof(float) * 128,
                             cudaMemcpyDeviceToDevice, stream);
  ASSERT_EQ(err, cudaSuccess);
}
XLA_REGISTER_CUSTOM_CALL_TARGET(Callback_Memcpy, "CUDA");
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
void Callback_Opaque(CUstream /*stream*/, void** /*buffers*/,
                     const char* opaque, size_t opaque_len) {
  std::string opaque_str(opaque, opaque_len);
  ASSERT_EQ(opaque_str, kExpectedOpaque);
}
XLA_REGISTER_CUSTOM_CALL_TARGET(Callback_Opaque, "CUDA");
TEST_F(CustomCallTest, Opaque) {
  XlaBuilder b(TestName());
  CustomCall(&b, "Callback_Opaque", /*operands=*/{},
             ShapeUtil::MakeShape(F32, {}), kExpectedOpaque);
  TF_ASSERT_OK(Execute(&b, {}).status());
}

void Callback_SubBuffers(CUstream stream, void** buffers,
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
  cudaMemcpyAsync(buffers[4], buffers[3], 8 * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(buffers[5], buffers[0], 128 * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(buffers[6], buffers[1], 256 * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(buffers[7], buffers[2], 1024 * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
}
XLA_REGISTER_CUSTOM_CALL_TARGET(Callback_SubBuffers, "CUDA");
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

void Callback_TupleSelect(CUstream stream, void** buffers,
                          const char* /*opaque*/, size_t /*opaque_len*/) {
  // Set the two output leaf buffers equal to the two input leaf buffers.
  cudaMemcpyAsync(buffers[2], buffers[0], 10 * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(buffers[3], buffers[1], 10 * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
}
XLA_REGISTER_CUSTOM_CALL_TARGET(Callback_TupleSelect, "CUDA");
// Tuple-shaped select is a case where XLA can't know all buffer assignments
// statically ahead of time and has to walk the on-device tuple sub-buffers.
TEST_F(CustomCallTest, TupleSelect) {
  XlaBuilder b(TestName());
  auto tuple_shape = ShapeUtil::MakeTupleShape({
      ShapeUtil::MakeShape(F32, {10}),
      ShapeUtil::MakeShape(F32, {10}),
  });
  auto p0 = AddParam(LiteralUtil::CreateR0(false), &b);
  auto p1 =
      AddParam(LiteralUtil::MakeTupleOwned(
                   LiteralUtil::CreateR1<float>(std::vector<float>(10, 1.0f)),
                   LiteralUtil::CreateR1<float>(std::vector<float>(10, 2.0f))),
               &b);
  auto p2 =
      AddParam(LiteralUtil::MakeTupleOwned(
                   LiteralUtil::CreateR1<float>(std::vector<float>(10, 10.0f)),
                   LiteralUtil::CreateR1<float>(std::vector<float>(10, 20.0f))),
               &b);
  auto cc = CustomCall(&b, "Callback_TupleSelect",
                       /*operands=*/{Select(p0, p1, p2)}, tuple_shape,
                       /*opaque=*/"");

  // Do a tuple-select on the custom-call result to ensure that the custom-call
  // sets its output tuple index buffers.
  Select(p0, p1, cc);
  TF_ASSERT_OK_AND_ASSIGN(auto result, ComputeAndTransfer(&b, {}));
  EXPECT_THAT(result.data<float>({0}), ::testing::Each(10));
  EXPECT_THAT(result.data<float>({1}), ::testing::Each(20));
}

}  // anonymous namespace
}  // namespace xla
