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
#include "third_party/gpus/cuda/includes/cuda_headers/third_party/gpus/cuda/include/driver_types.h"
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
  CustomCall(&b, "UknownTarget", /*operands=*/{}, ShapeUtil::MakeShape(F32, {}),
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
  //   0: root tuple of param 0
  //   1:   param 0 at tuple index {0}, shape f32[128]
  //   2:   param 0 at tuple index {1}, shape f32[256]
  //   3: root tuple of param 1
  //   4:   param 1 at tuple index {0}, shape f32[1024]
  //   5:   param 1 at tuple index {1}, shape f32[8]
  //   6: root tuple of custom-call result
  //   7:   result at tuple index {0}, shape f32[8]
  //   8:   result at tuple index {1}, shape (f32[128], f32[256])
  //   9:     result at tuple index {1, 0}, shape f32[128]
  //  10:     result at tuple index {1, 1}, shape f32[256]
  //  11:   result at tuple index {2}, shape f32[1024]
  //
  // It's the contract of custom-call that the non-root pointers (i.e.
  // everything other than indices 0, 3, and 6) may be null, if XLA is unable to
  // analyze the program well enough to determine for sure what's in those
  // buffers.  For this simple example, all of the buffers should be non-null.

  // Check the param 0 tuple, namely that
  //
  //   (*buffers[0])[0] == buffers[1] and
  //   (*buffers[0])[1] == buffers[2].
  //
  // because buffers contains pointers to device memory, we have to retrieve
  // these values via cudaMemcpy.
  void* p0[2];
  cudaMemcpy(p0, buffers[0], 2 * sizeof(void*), cudaMemcpyDeviceToHost);
  ASSERT_EQ(p0[0], buffers[1]);
  ASSERT_EQ(p0[1], buffers[2]);

  // Check the param 1 tuple, namely that
  //
  //   (*buffers[3])[0] == buffers[4]
  //   (*buffers[3])[1] == buffers[5].
  void* p1[2];
  cudaMemcpy(p1, buffers[3], 2 * sizeof(void*), cudaMemcpyDeviceToHost);
  ASSERT_EQ(p1[0], buffers[4]);
  ASSERT_EQ(p1[1], buffers[5]);

  // We don't have an equivalent check for the output tuple (i.e. we don't check
  // (*buffers[6])[0] == buffers[7]) because it's up to us to set the tuple
  // as part of this custom-call.

  // Write the results.  First set the root tuple output buffer to {b7, b8,
  // b11}.
  void* root[3] = {buffers[7], buffers[8], buffers[11]};
  cudaMemcpy(buffers[6], root, 3 * sizeof(void*), cudaMemcpyHostToDevice);

  // Now set the sub-tuple output buffer at index 8 to {b9, b10}.
  void* sub_tuple[2] = {buffers[9], buffers[10]};
  cudaMemcpy(buffers[8], sub_tuple, 2 * sizeof(void*), cudaMemcpyDeviceToHost);

  // Now set output leaf buffers 7, 9, 10, and 11, copying data from the
  // corresponding same-sized inputs.
  cudaMemcpyAsync(buffers[7], buffers[5], 8 * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(buffers[9], buffers[1], 128 * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(buffers[10], buffers[2], 256 * sizeof(float),
                  cudaMemcpyDeviceToDevice, stream);
  cudaMemcpyAsync(buffers[11], buffers[4], 1024 * sizeof(float),
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

}  // anonymous namespace
}  // namespace xla
