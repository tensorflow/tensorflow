// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/runtime/tensor_buffer_conversion.h"

#include <cstdint>
#include <cstring>
#include <memory>

#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_environment.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/cc/litert_layout.h"
#include "tensorflow/lite/experimental/litert/core/environment.h"
#include "tensorflow/lite/experimental/litert/runtime/tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"

#if LITERT_HAS_OPENGL_SUPPORT
#include "tensorflow/lite/delegates/gpu/gl/egl_environment.h"
#endif  // LITERT_HAS_OPENGL_SUPPORT

namespace {

constexpr const float kTensorData[] = {10, 20, 30, 40};

constexpr const int32_t kTensorDimensions[] = {sizeof(kTensorData) /
                                               sizeof(kTensorData[0])};

constexpr const LiteRtRankedTensorType kTensorType = {
    /*.element_type=*/kLiteRtElementTypeFloat32,
    ::litert::BuildLayout(kTensorDimensions)};

TEST(TensorBufferConversionTest, HostToGl) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr litert_env,
                              LiteRtEnvironmentT::CreateWithOptions({}));

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferT::Ptr tensor_buffer_host,
      LiteRtTensorBufferT::CreateManaged(kLiteRtTensorBufferTypeHostMemory,
                                         kTensorType, sizeof(kTensorData)));
  // Write data to the host memory.
  LITERT_ASSERT_OK_AND_ASSIGN(void* host_memory,
                              tensor_buffer_host->GetHostBuffer());
  std::memcpy(host_memory, kTensorData, sizeof(kTensorData));

#if LITERT_HAS_OPENGL_SUPPORT
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferT::Ptr tensor_buffer_gl,
      litert::internal::TensorBufferConvertTo(kLiteRtTensorBufferTypeGlBuffer,
                                              *tensor_buffer_host));

  // Ensure that data was copied correctly from host to GL.
  LITERT_ASSERT_OK_AND_ASSIGN(void* host_gl, tensor_buffer_gl->Lock());
  ASSERT_EQ(std::memcmp(host_gl, kTensorData, sizeof(kTensorData)), 0);
#else
  // Since GL support is not enabled, the conversion should fail.
  EXPECT_FALSE(litert::internal::TensorBufferConvertTo(
      kLiteRtTensorBufferTypeGlBuffer, *tensor_buffer_host));
#endif
}

#if LITERT_HAS_OPENGL_SUPPORT && LITERT_HAS_AHWB_SUPPORT
TEST(TensorBufferConversionTest, GlToAhwb) {
  std::unique_ptr<tflite::gpu::gl::EglEnvironment> env;
  ASSERT_TRUE(tflite::gpu::gl::EglEnvironment::NewEglEnvironment(&env).ok());

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferT::Ptr tensor_buffer_gl,
      LiteRtTensorBufferT::CreateManaged(kLiteRtTensorBufferTypeGlBuffer,
                                         kTensorType, sizeof(kTensorData)));
  // Write data to the GL buffer.
  LITERT_ASSERT_OK_AND_ASSIGN(litert::internal::GlBuffer * gl_buffer,
                              tensor_buffer_gl->GetGlBuffer());
  LITERT_ASSERT_OK_AND_ASSIGN(float* data, gl_buffer->Lock<float>());
  std::memcpy(data, kTensorData, sizeof(kTensorData));
  gl_buffer->Unlock<float>();

  // Convert.
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferT::Ptr tensor_buffer_ahwb,
      litert::internal::TensorBufferConvertTo(kLiteRtTensorBufferTypeAhwb,
                                              *tensor_buffer_gl));
  // Ensure that data was copied correctly from Gl to Ahwb.
  LITERT_ASSERT_OK_AND_ASSIGN(void* host_ahwb, tensor_buffer_ahwb->Lock());
  ASSERT_EQ(std::memcmp(host_ahwb, kTensorData, sizeof(kTensorData)), 0);
}
#endif  // LITERT_HAS_OPENGL_SUPPORT && LITERT_HAS_AHWB_SUPPORT

#if LITERT_HAS_OPENGL_SUPPORT && LITERT_HAS_OPENCL_SUPPORT
TEST(TensorBufferConversionTest, GlToCl) {
  // Environment setup.
  LITERT_ASSERT_OK_AND_ASSIGN(LiteRtEnvironmentT::Ptr litert_env,
                              LiteRtEnvironmentT::CreateWithOptions({}));
  if (!litert::internal::OpenClBuffer::IsSupported()) {
    GTEST_SKIP() << "OpenCL buffers are not supported on this platform; "
                    "skipping the test";
  }

  std::unique_ptr<tflite::gpu::gl::EglEnvironment> env;
  ASSERT_TRUE(tflite::gpu::gl::EglEnvironment::NewEglEnvironment(&env).ok());

  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferT::Ptr tensor_buffer_gl,
      LiteRtTensorBufferT::CreateManaged(kLiteRtTensorBufferTypeGlBuffer,
                                         kTensorType, sizeof(kTensorData)));
  // Write data to the GL buffer.
  LITERT_ASSERT_OK_AND_ASSIGN(litert::internal::GlBuffer * gl_buffer,
                              tensor_buffer_gl->GetGlBuffer());
  LITERT_ASSERT_OK_AND_ASSIGN(float* data, gl_buffer->Lock<float>());
  std::memcpy(data, kTensorData, sizeof(kTensorData));
  gl_buffer->Unlock<float>();

  // Convert.
  LITERT_ASSERT_OK_AND_ASSIGN(
      LiteRtTensorBufferT::Ptr tensor_buffer_cl,
      litert::internal::TensorBufferConvertTo(kLiteRtTensorBufferTypeOpenCl,
                                              *tensor_buffer_gl));

  // Ensure that data was copied correctly from Gl to CL.
  LITERT_ASSERT_OK_AND_ASSIGN(void* host_cl, tensor_buffer_cl->Lock());
  ASSERT_EQ(std::memcmp(host_cl, kTensorData, sizeof(kTensorData)), 0);
}
#endif  // LITERT_HAS_OPENGL_SUPPORT && LITERT_HAS_OPENCL_SUPPORT

}  // namespace
