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

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_gl_types.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_types.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_compiled_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_environment.h"
#include "tensorflow/lite/experimental/litert/cc/litert_event.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/core/model/model_buffer.h"
#include "tensorflow/lite/experimental/litert/runtime/ahwb_buffer.h"
#include "tensorflow/lite/experimental/litert/runtime/gl_buffer.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"
#include "tensorflow/lite/experimental/litert/test/testdata/simple_model_test_vectors.h"
#if LITERT_HAS_OPENGL_SUPPORT
#include "tensorflow/lite/delegates/gpu/gl/egl_environment.h"
#endif  // LITERT_HAS_OPENGL_SUPPORT

namespace litert {
namespace {

using ::testing::Eq;
using ::testing::FloatNear;
using ::testing::Pointwise;
using ::testing::SizeIs;

constexpr absl::string_view kNpuFile = kGoogleTensorModelFileName;
constexpr absl::string_view kTfliteFile = "simple_model_npu.tflite";
constexpr absl::string_view kDispatchLibraryDir = "/data/local/tmp";

TEST(CompiledModelTest, RunWithGoogleTensorModel) {
  if (!litert::internal::AhwbBuffer::IsSupported()) {
    GTEST_SKIP()
        << "The rest of this test is specific to Android devices with a "
           "GoogleTensor eTPU";
  }

  // Environment setup.
  const std::vector<litert::Environment::Option> environment_options = {
      litert::Environment::Option{
          litert::Environment::OptionTag::DispatchLibraryDir,
          kDispatchLibraryDir,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env,
                              litert::Environment::Create(environment_options));

  // Create Model.

  // TODO(gcarranza): Replace internal API with C++ API or single npu tflite
  // file.
  LITERT_ASSERT_OK_AND_ASSIGN(
      BufferRef<uint8_t> model_with_byte_code,
      internal::GetModelBufWithByteCode(testing::GetTestFilePath(kTfliteFile),
                                        testing::GetTestFilePath(kNpuFile)));

  LITERT_ASSERT_OK_AND_ASSIGN(Model model,
                              Model::CreateFromBuffer(model_with_byte_code));
  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(CompiledModel compiled_model,
                              CompiledModel::Create(env, model));

  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBuffer> input_buffers,
      compiled_model.CreateInputBuffers(model.DefaultSignatureKey()));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBuffer> output_buffers,
      compiled_model.CreateOutputBuffers(model.DefaultSignatureKey()));

  ASSERT_THAT(input_buffers, SizeIs(2));
  ASSERT_THAT(output_buffers, SizeIs(1));

  // Confirm input and output buffers are AHWB.
  EXPECT_THAT(*input_buffers[0].BufferType(), Eq(kLiteRtTensorBufferTypeAhwb));
  EXPECT_THAT(*input_buffers[1].BufferType(), Eq(kLiteRtTensorBufferTypeAhwb));
  EXPECT_THAT(*output_buffers[0].BufferType(), Eq(kLiteRtTensorBufferTypeAhwb));

  LITERT_ASSERT_OK(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  LITERT_ASSERT_OK(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Run compiled model.
  compiled_model.Run(model.DefaultSignatureKey(), input_buffers,
                     output_buffers);

  // Check model output.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(output_buffers[0]));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

TEST(CompiledModel, RunAsyncWithGoogleTensorModel) {
  if (!litert::internal::AhwbBuffer::IsSupported()) {
    GTEST_SKIP()
        << "The rest of this test is specific to Android devices with a "
           "GoogleTensor eTPU";
  }

  // Environment setup.
  const std::vector<litert::Environment::Option> environment_options = {
      litert::Environment::Option{
          litert::Environment::OptionTag::DispatchLibraryDir,
          kDispatchLibraryDir,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env,
                              litert::Environment::Create(environment_options));

  // Create Model.

  // TODO(gcarranza): Replace internal API with C++ API or single npu tflite
  // file.
  LITERT_ASSERT_OK_AND_ASSIGN(
      BufferRef<uint8_t> model_with_byte_code,
      internal::GetModelBufWithByteCode(testing::GetTestFilePath(kTfliteFile),
                                        testing::GetTestFilePath(kNpuFile)));

  LITERT_ASSERT_OK_AND_ASSIGN(Model model,
                              Model::CreateFromBuffer(model_with_byte_code));
  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(CompiledModel compiled_model,
                              CompiledModel::Create(env, model));

  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBuffer> input_buffers,
      compiled_model.CreateInputBuffers(model.DefaultSignatureKey()));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBuffer> output_buffers,
      compiled_model.CreateOutputBuffers(model.DefaultSignatureKey()));

  ASSERT_THAT(input_buffers, SizeIs(2));
  ASSERT_THAT(output_buffers, SizeIs(1));

  // Confirm input and output buffers are AHWB.
  EXPECT_THAT(*input_buffers[0].BufferType(), Eq(kLiteRtTensorBufferTypeAhwb));
  EXPECT_THAT(*input_buffers[1].BufferType(), Eq(kLiteRtTensorBufferTypeAhwb));
  EXPECT_THAT(*output_buffers[0].BufferType(), Eq(kLiteRtTensorBufferTypeAhwb));

  LITERT_ASSERT_OK(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  LITERT_ASSERT_OK(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Run compiled model.
  bool async;
  compiled_model.RunAsync(model.DefaultSignatureKey(), input_buffers,
                          output_buffers, async);
  // Since output buffers have events, async should be true.
  ASSERT_TRUE(async);

  // Check model output.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(output_buffers[0]));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

void FillGlBuffer1(LiteRtGLuint id, size_t size) {
#if LITERT_HAS_OPENGL_SUPPORT
  std::string shader_source = R"( #version 310 es
    precision highp float;
    layout(local_size_x = 1, local_size_y = 1) in;
    layout(std430, binding = 0) buffer Output {float elements[];} output_data;
    void main() {
      uint v = gl_GlobalInvocationID.x * 2u;
      output_data.elements[v] = float(v + 1u) / 1.0;
      output_data.elements[v + 1u] = float(v + 2u) / 1.0;
    })";
  GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
  const GLchar* sources[] = {shader_source.c_str()};
  glShaderSource(shader, 1, sources, nullptr);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glCompileShader(shader);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);

  GLuint to_buffer_program = glCreateProgram();
  glAttachShader(to_buffer_program, shader);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glDeleteShader(shader);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glLinkProgram(to_buffer_program);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);

  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, id);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glUseProgram(to_buffer_program);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glDispatchCompute(size / 2, 1, 1);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glDeleteProgram(to_buffer_program);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
#endif  // LITERT_HAS_OPENGL_SUPPORT
}

void FillGlBuffer2(LiteRtGLuint id, size_t size) {
#if LITERT_HAS_OPENGL_SUPPORT
  std::string shader_source = R"( #version 310 es
    precision highp float;
    layout(local_size_x = 1, local_size_y = 1) in;
    layout(std430, binding = 0) buffer Output {float elements[];} output_data;
    void main() {
      uint v = gl_GlobalInvocationID.x * 2u;
      output_data.elements[v] = float(v + 1u) / 0.1;
      output_data.elements[v + 1u] = float(v + 2u) / 0.1;
    })";
  GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
  const GLchar* sources[] = {shader_source.c_str()};
  glShaderSource(shader, 1, sources, nullptr);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glCompileShader(shader);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);

  GLuint to_buffer_program = glCreateProgram();
  glAttachShader(to_buffer_program, shader);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glDeleteShader(shader);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glLinkProgram(to_buffer_program);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);

  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, id);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glUseProgram(to_buffer_program);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glDispatchCompute(size / 2, 1, 1);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
  glDeleteProgram(to_buffer_program);
  ABSL_CHECK(glGetError() == GL_NO_ERROR);
#endif  // LITERT_HAS_OPENGL_SUPPORT
}

TEST(CompiledModel, RunAsyncWithGoogleTensorModelUseAhwbGlInterop) {
  if (!litert::internal::AhwbBuffer::IsSupported()) {
    GTEST_SKIP()
        << "The rest of this test is specific to Android devices with a "
           "GoogleTensor eTPU";
  }

  // Environment setup.
  const std::vector<litert::Environment::Option> environment_options = {
      litert::Environment::Option{
          litert::Environment::OptionTag::DispatchLibraryDir,
          kDispatchLibraryDir,
      },
  };
  LITERT_ASSERT_OK_AND_ASSIGN(Environment env,
                              litert::Environment::Create(environment_options));

  // Create Model.

  // TODO(gcarranza): Replace internal API with C++ API or single npu tflite
  // file.
  LITERT_ASSERT_OK_AND_ASSIGN(
      BufferRef<uint8_t> model_with_byte_code,
      internal::GetModelBufWithByteCode(testing::GetTestFilePath(kTfliteFile),
                                        testing::GetTestFilePath(kNpuFile)));

  LITERT_ASSERT_OK_AND_ASSIGN(Model model,
                              Model::CreateFromBuffer(model_with_byte_code));
  // Create CompiledModel.
  LITERT_ASSERT_OK_AND_ASSIGN(CompiledModel compiled_model,
                              CompiledModel::Create(env, model));

  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBuffer> input_buffers,
      compiled_model.CreateInputBuffers(model.DefaultSignatureKey()));
  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBuffer> output_buffers,
      compiled_model.CreateOutputBuffers(model.DefaultSignatureKey()));

  ASSERT_THAT(input_buffers, SizeIs(2));
  ASSERT_THAT(output_buffers, SizeIs(1));

  // Confirm input and output buffers are AHWB.
  EXPECT_THAT(*input_buffers[0].BufferType(), Eq(kLiteRtTensorBufferTypeAhwb));
  EXPECT_THAT(*input_buffers[1].BufferType(), Eq(kLiteRtTensorBufferTypeAhwb));
  EXPECT_THAT(*output_buffers[0].BufferType(), Eq(kLiteRtTensorBufferTypeAhwb));

  // TODO(gcarranza): Integrate with LiteRT Environment.
#if LITERT_HAS_OPENGL_SUPPORT
  std::unique_ptr<tflite::gpu::gl::EglEnvironment> egl_env;
  ASSERT_TRUE(
      tflite::gpu::gl::EglEnvironment::NewEglEnvironment(&egl_env).ok());
  LITERT_LOG(LITERT_INFO, "Initialized EGL environment");
#else
  LITERT_LOG(LITERT_INFO, "EGL environment not initialized");
#endif  // LITERT_HAS_OPENGL_SUPPORT

  // Write to input buffers on GPU.
  LITERT_ASSERT_OK_AND_ASSIGN(auto gl_buffer_1, input_buffers[0].GetGlBuffer());
  FillGlBuffer1(gl_buffer_1.id, 2);
  LITERT_ASSERT_OK_AND_ASSIGN(auto gl_buffer_2, input_buffers[1].GetGlBuffer());
  FillGlBuffer2(gl_buffer_2.id, 2);

  // Create EGL sync and fence before AHWB read.
  // TODO(gcarranza): Integrate into LiteRT C++ API.
  LITERT_ASSERT_OK_AND_ASSIGN(
      int native_fence, ::litert::internal::GlBuffer::CreateEglSyncAndFence());

  LITERT_ASSERT_OK_AND_ASSIGN(
      Event event_1,
      Event::CreateFromSyncFenceFd(native_fence, /*owns_fd=*/false));
  LITERT_ASSERT_OK_AND_ASSIGN(
      Event event_2,
      Event::CreateFromSyncFenceFd(native_fence, /*owns_fd=*/false));

  // Set event so that AHWB read is blocked by GPU write.
  input_buffers[0].SetEvent(std::move(event_1));
  input_buffers[1].SetEvent(std::move(event_2));

  // Run compiled model asynchronously.
  bool async;
  compiled_model.RunAsync(model.DefaultSignatureKey(), input_buffers,
                          output_buffers, async);
  // Since output buffers have events, async should be true.
  ASSERT_TRUE(async);

  // Check model output.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr,
        litert::TensorBufferScopedLock::Create<const float>(output_buffers[0]));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

}  // namespace
}  // namespace litert
