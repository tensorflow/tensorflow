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

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/absl_log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_tensor_buffer_types.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_compiled_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_environment.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/core/model/model_buffer.h"
#include "tensorflow/lite/experimental/litert/runtime/ahwb_buffer.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"
#include "tensorflow/lite/experimental/litert/test/testdata/simple_model_test_vectors.h"

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

}  // namespace
}  // namespace litert
