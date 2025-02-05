// Copyright 2024 Google LLC.
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

#include "tensorflow/lite/experimental/litert/cc/litert_compiled_model.h"

#include <cstring>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_environment.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"
#include "tensorflow/lite/experimental/litert/test/testdata/simple_model_test_vectors.h"

using testing::FloatNear;
using testing::Pointwise;

namespace litert {
namespace {

TEST(CompiledModelTest, Basic) {
  Model model = testing::LoadTestFileModel(kModelFileName);
  ASSERT_TRUE(model);

  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  LITERT_ASSERT_OK_AND_ASSIGN(CompiledModel compiled_model,
                              CompiledModel::Create(env, model));

  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<Signature> signatures,
                              model.GetSignatures());
  EXPECT_EQ(signatures.size(), 1);

  absl::string_view signature_key = signatures[0].Key();
  EXPECT_EQ(signature_key, Model::DefaultSignatureKey());
  size_t signature_index = 0;

  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBuffer> input_buffers,
      compiled_model.CreateInputBuffers(signature_index));

  LITERT_ASSERT_OK_AND_ASSIGN(
      std::vector<TensorBuffer> output_buffers,
      compiled_model.CreateOutputBuffers(signature_index));

  // Fill model inputs.
  std::vector<absl::string_view> input_names = signatures[0].InputNames();
  EXPECT_EQ(input_names.size(), 2);
  EXPECT_EQ(input_names.at(0), "arg0");
  EXPECT_EQ(input_names.at(1), "arg1");
  ASSERT_TRUE(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  ASSERT_TRUE(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));

  // Execute model.
  compiled_model.Run(signature_index, input_buffers, output_buffers);

  // Check model output.
  std::vector<absl::string_view> output_names = signatures[0].OutputNames();
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_EQ(output_names.at(0), "tfl.add");
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

TEST(CompiledModelTest, RunWithInputOutputMap) {
  Model model = testing::LoadTestFileModel(kModelFileName);

  LITERT_ASSERT_OK_AND_ASSIGN(Environment env, litert::Environment::Create({}));

  LITERT_ASSERT_OK_AND_ASSIGN(CompiledModel compiled_model,
                              CompiledModel::Create(env, model));

  LITERT_ASSERT_OK_AND_ASSIGN(std::vector<Signature> signatures,
                              model.GetSignatures());
  EXPECT_EQ(signatures.size(), 1);

  absl::string_view signature_key = signatures[0].Key();
  EXPECT_EQ(signature_key, Model::DefaultSignatureKey());

  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer input_buffer0,
      compiled_model.CreateInputBuffer(signature_key, "arg0"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer input_buffer1,
      compiled_model.CreateInputBuffer(signature_key, "arg1"));
  LITERT_ASSERT_OK_AND_ASSIGN(
      TensorBuffer output_buffer0,
      compiled_model.CreateOutputBuffer(signature_key, "tfl.add"));

  // Fill model inputs.
  std::vector<absl::string_view> input_names = signatures[0].InputNames();
  EXPECT_EQ(input_names.size(), 2);
  EXPECT_EQ(input_names.at(0), "arg0");
  EXPECT_EQ(input_names.at(1), "arg1");
  ASSERT_TRUE(input_buffer0.Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  ASSERT_TRUE(input_buffer1.Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));
  absl::flat_hash_map<absl::string_view, TensorBuffer> input_map;
  input_map["arg0"] = std::move(input_buffer0);
  input_map["arg1"] = std::move(input_buffer1);

  std::vector<absl::string_view> output_names = signatures[0].OutputNames();
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_EQ(output_names.at(0), "tfl.add");
  absl::flat_hash_map<absl::string_view, TensorBuffer> output_map;
  output_map["tfl.add"] = std::move(output_buffer0);

  // Execute model.
  compiled_model.Run(signature_key, input_map, output_map);

  // Check model output.
  {
    LITERT_ASSERT_OK_AND_ASSIGN(
        auto lock_and_addr, litert::TensorBufferScopedLock::Create<const float>(
                                output_map["tfl.add"]));
    auto output = absl::MakeSpan(lock_and_addr.second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

}  // namespace
}  // namespace litert
