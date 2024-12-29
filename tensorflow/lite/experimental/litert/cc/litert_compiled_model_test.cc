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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/log/absl_log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_tensor_buffer.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/testdata/simple_model_test_vectors.h"

using testing::FloatNear;
using testing::Pointwise;

namespace litert {
namespace {

TEST(CompiledModelTest, Basic) {
  auto model = testing::LoadTestFileModel(kModelFileName);
  ASSERT_TRUE(model);

  auto res_compiled_model = CompiledModel::Create(model);
  ASSERT_TRUE(res_compiled_model) << "Failed to initialize CompiledModel";

  auto& compiled_model = *res_compiled_model;
  auto signatures = model.GetSignatures().Value();
  EXPECT_EQ(signatures.size(), 1);

  auto signature_key = signatures[0].Key();
  EXPECT_EQ(signature_key, Model::DefaultSignatureKey());
  size_t signature_index = 0;

  auto input_buffers_res = compiled_model.CreateInputBuffers(signature_index);
  EXPECT_TRUE(input_buffers_res);
  auto& input_buffers = *input_buffers_res;

  auto output_buffers_res = compiled_model.CreateOutputBuffers(signature_index);
  EXPECT_TRUE(output_buffers_res);
  auto& output_buffers = *output_buffers_res;

  // Fill model inputs.
  auto input_names = signatures[0].InputNames();
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
  auto output_names = signatures[0].OutputNames();
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_EQ(output_names.at(0), "tfl.add");
  {
    auto lock_and_addr =
        litert::TensorBufferScopedLock::Create<const float>(output_buffers[0]);
    ASSERT_TRUE(lock_and_addr);
    auto output = absl::MakeSpan(lock_and_addr->second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

TEST(CompiledModelTest, RunWithInputOutputMap) {
  auto model = testing::LoadTestFileModel(kModelFileName);
  ASSERT_TRUE(model);

  auto res_compiled_model = CompiledModel::Create(model);
  ASSERT_TRUE(res_compiled_model) << "Failed to initialize CompiledModel";

  auto& compiled_model = *res_compiled_model;
  auto signatures = model.GetSignatures().Value();
  EXPECT_EQ(signatures.size(), 1);

  auto signature_key = signatures[0].Key();
  EXPECT_EQ(signature_key, Model::DefaultSignatureKey());
  size_t signature_index = 0;

  auto input_buffers_res = compiled_model.CreateInputBuffers(signature_index);
  EXPECT_TRUE(input_buffers_res);
  auto& input_buffers = *input_buffers_res;

  auto output_buffers_res = compiled_model.CreateOutputBuffers(signature_index);
  EXPECT_TRUE(output_buffers_res);
  auto& output_buffers = *output_buffers_res;

  // Fill model inputs.
  auto input_names = signatures[0].InputNames();
  EXPECT_EQ(input_names.size(), 2);
  EXPECT_EQ(input_names.at(0), "arg0");
  EXPECT_EQ(input_names.at(1), "arg1");
  ASSERT_TRUE(input_buffers[0].Write<float>(
      absl::MakeConstSpan(kTestInput0Tensor, kTestInput0Size)));
  ASSERT_TRUE(input_buffers[1].Write<float>(
      absl::MakeConstSpan(kTestInput1Tensor, kTestInput1Size)));
  absl::flat_hash_map<absl::string_view, TensorBuffer> input_map;
  input_map["arg0"] = std::move(input_buffers[0]);
  input_map["arg1"] = std::move(input_buffers[1]);

  auto output_names = signatures[0].OutputNames();
  EXPECT_EQ(output_names.size(), 1);
  EXPECT_EQ(output_names.at(0), "tfl.add");
  absl::flat_hash_map<absl::string_view, TensorBuffer> output_map;
  output_map["tfl.add"] = std::move(output_buffers[0]);

  // Execute model.
  compiled_model.Run(signature_index, input_map, output_map);

  // Check model output.
  {
    auto lock_and_addr = litert::TensorBufferScopedLock::Create<const float>(
        output_map["tfl.add"]);
    ASSERT_TRUE(lock_and_addr);
    auto output = absl::MakeSpan(lock_and_addr->second, kTestOutputSize);
    for (auto i = 0; i < kTestOutputSize; ++i) {
      ABSL_LOG(INFO) << "Result: " << output[i] << "\t" << kTestOutputTensor[i];
    }
    EXPECT_THAT(output, Pointwise(FloatNear(1e-5), kTestOutputTensor));
  }
}

}  // namespace
}  // namespace litert
