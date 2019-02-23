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
#include "tensorflow/lite/tools/evaluation/evaluation_stage.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/lite/tools/evaluation/evaluation_stage_factory.h"
#include "tensorflow/lite/tools/evaluation/identity_stage.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"

namespace tflite {
namespace evaluation {
namespace {

using ::tensorflow::DataType;
using ::tensorflow::Tensor;

constexpr char kIdentityStageName[] = "identity_stage";
constexpr char kInputTypeName[] = "type";
constexpr char kInputTensorsName[] = "in";
constexpr char kOutputTensorsName[] = "out";
constexpr char kInitializerMapping[] = "INPUT_TYPE:type";
constexpr char kInputMapping[] = "INPUT_TENSORS:in";
constexpr char kOutputMapping[] = "OUTPUT_TENSORS:out";

EvaluationStageConfig GetIdentityStageConfig() {
  EvaluationStageConfig config;
  config.set_name(kIdentityStageName);
  config.mutable_specification()->set_process_class(IDENTITY);
  config.add_initializers(kInitializerMapping);
  config.add_inputs(kInputMapping);
  config.add_outputs(kOutputMapping);
  return config;
}

TEST(EvaluationStage, IncompleteConfig) {
  // Construct
  EvaluationStageConfig config = GetIdentityStageConfig();
  config.clear_inputs();
  std::unique_ptr<EvaluationStage> stage_ptr =
      CreateEvaluationStageFromConfig(config);
  // Initialize
  absl::flat_hash_map<std::string, void*> object_map;
  DataType input_type = tensorflow::DT_FLOAT;
  object_map[kInputTypeName] = &input_type;
  EXPECT_FALSE(stage_ptr->Init(object_map));
}

TEST(EvaluationStage, IncorrectlyFormattedConfig) {
  // Construct
  EvaluationStageConfig config = GetIdentityStageConfig();
  config.clear_initializers();
  config.add_initializers("INPUT_TYPE-type");
  std::unique_ptr<EvaluationStage> stage_ptr =
      CreateEvaluationStageFromConfig(config);
  // Initialize
  absl::flat_hash_map<std::string, void*> object_map;
  DataType input_type = tensorflow::DT_FLOAT;
  object_map[kInputTypeName] = &input_type;
  EXPECT_FALSE(stage_ptr->Init(object_map));
}

TEST(EvaluationStage, ConstructFromConfig_UnknownProcess) {
  // Construct
  EvaluationStageConfig config = GetIdentityStageConfig();
  config.mutable_specification()->clear_process_class();
  std::unique_ptr<EvaluationStage> stage_ptr =
      CreateEvaluationStageFromConfig(config);
  EXPECT_EQ(stage_ptr.get(), nullptr);
}

TEST(EvaluationStage, NoInitializer) {
  // Construct
  EvaluationStageConfig config = GetIdentityStageConfig();
  std::unique_ptr<EvaluationStage> stage_ptr =
      CreateEvaluationStageFromConfig(config);
  // Initialize
  absl::flat_hash_map<std::string, void*> object_map;
  EXPECT_FALSE(stage_ptr->Init(object_map));
}

TEST(EvaluationStage, NoInputs) {
  // Construct
  EvaluationStageConfig config = GetIdentityStageConfig();
  std::unique_ptr<EvaluationStage> stage_ptr =
      CreateEvaluationStageFromConfig(config);
  // Initialize
  absl::flat_hash_map<std::string, void*> object_map;
  DataType input_type = tensorflow::DT_FLOAT;
  object_map[kInputTypeName] = &input_type;
  EXPECT_TRUE(stage_ptr->Init(object_map));

  // Run
  EvaluationStageMetrics metrics;
  EXPECT_FALSE(stage_ptr->Run(object_map, metrics));
}

TEST(EvaluationStage, ExpectedIdentityOutput) {
  // Construct
  EvaluationStageConfig config = GetIdentityStageConfig();
  std::unique_ptr<EvaluationStage> stage_ptr =
      CreateEvaluationStageFromConfig(config);
  // Initialize
  absl::flat_hash_map<std::string, void*> object_map;
  DataType input_type = tensorflow::DT_FLOAT;
  object_map[kInputTypeName] = &input_type;
  EXPECT_TRUE(stage_ptr->Init(object_map));

  // Input Data
  float float_value = 5.6f;
  Tensor input_tensor(float_value);
  std::vector<Tensor> input_tensors = {input_tensor};
  // Run
  object_map[kInputTensorsName] = &input_tensors;
  EvaluationStageMetrics metrics;
  EXPECT_TRUE(stage_ptr->Run(object_map, metrics));

  // Check output
  std::vector<Tensor>* output_tensors_ptr =
      static_cast<std::vector<Tensor>*>(object_map[kOutputTensorsName]);
  EXPECT_TRUE(output_tensors_ptr != nullptr);
  EXPECT_FLOAT_EQ(output_tensors_ptr->at(0).scalar<float>()(), float_value);
  EXPECT_GE(metrics.total_latency_ms(), 0);
}

}  // namespace
}  // namespace evaluation
}  // namespace tflite
