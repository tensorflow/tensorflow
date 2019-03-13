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
#include "tensorflow/lite/tools/evaluation/identity_stage.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_config.pb.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"

namespace tflite {
namespace evaluation {
namespace {

constexpr char kIdentityStageName[] = "identity_stage";
constexpr char kDefaultValueName[] = "default";
constexpr char kInputValueName[] = "in";
constexpr char kOutputValueName[] = "out";
constexpr char kInitializerMapping[] = "DEFAULT_VALUE:default";
constexpr char kInputMapping[] = "INPUT_VALUE:in";
constexpr char kOutputMapping[] = "OUTPUT_VALUE:out";

EvaluationStageConfig GetIdentityStageConfig() {
  IdentityStage_ENABLE();
  EvaluationStageConfig config;
  config.set_name(kIdentityStageName);
  config.mutable_specification()->set_process_class(IDENTITY);
  config.add_initializers(kInitializerMapping);
  config.add_inputs(kInputMapping);
  config.add_outputs(kOutputMapping);
  return config;
}

TEST(EvaluationStage, CreateFailsForMissingSpecification) {
  // Construct
  EvaluationStageConfig config;
  config.set_name(kIdentityStageName);
  std::unique_ptr<EvaluationStage> stage_ptr = EvaluationStage::Create(config);
  EXPECT_EQ(stage_ptr, nullptr);
}

TEST(EvaluationStage, StageEnableRequired) {
  // Construct
  EvaluationStageConfig config;
  config.set_name(kIdentityStageName);
  config.mutable_specification()->set_process_class(IDENTITY);
  config.add_initializers(kInitializerMapping);
  config.add_inputs(kInputMapping);
  config.add_outputs(kOutputMapping);
  config.clear_inputs();
  std::unique_ptr<EvaluationStage> stage_ptr = EvaluationStage::Create(config);
  EXPECT_EQ(stage_ptr, nullptr);
  IdentityStage_ENABLE();
  stage_ptr = EvaluationStage::Create(config);
  EXPECT_NE(stage_ptr, nullptr);
}

TEST(EvaluationStage, IncompleteConfig) {
  // Construct
  EvaluationStageConfig config = GetIdentityStageConfig();
  config.clear_inputs();
  std::unique_ptr<EvaluationStage> stage_ptr = EvaluationStage::Create(config);
  // Initialize
  absl::flat_hash_map<std::string, void*> object_map;
  float default_value = 1.0;
  object_map[kDefaultValueName] = &default_value;
  EXPECT_FALSE(stage_ptr->Init(object_map));
}

TEST(EvaluationStage, IncorrectlyFormattedConfig) {
  // Construct
  EvaluationStageConfig config = GetIdentityStageConfig();
  config.clear_initializers();
  config.add_initializers("DEFAULT_VALUE-default");
  std::unique_ptr<EvaluationStage> stage_ptr = EvaluationStage::Create(config);
  // Initialize
  absl::flat_hash_map<std::string, void*> object_map;
  float default_value = 1.0;
  object_map[kDefaultValueName] = &default_value;
  EXPECT_FALSE(stage_ptr->Init(object_map));
}

TEST(EvaluationStage, ConstructFromConfig_UnknownProcess) {
  // Construct
  EvaluationStageConfig config = GetIdentityStageConfig();
  config.mutable_specification()->clear_process_class();
  std::unique_ptr<EvaluationStage> stage_ptr = EvaluationStage::Create(config);
  EXPECT_EQ(stage_ptr.get(), nullptr);
}

TEST(EvaluationStage, NoInitializer) {
  // Construct
  EvaluationStageConfig config = GetIdentityStageConfig();
  std::unique_ptr<EvaluationStage> stage_ptr = EvaluationStage::Create(config);
  // Initialize
  absl::flat_hash_map<std::string, void*> object_map;
  EXPECT_FALSE(stage_ptr->Init(object_map));
}

TEST(EvaluationStage, InvalidInitializer) {
  // Construct
  EvaluationStageConfig config = GetIdentityStageConfig();
  std::unique_ptr<EvaluationStage> stage_ptr = EvaluationStage::Create(config);
  // Initialize
  absl::flat_hash_map<std::string, void*> object_map;
  object_map[kDefaultValueName] = nullptr;
  EXPECT_FALSE(stage_ptr->Init(object_map));
}

TEST(EvaluationStage, NoInputs) {
  // Construct
  EvaluationStageConfig config = GetIdentityStageConfig();
  std::unique_ptr<EvaluationStage> stage_ptr = EvaluationStage::Create(config);
  // Initialize
  absl::flat_hash_map<std::string, void*> object_map;
  float default_value = 1.0;
  object_map[kDefaultValueName] = &default_value;
  EXPECT_TRUE(stage_ptr->Init(object_map));

  // Run
  EXPECT_FALSE(stage_ptr->Run(object_map));
}

TEST(EvaluationStage, ExpectedIdentityOutput) {
  // Construct
  EvaluationStageConfig config = GetIdentityStageConfig();
  std::unique_ptr<EvaluationStage> stage_ptr = EvaluationStage::Create(config);
  // Initialize
  absl::flat_hash_map<std::string, void*> object_map;
  float default_value = 1.0;
  object_map[kDefaultValueName] = &default_value;
  EXPECT_TRUE(stage_ptr->Init(object_map));

  // Input Data
  float input_value = 2.0f;
  // Run
  object_map[kInputValueName] = &input_value;
  EXPECT_TRUE(stage_ptr->Run(object_map));
  EvaluationStageMetrics metrics = stage_ptr->LatestMetrics();
  // Check output
  float* output_value_ptr = static_cast<float*>(object_map[kOutputValueName]);
  EXPECT_NE(output_value_ptr, nullptr);
  EXPECT_FLOAT_EQ(*output_value_ptr, input_value);
  EXPECT_EQ(metrics.num_runs(), 1);

  // Input Data
  input_value = 0.0f;
  // Run
  object_map[kInputValueName] = &input_value;
  EXPECT_TRUE(stage_ptr->Run(object_map));
  metrics = stage_ptr->LatestMetrics();
  // Check output
  output_value_ptr = static_cast<float*>(object_map[kOutputValueName]);
  EXPECT_NE(output_value_ptr, nullptr);
  EXPECT_FLOAT_EQ(*output_value_ptr, default_value);
  EXPECT_EQ(metrics.num_runs(), 2);
}

}  // namespace
}  // namespace evaluation
}  // namespace tflite
