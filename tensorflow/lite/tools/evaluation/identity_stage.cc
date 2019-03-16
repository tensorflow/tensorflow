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
#include "tensorflow/lite/tools/evaluation/identity_stage.h"

#include "tensorflow/lite/tools/evaluation/evaluation_stage.h"
#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"

namespace tflite {
namespace evaluation {

bool IdentityStage::DoInit(
    absl::flat_hash_map<std::string, void*>& object_map) {
  float* default_value;
  if (!GetObjectFromTag(kDefaultValueTag, object_map, &default_value)) {
    return false;
  }
  default_value_ = *default_value;
  return true;
}

bool IdentityStage::Run(absl::flat_hash_map<std::string, void*>& object_map) {
  float* current_value;
  GET_OBJECT(kInputValueTag, object_map, &current_value);
  current_value_ = *current_value ? *current_value : default_value_;
  ASSIGN_OBJECT(kOutputValueTag, &current_value_, object_map);
  ++num_runs_;
  return true;
}

EvaluationStageMetrics IdentityStage::LatestMetrics() {
  EvaluationStageMetrics metrics;
  metrics.set_num_runs(num_runs_);
  return metrics;
}

const char IdentityStage::kDefaultValueTag[] = "DEFAULT_VALUE";
const char IdentityStage::kInputValueTag[] = "INPUT_VALUE";
const char IdentityStage::kOutputValueTag[] = "OUTPUT_VALUE";

DEFINE_FACTORY(IdentityStage, IDENTITY);

}  // namespace evaluation
}  // namespace tflite
