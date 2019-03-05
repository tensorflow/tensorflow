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

#include <string>

#include "absl/strings/str_split.h"

namespace tflite {
namespace evaluation {

bool EvaluationStage::Init(
    absl::flat_hash_map<std::string, void*>& object_map) {
  // Process & validate configuration of tags.
  std::vector<std::string> initializers, inputs, outputs;
  for (const auto& init : config_.initializers()) {
    initializers.emplace_back(init);
  }
  for (const auto& in : config_.inputs()) {
    inputs.emplace_back(in);
  }
  for (const auto& out : config_.outputs()) {
    outputs.emplace_back(out);
  }
  if (!ProcessExpectedTags(GetInitializerTags(), initializers) ||
      !ProcessExpectedTags(GetInputTags(), inputs) ||
      !ProcessExpectedTags(GetOutputTags(), outputs)) {
    return false;
  }
  // Class-specific stuff.
  return DoInit(object_map);
}

bool EvaluationStage::ProcessExpectedTags(
    const std::vector<std::string>& expected_tags,
    std::vector<std::string>& tag_to_name_mappings) {
  // Validate format of each TAG:name mapping in tag_to_name_mappings, and add
  // it to tags_to_names_map_.
  for (const std::string& tag_name_mapping : tag_to_name_mappings) {
    if (!std::regex_match(tag_name_mapping, kTagNameMappingPattern)) {
      LOG(ERROR) << "Invalid TAG:name mapping: " << tag_name_mapping;
      return false;
    }
    std::vector<std::string> tag_and_name =
        absl::StrSplit(tag_name_mapping, ':');
    tags_to_names_map_[tag_and_name[0]] = tag_and_name[1];
  }

  // Ensure each expected TAG is valid & has been mapped to a name.
  for (const std::string& tag : expected_tags) {
    if (!std::regex_match(std::string(tag), kTagPattern)) {
      LOG(ERROR) << "Invalid expected TAG: " << tag;
      return false;
    }
    if (tags_to_names_map_.find(tag) == tags_to_names_map_.end()) {
      LOG(ERROR) << "TAG " << tag << " has not been mapped to a name in config "
                 << config_.name();
      return false;
    }
  }
  return true;
}

}  // namespace evaluation
}  // namespace tflite
