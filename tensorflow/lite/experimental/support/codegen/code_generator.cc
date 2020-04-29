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

#include "tensorflow/lite/experimental/support/codegen/code_generator.h"

#include <cctype>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "tensorflow/lite/experimental/support/codegen/utils.h"
#include "tensorflow/lite/experimental/support/metadata/metadata_schema_generated.h"

namespace tflite {
namespace support {
namespace codegen {

namespace {

void ResolveConflictedNamesByAddingIndex(std::vector<std::string>* names_ptr) {
  auto& names = *names_ptr;
  std::unordered_map<std::string, int> indexes;
  std::unordered_map<std::string, int> first_appearance;
  for (int i = 0; i < names.size(); i++) {
    if (indexes.find(names[i]) == indexes.end()) {
      indexes[names[i]] = 1;
      first_appearance[names[i]] = i;
    } else {
      indexes[names[i]] += 1;
      names[i].append(std::to_string(indexes[names[i]]));
    }
  }
  for (const auto& it : first_appearance) {
    const auto& name = it.first;
    const auto i = it.second;
    if (indexes[name] > 1) {
      names[i].append("1");
    }
  }
}

}  // namespace

CodeGenerator::CodeGenerator() {}

bool CodeGenerator::VerifyMetadata(const ModelMetadata* metadata,
                                   ErrorReporter* err) {
  if (metadata == nullptr) {
    err->Error("Loading nullptr is not allowed");
    return false;
  }
  if (metadata->subgraph_metadata()->size() != 1) {
    err->Error("Only exact 1 subgraph is supported");
    return false;
  }
  return true;
}

std::pair<std::vector<std::string>, std::vector<std::string>>
CodeGenerator::NameInputsAndOutputs(const TensorMetadataList* inputs,
                                    const TensorMetadataList* outputs) {
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  if (inputs != nullptr) {
    input_names.reserve(inputs->size());
    for (const auto* tensor : *inputs) {
      input_names.push_back(NameTensor(*tensor, "input"));
    }
  }
  if (outputs != nullptr) {
    output_names.reserve(outputs->size());
    for (const auto* tensor : *outputs) {
      output_names.push_back(NameTensor(*tensor, "output"));
    }
  }
  // Solve conflict
  ResolveConflictedInputAndOutputNames(&input_names, &output_names);
  return std::make_pair(input_names, output_names);
}

std::string CodeGenerator::ConvertToValidName(const std::string& name) {
  // lowercase all
  std::string result = name;
  for (int i = 0; i < result.size(); i++) {
    result[i] = std::tolower(result[i]);
  }
  // replace all non-alpha or non-numeric with underscores, except underscore
  // itself
  for (int i = 0; i < result.size(); i++) {
    if (result[i] != '_' && !std::isalnum(result[i])) {
      result[i] = '_';
    }
  }
  // remove leading underscores
  int leading_underscores = 0;
  while (leading_underscores < result.size() &&
         result[leading_underscores] == '_') {
    leading_underscores++;
  }
  result.erase(0, leading_underscores);
  if (result.empty()) {
    return "";
  }
  // first char should be alpha
  if (std::isalpha(result[0])) {
    return result;
  }
  return "tensor_" + result;
}

std::string CodeGenerator::NameTensor(const TensorMetadata& tensor,
                                      const std::string& default_name) {
  if (tensor.name() != nullptr && tensor.name()->size() > 0) {
    // TODO(b/141225157) Validate tensor name. It should be in lower case.
    auto suggested_name = ConvertToValidName(tensor.name()->str());
    if (!suggested_name.empty()) {
      return suggested_name;
    }
  }
  auto* content = tensor.content();
  if (content == nullptr || content->content_properties() == nullptr) {
    return default_name;
  }
  switch (content->content_properties_type()) {
    case ContentProperties_ImageProperties:
      return "image";
    case ContentProperties_FeatureProperties:
      return "feature";
    default:
      return default_name;
  }
}

void CodeGenerator::ResolveConflictedInputAndOutputNames(
    std::vector<std::string>* inputs, std::vector<std::string>* outputs) {
  std::unordered_set<std::string> io_conflict;
  auto& input_names = *inputs;
  auto& output_names = *outputs;
  for (const auto& input : input_names) {
    if (io_conflict.find(input) != io_conflict.end()) {
      continue;
    }
    for (const auto& output : output_names) {
      if (input == output) {
        io_conflict.insert(input);
        break;
      }
    }
  }
  for (int i = 0; i < input_names.size(); i++) {
    if (io_conflict.find(input_names[i]) != io_conflict.end()) {
      input_names[i] = "input_" + input_names[i];
    }
  }
  for (int i = 0; i < output_names.size(); i++) {
    if (io_conflict.find(output_names[i]) != io_conflict.end()) {
      output_names[i] = "output_" + output_names[i];
    }
  }
  // 2. Second, add index if input[i] == input[j]
  ResolveConflictedNamesByAddingIndex(&input_names);
  ResolveConflictedNamesByAddingIndex(&output_names);
}

}  // namespace codegen
}  // namespace support
}  // namespace tflite
