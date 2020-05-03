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

#include "tensorflow/lite/delegates/gpu/gl/compiler/fuse_auto_input.h"

#include <string>
#include <unordered_set>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/types/any.h"
#include "absl/types/variant.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/compiled_node.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

std::pair<std::string, std::string> MakeValueReplacement(int n, int k) {
  return {absl::StrCat("value_", n), absl::StrCat("value_", k)};
}

std::pair<std::string, std::string> MakeDataReplacement(int n, int k) {
  return {absl::StrCat("input_data_", n), absl::StrCat("input_data_", k)};
}

}  // namespace

TransformResult FuseAutoInput::ApplyToNode(Node* node, GraphFloat32* graph) {
  auto& node_attr =
      absl::any_cast<CompiledNodeAttributes&>(node->operation.attributes);
  auto& node_code = node_attr.code;

  if (node_code.input != IOStructure::AUTO) {
    return {TransformStatus::SKIPPED, ""};
  }
  uint3 workgroup = node_code.workgroup;

  auto node_outputs = graph->FindOutputs(node->id);

  // Check which inputs could be fused into the current node.
  std::vector<std::pair<Node*, int>> nodes_to_fuse;
  std::vector<std::pair<ValueId, int>> input_values;
  int input_num = -1;
  for (auto input_value : graph->FindInputs(node->id)) {
    input_num++;
    const ValueId input_id = input_value->id;
    input_values.push_back({input_id, input_num});

    if (graph->FindConsumers(input_id).size() > 1) {
      continue;  // input is consumed by >1 nodes
    }
    Node* input_producer = graph->FindProducer(input_id);
    if (input_producer == nullptr) {
      continue;  // graph's input
    }
    if (graph->FindOutputs(input_producer->id).size() != 1) {
      continue;  // input node has more than one output
    }
    auto& input_producer_attr = absl::any_cast<const CompiledNodeAttributes&>(
        input_producer->operation.attributes);
    if (input_producer_attr.code.output != IOStructure::AUTO) {
      continue;
    }
    if (input_producer_attr.code.workload != node_code.workload &&
        uint3() != input_producer_attr.code.workload) {
      continue;
    }
    if (input_producer_attr.code.workgroup != uint3()) {
      // New fused node should fuse only a single shader that has pre-defined
      // workgroup. Such shader is considered "heavy". Do not fuse two heavy
      // shaders into one.
      // TODO(eignasheva): make sure it still works.
      if (workgroup != uint3()) {
        continue;
      }
      workgroup = input_producer_attr.code.workgroup;
    }
    nodes_to_fuse.push_back({input_producer, input_num});
    input_values.pop_back();  // this value will not be used as input.
  }
  if (nodes_to_fuse.empty()) {
    return {TransformStatus::SKIPPED, ""};
  }

  // Skip fusions which will result in duplicate inputs, e.g. diamond shapes.
  {
    std::unordered_set<ValueId> all_inputs;
    for (const auto& node_to_fuse : nodes_to_fuse) {
      for (const auto& input : graph->FindInputs(node_to_fuse.first->id)) {
        if (all_inputs.find(input->id) != all_inputs.end()) {
          return {TransformStatus::SKIPPED, ""};
        }
        all_inputs.insert(input->id);
      }
    }
    for (const auto& input : graph->FindInputs(node->id)) {
      if (all_inputs.find(input->id) != all_inputs.end()) {
        return {TransformStatus::SKIPPED, ""};
      }
      all_inputs.insert(input->id);
    }
  }

  // Break connections between current node and its inputs.
  for (auto value : graph->FindInputs(node->id)) {
    if (!graph->RemoveConsumer(node->id, value->id).ok()) {
      return {TransformStatus::INVALID, ""};
    }
  }

  std::string operation_type;
  std::string source_code;
  std::string values;

  // Node source code need to be appended later to the end.
  std::swap(source_code, node_code.source_code);

  // Indicates value_k that is beyond originally declared [0..n] values,
  // therefore, it can be used by newly added dependencies.
  int extra_input_num = input_num;
  input_num = 0;

  // Fuse all nodes into one.
  for (auto input_and_num : nodes_to_fuse) {
    auto& input = input_and_num.first;
    auto& attr =
        absl::any_cast<CompiledNodeAttributes&>(input->operation.attributes);
    auto super_inputs = graph->FindInputs(input->id);

    // Replace all internal references in the input source code. For example:
    // source code "value_0 = max(0, value_0);" will be rewritten into
    // "value_2 = max(0, value_2);"
    std::vector<std::pair<std::string, std::string>> replacements;
    for (int i = 0; i < super_inputs.size(); ++i) {
      // Node source code uses value_N to access output value from the fused
      // node. Use correct reference.
      //
      // Here value_N does not correspond to input_N anymore. Instead it tracks
      // value_n and input_m independently. Value_index uses an index needed
      // for the "final" shader, while input_num preserves the order of inputs.
      // For example:
      //    Shader A: input_0, input_1
      //    value_0 = value_0 > value_1 ? value_0 : value_1;
      //
      //    Shader B:  input_0
      //    value_0 = max(0, value_0);
      //
      //    AddShader: input_0, input_1
      //    value_0 = value_0 + value_1;
      //
      //    Fused shader is going to have 3 inputs: input_0 (A), input_1 (A),
      //    input_2 (B). But Shader B need to store result in value_1, because
      //    AddShader refers to it as 'value_1'. So, fused shader will look as
      //    follows:
      //
      //    // Shader A
      //    vec4 value_0 = input_data_0.data[gid.x, gid.y, gid.z];
      //    vec4 value_2 = input_data_1.data[gid.x, gid.y, gid.z];
      //    value_0 = value_0 > value_2 ? value_0 : value_2;
      //
      //    // Shader B
      //    vec4 value_1 = input_data_2.data[gid.x, gid.y, gid.z];
      //    value_1 = max(0, value_1);
      //
      //    // AddShader
      //    value_0 = value_0 + value_1;
      //
      //    output_data_0.data[gid.x, gid.y, gid.z] = value_0;
      int value_index = i == 0 ? input_and_num.second : ++extra_input_num;
      replacements.push_back(MakeValueReplacement(i, value_index));
      replacements.push_back(MakeDataReplacement(i, input_num));

      // Declare input values based on the input structure of the merged node.
      // This code copies what shader_codegen would do automatically.
      if (attr.code.input == IOStructure::AUTO) {
        absl::StrAppend(&values, "  value_", value_index, " = $input_data_",
                        input_num, "[gid.x, gid.y, gid.z]$;\n");
      }

      if (!graph->AddConsumer(node->id, super_inputs[i]->id).ok()) {
        return {TransformStatus::INVALID, ""};
      }
      input_num++;
    }

    // Also rename all _h and _w parameters to the new names.
    for (auto& param : attr.code.parameters) {
      param.name = absl::StrReplaceAll(param.name, replacements);
    }
    attr.code.source_code =
        absl::StrReplaceAll(attr.code.source_code, replacements);

    // Merge all objects, parameters and source code.
    if (!MergeCode(&attr, &node_attr).ok()) {
      return {TransformStatus::INVALID, "Unable to merge the code"};
    }
    absl::StrAppend(&node_attr.code.source_code, "{\n", attr.code.source_code,
                    "\n}");

    if (!operation_type.empty()) {
      operation_type += ",";
    }
    operation_type += input->operation.type;

    if (!graph->DeleteNode(input->id).ok()) {
      return {TransformStatus::INVALID, ""};
    }
  }

  // Add back all inputs that are used directly by the fused node.
  for (int i = 0; i < input_values.size(); i++) {
    if (node_code.input == IOStructure::AUTO) {
      absl::StrAppend(&values, "  value_", input_values[i].second,
                      " = $input_data_", input_num,
                      "[gid.x, gid.y, gid.z]$;\n");
    }
    if (!graph->AddConsumer(node->id, input_values[i].first).ok()) {
      return {TransformStatus::INVALID, ""};
    }
    input_num++;
  }

  node_code.input = IOStructure::ONLY_DEFINITIONS;

  absl::StrAppend(&node->operation.type, "(", operation_type, ")");
  node_code.source_code =
      absl::StrCat(values, node_code.source_code, "{//FUSED",
                   node->operation.type, "\n", source_code, "\n}");

  return {TransformStatus::APPLIED, ""};
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
