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

#include "tensorflow/lite/delegates/gpu/gl/compiler/fuse_inline.h"

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/compiled_node.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/shader_code.h"
#include "tensorflow/lite/delegates/gpu/gl/node_shader.h"

namespace tflite {
namespace gpu {
namespace gl {

TransformResult FuseAutoOutputWithInline::ApplyToNodesSequence(
    const std::vector<Node*>& sequence, GraphFloat32* graph) {
  Node* node1 = sequence.front();
  Node* node2 = sequence.back();
  auto& attr1 =
      absl::any_cast<CompiledNodeAttributes&>(node1->operation.attributes);
  auto& attr2 =
      absl::any_cast<CompiledNodeAttributes&>(node2->operation.attributes);

  if (attr1.code.output != IOStructure::AUTO ||
      graph->FindInputs(node2->id).size() != 1 ||
      graph->FindOutputs(node2->id).size() != 1 ||
      attr2.code.output != IOStructure::AUTO ||
      attr2.code.input != IOStructure::AUTO ||
      (attr1.code.workload != attr2.code.workload &&
       uint3() != attr2.code.workload) ||
      graph->FindOutputs(node1->id).size() !=
          graph->FindInputs(node2->id).size()) {
    return {TransformStatus::SKIPPED, ""};
  }

  // Check if the code was not fused yet, and wrap source code into {}.
  if (node1->operation.type.find('+') == std::string::npos) {
    attr1.code.source_code =
        absl::StrCat("\n{\n", attr1.code.source_code, "\n}\n");
  }
  if (!MergeCode(&attr2, &attr1).ok()) {
    return {TransformStatus::INVALID, "Unable to merge two nodes"};
  }
  absl::StrAppend(&attr1.code.source_code, "{\n", attr2.code.source_code,
                  "\n}");
  node1->operation.type += "+" + node2->operation.type;

  if (!RemoveFollowingNode(graph, node2, node1).ok()) {
    return {TransformStatus::INVALID,
            "Unable to remove node " + std::to_string(node2->id)};
  }
  return {TransformStatus::APPLIED, ""};
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
