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

#include "tensorflow/lite/delegates/gpu/gl/compiler/fuse_inplace.h"

#include <any>
#include <cstring>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/compiled_node.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/preprocessor.h"
#include "tensorflow/lite/delegates/gpu/gl/node_shader.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

static const char* kInplacePrefix = "inplace_update:\0";

class EmptyInplaceRewrite : public InlineRewrite {
 public:
  RewriteStatus Rewrite(absl::string_view input, std::string* output) final {
    if (input.compare(0, strlen(kInplacePrefix), kInplacePrefix) == 0) {
      num_rewrites_++;
      return RewriteStatus::SUCCESS;
    }
    return RewriteStatus::NOT_RECOGNIZED;
  }

  int num_rewrites() const { return num_rewrites_; }

 private:
  int num_rewrites_ = 0;
};

// Takes a code as an input. Replaces 'value_0' in the code with a value that
// comes in a rewrite. For example:
//   code:    value_0 = max(value_0, 0);
//   rewrite: inplace_update:result_12 -> result_12 = max(result_12, 0);
//
class InplaceCodeRewrite : public InlineRewrite {
 public:
  explicit InplaceCodeRewrite(const std::string& code) : code_(code) {}

  RewriteStatus Rewrite(absl::string_view input, std::string* output) final {
    int len = strlen(kInplacePrefix);
    if (input.compare(0, len, kInplacePrefix) == 0) {
      auto variable_name = input.substr(len);
      absl::StrAppend(output,
                      absl::StrReplaceAll(code_, {{"value_0", variable_name}}));
      return RewriteStatus::SUCCESS;
    }
    return RewriteStatus::NOT_RECOGNIZED;
  }

 private:
  std::string code_;
};

}  // namespace

TransformResult RemoveUnusedInplaceUpdates::ApplyToNode(Node* node,
                                                        GraphFloat32* graph) {
  auto& attr =
      std::any_cast<CompiledNodeAttributes&>(node->operation.attributes);
  // Remove inplace block by rewriting to empty string.
  EmptyInplaceRewrite rewrite;
  TextPreprocessor preprocessor('$', true);
  preprocessor.AddRewrite(&rewrite);
  if (!preprocessor.Rewrite(attr.code.source_code, &attr.code.source_code)
           .ok()) {
    return {TransformStatus::INVALID, ""};
  }
  return {rewrite.num_rewrites() > 0 ? TransformStatus::APPLIED
                                     : TransformStatus::SKIPPED,
          ""};
}

TransformResult FuseInplaceUpdate::ApplyToNodesSequence(
    const std::vector<Node*>& sequence, GraphFloat32* graph) {
  Node* node1 = sequence.front();
  Node* node2 = sequence.back();
  auto& attr1 =
      std::any_cast<CompiledNodeAttributes&>(node1->operation.attributes);
  auto& attr2 =
      std::any_cast<CompiledNodeAttributes&>(node2->operation.attributes);

  if (graph->FindInputs(node2->id).size() != 1 ||
      graph->FindOutputs(node2->id).size() != 1 ||
      attr2.code.output != IOStructure::AUTO ||
      attr2.code.input != IOStructure::AUTO ||
      (attr1.code.workload != attr2.code.workload &&
       uint3() != attr2.code.workload)) {
    return {TransformStatus::SKIPPED, ""};
  }

  // First count of replaces that would happen to check whether rewrite is
  // needed.
  {
    EmptyInplaceRewrite counting_rewrite;
    TextPreprocessor preprocessor('$', true);
    preprocessor.AddRewrite(&counting_rewrite);
    std::string temp;
    if (!preprocessor.Rewrite(attr1.code.source_code, &temp).ok()) {
      return {TransformStatus::INVALID, ""};
    }
    // no rewrites in the source code. skip it.
    if (counting_rewrite.num_rewrites() == 0) {
      return {TransformStatus::SKIPPED, ""};
    }
  }
  if (!MergeCode(&attr2, &attr1).ok()) {
    return {TransformStatus::INVALID, "Unable to merge two nodes"};
  }
  TextPreprocessor preprocessor('$', true);
  InplaceCodeRewrite rewrite(attr2.code.source_code);
  preprocessor.AddRewrite(&rewrite);
  if (!preprocessor.Rewrite(attr1.code.source_code, &attr1.code.source_code)
           .ok()) {
    return {TransformStatus::INVALID, ""};
  }
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
