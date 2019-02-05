/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"

#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/lib/strings/scanner.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace grappler {
namespace function_utils {

FunctionDefTensorDesc::FunctionDefTensorDesc(const string& node_name,
                                             const string& output, int position)
    : node_name(node_name), node_output(output), position(position) {
  full_str = strings::StrCat(node_name, ":", node_output, ":", position);
}

FunctionDefTensorDesc::FunctionDefTensorDesc(const string& input) {
  // Parses node_name:node_output:position string into its components.
  full_str = input;
  StringPiece capture;
  StringPiece remaining;

  // Parse "node_name"
  if (strings::Scanner(input)
          .One(strings::Scanner::LETTER_DIGIT_DOT_UNDERSCORE)
          .Any(strings::Scanner::LETTER_DIGIT_DASH_DOT_SLASH_UNDERSCORE)
          .GetResult(&remaining, &capture)) {
    node_name = string(capture.data(), capture.size());
  }

  // Parse "node_output" if it exists
  if (strings::Scanner(remaining)
          .OneLiteral(":")
          .RestartCapture()
          .One(strings::Scanner::LETTER)
          .Any(strings::Scanner::LETTER_DIGIT_UNDERSCORE)
          .GetResult(&remaining, &capture)) {
    node_output = string(capture.data(), capture.size());
  }

  // Parse "position" if it exists
  if (strings::Scanner(remaining)
          .OneLiteral(":")
          .RestartCapture()
          .Many(strings::Scanner::DIGIT)
          .GetResult(nullptr, &capture)) {
    CHECK(strings::safe_strto32(capture, &position));
  }
}

// TODO(rachelim): Create a utility class similar to MutableGraphView for
// FunctionDefs, and use that to manipulate functions. It'll be more
// performant if we kept mappings of nodes->inputs/outputs, so that we don't
// have to search over all nodes each time.
// Note that we're not using GrapplerFunctionItem because it doesn't cover
// some of our desired uses (eg changing the outputs of a function), and the
// FunctionDef -> GraphDef conversion isn't really necessary in this case.
void ReplaceReferences(const string& from, const string& to,
                       FunctionDef* func) {
  for (NodeDef& n : *func->mutable_node_def()) {
    std::replace(n.mutable_input()->begin(), n.mutable_input()->end(), from,
                 to);
  }

  for (auto& p : *func->mutable_ret()) {
    if (p.second == from) {
      p.second = to;
    }
  }
}

void AddFunctionOutputWithUniqueName(StringPiece prefix,
                                     StringPiece output_tensor_name,
                                     FunctionDef* function, DataType dt) {
  string name = string(prefix);
  int id = function->signature().output_arg_size();
  while (ContainsFunctionOutputWithName(name, *function)) {
    name = strings::StrCat(prefix, "/_", id);
    ++id;
  }
  auto* output = function->mutable_signature()->mutable_output_arg()->Add();
  output->set_name(name);
  output->set_type(dt);

  (*function->mutable_ret())[name] = string(output_tensor_name);
}

NodeDef* AddNode(StringPiece name, StringPiece op,
                 const std::vector<string>& inputs,
                 const std::vector<std::pair<string, AttrValue>>& attributes,
                 FunctionDef* fd) {
  NodeDef* node = fd->add_node_def();
  if (!name.empty()) {
    node->set_name(string(name));
  } else {
    SetUniqueFunctionNodeName(op, fd, node);
  }
  node->set_op(string(op));
  for (const string& input : inputs) {
    node->add_input(input);
  }
  for (auto attr : attributes) {
    (*node->mutable_attr())[attr.first] = attr.second;
  }
  return node;
}

bool ContainsFunctionNodeWithName(StringPiece name,
                                  const FunctionDef& function) {
  return FindFunctionNodeWithName(name, function) != -1;
}

bool ContainsFunctionNodeWithOp(StringPiece op, const FunctionDef& function) {
  return FindFunctionNodeWithOp(op, function) != -1;
}

bool ContainsFunctionOutputWithName(StringPiece name,
                                    const FunctionDef& function) {
  return FindFunctionOutputWithName(name, function) != -1;
}

int FindFunctionInputWithName(StringPiece name, const FunctionDef& function) {
  return graph_utils::GetFirstElementIndexWithPredicate(
      [&name](const OpDef_ArgDef& arg) { return arg.name() == name; },
      function.signature().input_arg());
}

int FindFunctionOutputWithName(StringPiece name, const FunctionDef& function) {
  return graph_utils::GetFirstElementIndexWithPredicate(
      [&name](const OpDef_ArgDef& arg) { return arg.name() == name; },
      function.signature().output_arg());
}

int FindFunctionNodeWithName(StringPiece name, const FunctionDef& function) {
  return graph_utils::GetFirstElementIndexWithPredicate(
      [&name](const NodeDef& node) { return node.name() == name; },
      function.node_def());
}

int FindFunctionNodeWithOp(StringPiece op, const FunctionDef& function) {
  return graph_utils::GetFirstElementIndexWithPredicate(
      [&op](const NodeDef& node) { return node.op() == op; },
      function.node_def());
}

void SetUniqueFunctionNodeName(StringPiece prefix, FunctionDef* function,
                               NodeDef* node) {
  string name = string(prefix);
  int id = function->node_def_size();
  while (ContainsFunctionNodeWithName(name, *function)) {
    name = strings::StrCat(prefix, "/_", id);
    ++id;
  }
  node->set_name(std::move(name));
}

bool IsFunctionStateful(const FunctionLibraryDefinition& library,
                        const FunctionDef& function_def, bool skip_assert) {
  if (!function_def.signature().is_stateful()) return false;

  for (const NodeDef& node_def : function_def.node_def()) {
    if (IsNodeStateful(library, node_def, skip_assert)) return true;
  }
  return false;
}

bool IsNodeStateful(const FunctionLibraryDefinition& library,
                    const NodeDef& node, bool skip_assert) {
  const OpDef* op_def;
  Status s = OpRegistry::Global()->LookUpOpDef(node.op(), &op_def);

  if (!s.ok()) return true;

  if (!op_def->is_stateful()) return false;

  if (skip_assert && op_def->name() == "Assert") {
    return false;
  }

  if (op_def->name() == "If") {
    const FunctionDef* then_func =
        library.Find(node.attr().at("then_branch").func().name());
    const FunctionDef* else_func =
        library.Find(node.attr().at("else_branch").func().name());
    if ((then_func != nullptr &&
         !IsFunctionStateful(library, *then_func, skip_assert)) &&
        (else_func != nullptr &&
         !IsFunctionStateful(library, *else_func, skip_assert))) {
      return false;
    }
  }

  if (op_def->name() == "While") {
    const FunctionDef* cond_func =
        library.Find(node.attr().at("cond").func().name());
    const FunctionDef* body_func =
        library.Find(node.attr().at("body").func().name());
    if ((cond_func != nullptr &&
         !IsFunctionStateful(library, *cond_func, skip_assert)) &&
        (body_func != nullptr &&
         !IsFunctionStateful(library, *body_func, skip_assert))) {
      return false;
    }
  }
  return true;
}

}  // namespace function_utils
}  // namespace grappler
}  // namespace tensorflow
