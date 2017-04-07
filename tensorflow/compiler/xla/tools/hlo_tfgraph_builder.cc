/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

LIcensed under the Apache License, Version 2.0 (the "License");
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/tools/hlo_tfgraph_builder.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/strings/strcat.h"

using ::tensorflow::GraphDef;
using ::tensorflow::NodeDef;
using ::tensorflow::strings::StrAppend;
using ::tensorflow::strings::StrCat;

namespace xla {
namespace tools {

static string GetOpDefName(const HloInstruction* instruction) {
  string name = StrCat("hlo-", HloOpcodeString(instruction->opcode()));
  tensorflow::str_util::TitlecaseString(&name, "-");
  name.erase(std::remove(name.begin(), name.end(), '-'), name.end());

  if (instruction->opcode() == HloOpcode::kFusion) {
    string fusion_name = ToString(instruction->fusion_kind());
    StrAppend(&name, tensorflow::StringPiece(fusion_name).substr(1));
  }
  return name;
}

void CleanNodeName(string* name) {
  name->erase(std::remove(name->begin(), name->end(), '%'), name->end());
  const string chars_to_replace = "<>[]";
  auto pred = [&](char c) {
    return std::find(chars_to_replace.begin(), chars_to_replace.end(), c) !=
           chars_to_replace.end();
  };
  std::replace_if(name->begin(), name->end(), pred, '_');
}

Status HloTfGraphBuilder::AddComputation(const HloComputation& computation) {
  LOG(INFO) << "Adding computation " << computation.name();
  for (auto embedded : computation.MakeEmbeddedComputationsList()) {
    LOG(INFO) << "Adding embedded computation " << embedded->name();
    for (auto& instruction : embedded->instructions()) {
      TF_RETURN_IF_ERROR(AddInstruction(instruction.get()));
    }
  }
  for (auto& instruction : computation.instructions()) {
    TF_RETURN_IF_ERROR(AddInstruction(instruction.get()));
  }
  return Status::OK();
}

const GraphDef& HloTfGraphBuilder::GetGraphDef() const { return graph_def_; }

const string& HloTfGraphBuilder::GetNodeNameForInstruction(
    const HloInstruction* instruction) {
  if (ContainsKey(instruction_to_node_name_, instruction)) {
    return instruction_to_node_name_[instruction];
  }
  // If an instruction is fused, put it in the subgraph of the fusion;
  // otherwise, put it in the computation subgraph.
  string node_name =
      instruction->IsFused()
          ? GetNodeNameForInstruction(instruction->fusion_instruction())
          : instruction->parent()->name();
  string instruction_name = instruction->name();
  if (instruction->opcode() == HloOpcode::kParameter) {
    StrAppend(&instruction_name, ".", instruction->parameter_number());
  }
  StrAppend(&node_name, "/", instruction_name);
  CleanNodeName(&node_name);
  auto ret =
      instruction_to_node_name_.insert(std::make_pair(instruction, node_name));
  CHECK(ret.second);
  return ret.first->second;
}

// TODO(b/36987876): Add more attribute information e.g. shapes, dimensions etc.
void HloTfGraphBuilder::SetNodeAttrs(const HloInstruction* instruction,
                                     NodeDef* node_def) const {
  // Set the number of arguments for instructions that have variadic operands.
  if (HloOpcodeIsVariadic(instruction->opcode())) {
    tensorflow::AttrValue attr_value;
    attr_value.set_i(instruction->operands().size());
    (*node_def->mutable_attr())["ArgNum"] = attr_value;
  }
}

Status HloTfGraphBuilder::AddInstruction(const HloInstruction* instruction) {
  if (!visited_instructions_.insert(instruction).second) {
    // Skip instructions that have already been added.
    return Status::OK();
  }

  NodeDef* node_def = graph_def_.add_node();
  node_def->set_name(GetNodeNameForInstruction(instruction));
  node_def->set_op(GetOpDefName(instruction));
  SetNodeAttrs(instruction, node_def);
  if (instruction->opcode() == HloOpcode::kFusion) {
    for (auto& fused_instruction : instruction->fused_instructions()) {
      TF_RETURN_IF_ERROR(AddInstruction(fused_instruction.get()));
    }
  }
  // Add all edges including control edges.
  for (unsigned i = 0; i < instruction->operands().size(); ++i) {
    *node_def->add_input() = GetNodeNameForInstruction(instruction->operand(i));
  }
  // Called computations are control dependencies.
  for (const auto* called_computation : instruction->called_computations()) {
    *node_def->add_input() = StrCat(
        "^", GetNodeNameForInstruction(called_computation->root_instruction()));
  }
  return Status::OK();
}

}  // namespace tools
}  // namespace xla
