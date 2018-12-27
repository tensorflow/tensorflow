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

#include "tensorflow/compiler/xla/service/hlo_tfgraph_builder.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"

namespace xla {
namespace hlo_graph_dumper {
namespace {

using absl::StrAppend;
using absl::StrCat;
using tensorflow::GraphDef;
using tensorflow::NodeDef;
using tensorflow::TensorShapeProto;

string GetOpDefName(const HloInstruction* instruction) {
  string name = StrCat("hlo-", HloOpcodeString(instruction->opcode()));
  tensorflow::str_util::TitlecaseString(&name, "-");  // non-absl ok
  name.erase(std::remove(name.begin(), name.end(), '-'), name.end());

  if (instruction->opcode() == HloOpcode::kFusion) {
    string fusion_name = ToString(instruction->fusion_kind());
    StrAppend(&name, absl::string_view(fusion_name).substr(1));
  }
  return name;
}

TensorShapeProto GetTensorShape(const HloInstruction* instruction) {
  TensorShapeProto tensor_shape;
  const Shape& shape = instruction->shape();
  for (auto dim : shape.dimensions()) {
    tensor_shape.add_dim()->set_size(dim);
  }
  return tensor_shape;
}

string GetDeviceName(int device) { return StrCat("/device/XLA:", device); }

void CleanNodeName(string* name) {
  name->erase(std::remove(name->begin(), name->end(), '%'), name->end());
  const string chars_to_replace = "<>[]";
  auto pred = [&](char c) {
    return std::find(chars_to_replace.begin(), chars_to_replace.end(), c) !=
           chars_to_replace.end();
  };
  std::replace_if(name->begin(), name->end(), pred, '_');
}

}  // namespace

HloTfGraphBuilder::HloTfGraphBuilder(const DebugOptions& debug_options)
    : debug_options_(debug_options) {}

Status HloTfGraphBuilder::AddComputation(const HloComputation& computation) {
  VLOG(2) << "Adding computation " << computation.name();
  for (auto embedded : computation.MakeEmbeddedComputationsList()) {
    for (auto* instruction : embedded->instructions()) {
      TF_RETURN_IF_ERROR(AddInstruction(instruction));
    }
  }
  for (auto* instruction : computation.instructions()) {
    TF_RETURN_IF_ERROR(AddInstruction(instruction));
  }
  return Status::OK();
}

const GraphDef& HloTfGraphBuilder::GetGraphDef() const { return graph_def_; }

const string& HloTfGraphBuilder::GetNodeNameForInstruction(
    const HloInstruction* instruction) {
  if (ContainsKey(instruction_to_node_name_, instruction)) {
    return instruction_to_node_name_[instruction];
  }
  auto append = [](string* str, const string& other) {
    if (str->empty()) {
      *str = other;
    } else if (!other.empty()) {
      StrAppend(str, "/", other);
    }
  };
  string node_name;
  if (debug_options_.xla_hlo_tfgraph_device_scopes()) {
    auto device = instruction->sharding_unique_device();
    if (device) {
      node_name = StrCat("dev", *device);
    }
  }
  // If an instruction is fused, put it in the subgraph of the fusion;
  // otherwise, put it in the computation subgraph.
  const HloComputation* computation = instruction->parent();
  if (computation->IsFusionComputation()) {
    append(&node_name,
           GetNodeNameForInstruction(computation->FusionInstruction()));
  } else {
    append(&node_name, computation->name());
    if (!instruction->metadata().op_name().empty()) {
      // Always make computations contain TF ops but not the other way around.
      append(&node_name, instruction->metadata().op_name());
    }
  }
  string instruction_name = instruction->name();
  if (instruction->opcode() == HloOpcode::kParameter) {
    StrAppend(&instruction_name, ".", instruction->parameter_number());
  }
  append(&node_name, instruction_name);
  CleanNodeName(&node_name);
  auto ret =
      instruction_to_node_name_.insert(std::make_pair(instruction, node_name));
  CHECK(ret.second);
  return ret.first->second;
}

void HloTfGraphBuilder::SetNodeAttrs(const HloInstruction* instruction,
                                     NodeDef* node_def) const {
  auto& attrs = *node_def->mutable_attr();

  // Set the number of arguments for instructions that have variadic operands.
  if (HloOpcodeIsVariadic(instruction->opcode())) {
    tensorflow::AttrValue attr_value;
    attr_value.set_i(instruction->operands().size());
    attrs["arg_num"] = attr_value;
  }

  // Set the node type.
  attrs["type"].set_s(
      xla::PrimitiveType_Name(instruction->shape().element_type()));

  // Set the framework op (e.g. Tensorflow op) that generated this XLA op.
  attrs["tf_op_type"].set_s(instruction->metadata().op_type());
  attrs["tf_op_name"].set_s(instruction->metadata().op_name());

  // Set the shape of the output tensor. "_output_shapes" is a special attribute
  // name used by Tensorboard for shapes of output tensors.
  tensorflow::AttrValue shapes;
  *shapes.mutable_list()->add_shape() = GetTensorShape(instruction);
  attrs["_output_shapes"] = shapes;

  // Set the layout.
  if (LayoutUtil::HasLayout(instruction->shape())) {
    string layout_string;
    if (instruction->shape().IsTuple()) {
      // For tuples, emit the full shape because the layout of a tuple is not
      // represented in a single Layout field.
      layout_string = ShapeUtil::HumanStringWithLayout(instruction->shape());
    } else {
      layout_string = StrCat(
          "{",
          absl::StrJoin(LayoutUtil::MinorToMajor(instruction->shape()), ","),
          "}");
    }
    attrs["layout"].set_s(layout_string);
  }

  // Set op-specific attributes.
  switch (instruction->opcode()) {
    case HloOpcode::kConcatenate:
    case HloOpcode::kBroadcast:
    case HloOpcode::kReduce:
    case HloOpcode::kReverse:
    case HloOpcode::kTranspose:
      for (auto dim : instruction->dimensions()) {
        attrs["dims"].mutable_list()->add_i(dim);
      }
      break;
    case HloOpcode::kGetTupleElement:
      attrs["index"].set_i(instruction->tuple_index());
      break;
    case HloOpcode::kRng:
      attrs["dist"].set_s(
          RandomDistribution_Name(instruction->random_distribution()));
      break;
    case HloOpcode::kConstant:
      if (ShapeUtil::IsScalar(instruction->shape())) {
        attrs["value"].set_s(instruction->literal().GetAsString({}));
      }
      break;
    case HloOpcode::kCustomCall:
      attrs["custom_call_target"].set_s(instruction->custom_call_target());
      break;
    case HloOpcode::kSend:
    case HloOpcode::kRecv:
      attrs["channel_id"].set_i(instruction->channel_id());
      break;
    default:
      break;
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

  auto device = instruction->sharding_unique_device();
  if (device) {
    node_def->set_device(GetDeviceName(*device));
  }
  SetNodeAttrs(instruction, node_def);
  if (instruction->opcode() == HloOpcode::kFusion) {
    for (auto* fused_instruction : instruction->fused_instructions()) {
      TF_RETURN_IF_ERROR(AddInstruction(fused_instruction));
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

}  // namespace hlo_graph_dumper
}  // namespace xla
