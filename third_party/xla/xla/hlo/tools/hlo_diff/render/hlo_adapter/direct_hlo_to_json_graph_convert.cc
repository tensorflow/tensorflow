/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/tools/hlo_diff/render/hlo_adapter/direct_hlo_to_json_graph_convert.h"

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module_metadata.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/tools/hlo_diff/render/hlo_adapter/graphnode_builder.h"
#include "xla/hlo/tools/hlo_diff/render/hlo_adapter/schema_structs.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace tooling {
namespace visualization_client {
namespace {

constexpr absl::string_view kShapeWithLayout = "shape_with_layout";
constexpr absl::string_view kOpName = "op_name";
constexpr absl::string_view kOpType = "op_type";
constexpr absl::string_view kSourceFile = "source_file";
constexpr absl::string_view kSourceLine = "source_line";
constexpr absl::string_view kSourceStack = "source_stack";
constexpr absl::string_view kOpcode = "opcode";
constexpr absl::string_view kAddress = "address";
constexpr absl::string_view kBackendConfig = "backend_config";
constexpr absl::string_view kExtraAttributes = "extra_attributes";
constexpr absl::string_view kGetTupleElementIndex = "get_tuple_element_index";
constexpr absl::string_view kUsers = "users";
constexpr absl::string_view kOperands = "operands";
constexpr absl::string_view kLiteral = "literal";
constexpr absl::string_view kHide = "hide_node";
constexpr absl::string_view kAcfComputationName = "async_collective_fusion";
constexpr absl::string_view kAcsInstructionName = "AsyncCollectiveStart";
constexpr absl::string_view kAcdInstructionName = "AsyncCollectiveDone";
constexpr absl::string_view kFusionComputation = "fusion_computation";
constexpr absl::string_view kInlinedOperands = "inlined_operands";

constexpr int kMaxUsersToRender = 16;

// OutputEdges is a map from source instruction id to a list of its users.
using OutputEdges =
    absl::flat_hash_map<std::string, std::vector<const xla::HloInstruction*>>;

// GroupNodeAttributes is a map from group namespace to a map of attribute key
// to attribute value.
using GroupNodeAttributes =
    absl::flat_hash_map<std::string,
                        absl::flat_hash_map<std::string, std::string>>;

// Returns true if value is considered empty.
bool IsEmpty(const llvm::json::Value& value) {
  if (const auto* obj = value.getAsObject()) {
    return obj->empty();
  }
  if (const auto* arr = value.getAsArray()) {
    return arr->empty();
  }
  return false;
}

// Recursively removes empty fields from objects.
void RemoveEmptyFields(llvm::json::Value& value) {
  if (auto* obj = value.getAsObject()) {
    std::vector<llvm::StringRef> keys_to_remove;
    for (auto& it : *obj) {
      llvm::json::Value& child = it.second;
      RemoveEmptyFields(child);
      if (IsEmpty(child)) {
        keys_to_remove.push_back(it.first);
      }
    }
    for (const llvm::StringRef& key : keys_to_remove) {
      obj->erase(key);
    }
  } else if (auto* arr = value.getAsArray()) {
    for (llvm::json::Value& item : *arr) {
      RemoveEmptyFields(item);
    }
  }
}

// Generates a string representation of an HloConstantInstruction.
// - For zero-element arrays, returns "{}(shape)".
// - For small arrays (up to 8 elements) with literal size up to 64,
//   returns "shape_with_layout literal_value".
// - Otherwise, returns "constant_name shape". The constant name is prefixed
//   with "constant " if it doesn't already start with "constant".
std::string StringifyConstant(const xla::HloConstantInstruction* constant,
                              const xla::Shape& shape) {
  if (xla::ShapeUtil::IsZeroElementArray(shape)) {
    return absl::StrFormat("{} (%s)",
                           xla::ShapeUtil::HumanString(constant->shape()));
  }

  std::optional<int64_t> elem_count;
  if (shape.IsArray()) {
    elem_count = xla::ShapeUtil::ElementsIn(constant->shape());
  }
  if (elem_count.has_value() && *elem_count <= 8 && constant->HasLiteral()) {
    std::string literal_str = constant->literal().ToStringWithoutShape();
    if (literal_str.size() <= 64) {
      return absl::StrFormat(
          "%s %s", xla::ShapeUtil::HumanStringWithLayout(shape), literal_str);
    }
  }

  std::string constant_name;
  // Prefix the constant name with "constant " unless it already starts with it
  // to avoid redundant prefixes like "constant constant_foo".
  if (absl::StartsWith(constant->name(), "constant")) {
    constant_name = std::string(constant->name());
  } else {
    constant_name = absl::StrCat("constant ", constant->name());
  }
  return absl::StrFormat("%s %s", constant_name,
                         xla::ShapeUtil::HumanString(shape));
}

bool IsFusedBroadcastOfConstantEffectiveScalar(
    const xla::HloInstruction* instr) {
  namespace m = xla::match;
  return instr->parent()->IsFusionComputation() &&
         Match(instr, m::Broadcast(m::ConstantEffectiveScalar()));
}

// TODO(b/402148725) Move utility functions to a separate file.
// Detect if an instruction is an AsyncCollectiveFusion parameter that is
// implementation details.
bool IsAcfParameter(const xla::HloInstruction* instruction) {
  // Parameter is fused
  if (instruction->opcode() != xla::HloOpcode::kParameter ||
      !instruction->IsFused()) {
    return false;
  }

  // Parameter piped through and is only consumed by 1 user
  // Parameter 0 consumed by both root and all-gather will always persist.
  if (instruction->user_count() != 1) {
    return false;
  }

  const xla::HloComputation* parent_computation = instruction->parent();
  int64_t parameter_number = instruction->parameter_number();
  xla::HloInstruction* fusion_instruction =
      parent_computation->FusionInstruction();
  const xla::HloInstruction* parameterOperand =
      fusion_instruction->operand(parameter_number);
  // Operand is get-tuple-element
  if (parameterOperand->opcode() != xla::HloOpcode::kGetTupleElement) {
    return false;
  }

  const xla::HloInstruction* gteOperand = parameterOperand->operand(0);
  if (gteOperand->opcode() != xla::HloOpcode::kFusion) {
    return false;
  }
  auto src_instruction =
      gteOperand->fused_instructions_computation()->root_instruction();
  // (1) Parameter is fused into AsyncCollectiveFusion, operand is gte from
  // AsyncCollectiveStart custom call and user is the root node of ACF
  // (2) Parameter is mapped from Params in AsyncCollectiveFusion - operand is
  // gte from ACF, and user is AsyncCollectiveDone custom call
  return (absl::StartsWith(parent_computation->name(), kAcfComputationName) &&
          src_instruction->IsCustomCall(kAcsInstructionName) &&
          instruction->users()[0] == parent_computation->root_instruction()) ||
         (instruction->users()[0]->IsCustomCall(kAcdInstructionName) &&
          absl::StartsWith(gteOperand->fused_instructions_computation()->name(),
                           kAcfComputationName));
}

// Recursively include all instructions in the nested computations.
void RecursiveIncludeNestedComputations(
    const xla::HloInstruction* instruction,
    absl::flat_hash_set<const xla::HloInstruction*>& included_nodes) {
  for (const xla::HloComputation* subcomputation :
       instruction->called_computations()) {
    for (const xla::HloInstruction* subcomputation_instruction :
         subcomputation->instructions()) {
      included_nodes.insert(subcomputation_instruction);
      RecursiveIncludeNestedComputations(subcomputation_instruction,
                                         included_nodes);
    }
  }
}

// Gets a NodeFilter that includes roughly all instructions whose distance from
// root is <= radius. The fusion instruction (and its fused computation) is
// treated as a single entity. The scope will not go beyond instruction's parent
// computation.
NodeFilter MakeInstructionRadiusAroundFilter(const xla::HloInstruction* root,
                                             const int radius) {
  absl::flat_hash_set<const xla::HloInstruction*> included_nodes;
  std::deque<std::pair<const xla::HloInstruction*, int>> worklist;
  worklist.push_back({root, 0});

  while (!worklist.empty()) {
    const auto [instruction, depth] = worklist.front();
    worklist.pop_front();
    if (depth > radius) {
      continue;
    }
    included_nodes.insert(instruction);

    // Traverse instruction's operands.
    // Don't traverse into tuples' operands unless the tuple is the root.
    // Usually a tuple is the bottommost node in the graph, and so its operands
    // are not interesting to the graph at hand.
    if (instruction == root ||
        instruction->opcode() != xla::HloOpcode::kTuple) {
      for (const xla::HloInstruction* operand : instruction->operands()) {
        if (!included_nodes.contains(operand)) {
          worklist.push_back({operand, depth + 1});
        }
      }
    }

    // Recursively include all the instructions in nested computations.
    RecursiveIncludeNestedComputations(instruction, included_nodes);

    // Traverse instruction's users, omit users if there are too many.
    // Consider provide more context in filter lambda return value, so we can
    // provide omitted information.
    if (instruction->user_count() > kMaxUsersToRender) {
      included_nodes.insert(instruction);
      continue;
    }
    for (const xla::HloInstruction* user : instruction->users()) {
      if (!included_nodes.contains(user)) {
        worklist.push_back({user, depth + 1});
      }
    }
  }

  return [=](const xla::HloInstruction* instruction) {
    return included_nodes.contains(instruction);
  };
}

// Gets the computation hierarchy, split by "/", e.g., "computation_0/fusion_1".
std::string GetComputationHierarchy(
    const std::vector<std::string>& computation_stack) {
  return absl::StrJoin(computation_stack, "/");
}

bool IsGetTupleElement(const HloAdapterOption& options,
                       const xla::HloInstruction* instruction) {
  return options.get_tuple_element_folding &&
         instruction->opcode() == xla::HloOpcode::kGetTupleElement;
}

// Gets the shape string with layout for the given instruction.
std::string GetShapeString(const xla::HloInstruction* instruction) {
  return xla::ShapeUtil::HumanStringWithLayout(instruction->shape());
}

bool IsRootInstruction(const xla::HloInstruction* instruction) {
  return instruction == instruction->parent()->root_instruction();
}

bool ShouldFoldConstant(const xla::HloInstruction* instruction) {
  if (IsRootInstruction(instruction)) {
    return false;
  }
  return instruction->opcode() == xla::HloOpcode::kConstant ||
         IsFusedBroadcastOfConstantEffectiveScalar(instruction);
}

absl::Status AddHloInstructionIncomingEdges(
    const xla::HloInstruction* instruction, const HloAdapterOption& options,
    GraphNodeBuilder& builder, OutputEdges& output_edges,
    const ComputationExpand& computation_expand) {
  if (instruction->opcode() == xla::HloOpcode::kFusion &&
      computation_expand(instruction,
                         instruction->fused_instructions_computation())) {
    // If the instruction is an expanded fusion, don't connect it to anything
    // because the operands will be connected to the parameters.
    return absl::OkStatus();
  }
  std::vector<const xla::HloInstruction*> operands;
  // If the instruction is a Parameter within a Fusion computation, we connect
  // the operands of the fusion computation to the parameters.
  if (instruction->opcode() == xla::HloOpcode::kParameter &&
      instruction->IsFused()) {
    const xla::HloInstruction* fusion_instruction =
        instruction->parent()->FusionInstruction();
    if (fusion_instruction == nullptr) {
      return absl::InternalError("Fusion instruction not found");
    }
    operands.push_back(
        fusion_instruction->operand(instruction->parameter_number()));
  } else {
    operands.insert(operands.end(), instruction->operands().begin(),
                    instruction->operands().end());
  }

  int input_id = 0;
  for (const xla::HloInstruction* operand : operands) {
    // Skip rendering constant nodes or fused broadcast of constant effective
    // scalar nodes as separate entities when `constant_folding` is enabled,
    // and if they are not the root instruction.
    if (options.constant_folding && ShouldFoldConstant(operand)) {
      continue;
    }

    std::string src_instruction_id;
    if (IsGetTupleElement(options, operand)) {
      // Skip the GTE operand, and connect the user to the tuple directly.
      operand = operand->operand(0);
    }

    if (operand->opcode() == xla::HloOpcode::kFusion &&
        computation_expand(operand,
                           operand->fused_instructions_computation())) {
      // If the operand is a fusion, we connect the user to the root of the
      // fusion computation.
      src_instruction_id = GetInstructionId(
          operand->fused_instructions_computation()->root_instruction());
    } else {
      src_instruction_id = GetInstructionId(operand);
    }

    const std::string output_id =
        absl::StrCat(output_edges[src_instruction_id].size());
    builder.AppendEdgeInfo(
        /*source_node_id_str=*/src_instruction_id,
        output_id, /*target_node_input_id_str=*/
        absl::StrCat(input_id++));
    output_edges[src_instruction_id].push_back(instruction);
  }
  return absl::OkStatus();
}

void SetInstructionNodeLabel(const xla::HloInstruction* instruction,
                             GraphNodeBuilder& builder) {
  // Instruction label.
  std::string instruction_label;
  if (instruction->opcode() == xla::HloOpcode::kParameter) {
    instruction_label =
        absl::StrFormat("Parameter %d", instruction->parameter_number());
  } else if (instruction->opcode() == xla::HloOpcode::kConstant) {
    instruction_label = "Constant";
  } else {
    instruction_label = instruction->name();
  }

  // Set the text inside the instruction node.
  builder.SetNodeLabel(instruction_label);
}

void SetInstructionNodeAttributes(const xla::HloInstruction* instruction,
                                  const HloAdapterOption& options,
                                  GraphNodeBuilder& builder) {
  // Instruction opcode.
  std::string opcode = std::string(HloOpcodeString(instruction->opcode()));

  // Adds fusion kind to the opcode for fusion instructions.
  if (instruction->opcode() == xla::HloOpcode::kFusion) {
    absl::StrAppend(&opcode, ":", xla::ToString(instruction->fusion_kind()));
  }
  builder.AppendNodeAttribute(kOpcode, opcode);

  // Instruction shape with layout.
  builder.AppendNodeAttribute(kShapeWithLayout, GetShapeString(instruction));

  // Add instruction users if the users are omitted with max threshold.
  // If within threshold, users are the same as inputs shown in the graph.
  if (instruction->user_count() > kMaxUsersToRender) {
    std::vector<std::string> users_str;
    users_str.reserve(instruction->user_count());
    for (const xla::HloInstruction* user : instruction->users()) {
      users_str.push_back(absl::StrCat(
          user->name(), "=", xla::ShapeUtil::HumanString(user->shape())));
    }
    builder.AppendNodeAttribute(kUsers, absl::StrJoin(users_str, "\n"));
  }

  // Add operands info if it is a tuple instruction.
  // Because operands for non-root tuple are emitted.
  if (instruction->opcode() == xla::HloOpcode::kTuple) {
    std::vector<std::string> operands_str;
    operands_str.reserve(instruction->operand_count());
    for (const xla::HloInstruction* operand : instruction->operands()) {
      operands_str.push_back(absl::StrCat(
          operand->name(), "=", xla::ShapeUtil::HumanString(operand->shape())));
    }
    builder.AppendNodeAttribute(kOperands, absl::StrJoin(operands_str, "\n"));
  }

  // Instruction metadata.
  if (!instruction->metadata().op_name().empty()) {
    builder.AppendNodeAttribute(kOpName, instruction->metadata().op_name());
  }
  if (!instruction->metadata().op_type().empty()) {
    builder.AppendNodeAttribute(kOpType, instruction->metadata().op_type());
  }
  if (!instruction->metadata().source_file().empty()) {
    builder.AppendNodeAttribute(kSourceFile,
                                instruction->metadata().source_file());
  }
  if (instruction->metadata().source_line() != 0) {
    builder.AppendNodeAttribute(
        kSourceLine, absl::StrCat(instruction->metadata().source_line()));
  }
  if (instruction->metadata().stack_frame_id() != 0) {
    xla::StackFrameId frame_id{instruction->metadata().stack_frame_id()};
    std::string stack_lines;
    xla::HloModule* hlo_module = instruction->GetModule();
    while (frame_id.valid()) {
      xla::HloModule::StackFrame frame = hlo_module->get_stack_frame(frame_id);
      if (frame.empty()) {
        break;
      }
      absl::StrAppend(&stack_lines, frame.file_name, ":", frame.line, ":",
                      frame.column, "\n");
      frame_id = frame.parent_frame_id;
    }
    builder.AppendNodeAttribute(kSourceStack, stack_lines);
  }

  // Instruction address.
  builder.AppendNodeAttribute(kAddress, absl::StrFormat("%p", instruction));

  // Instruction backend config.
  if (instruction->has_backend_config()) {
    std::string backend_config_str = instruction->raw_backend_config_string();
    llvm::Expected<llvm::json::Value> backend_config_json =
        llvm::json::parse(backend_config_str);
    if (backend_config_json) {
      if (llvm::json::Object* obj = backend_config_json->getAsObject()) {
        // Removes custom_call_config as it's a binary string of a serialized
        // MLIR module, unreadable without deserialization. It will be
        // visualized separately.
        obj->erase("custom_call_config");
        // Removes any fields that are empty lists or objects to reduce clutter.
        RemoveEmptyFields(*backend_config_json);
        // If the backend config is not empty after removing empty fields,
        // then add it as a node attribute.
        if (!IsEmpty(*backend_config_json)) {
          std::string result;
          llvm::raw_string_ostream os(result);
          os << *backend_config_json;
          builder.AppendNodeAttribute(kBackendConfig, os.str());
        }
      } else {
        builder.AppendNodeAttribute(kBackendConfig, backend_config_str);
      }
    } else {
      // If backend config is not valid JSON, append it as is.
      builder.AppendNodeAttribute(kBackendConfig, backend_config_str);
    }
  }

  // Instruction extra attributes.
  const std::vector<std::string>& extra_attributes =
      instruction->ExtraAttributesToString(xla::HloPrintOptions::Default());
  if (!extra_attributes.empty()) {
    builder.AppendNodeAttribute(kExtraAttributes,
                                absl::StrJoin(extra_attributes, "\n"));
  }

  if (options.constant_folding) {
    // Collect information about inlined constant operands.
    std::vector<std::string> inlined_operands;
    for (int i = 0; i < instruction->operand_count(); ++i) {
      const xla::HloInstruction* operand = instruction->operand(i);
      std::string operand_str;
      // Inline direct constant operands, unless they are the root.
      if (operand->opcode() == xla::HloOpcode::kConstant &&
          !IsRootInstruction(operand)) {
        operand_str = StringifyConstant(
            xla::Cast<xla::HloConstantInstruction>(operand), operand->shape());
        // Inline fused broadcasts of effective scalar constants, unless they
        // are the root.
      } else if (IsFusedBroadcastOfConstantEffectiveScalar(operand) &&
                 !IsRootInstruction(operand)) {
        operand_str = StringifyConstant(
            xla::Cast<xla::HloConstantInstruction>(operand->operand(0)),
            operand->shape());
      }
      if (!operand_str.empty()) {
        if (instruction->operand_count() > 1) {
          inlined_operands.push_back(
              absl::StrFormat("operand %d = %s", i, operand_str));
        } else {
          inlined_operands.push_back(
              absl::StrFormat("operand = %s", operand_str));
        }
      }
    }
    // Add the collected inlined operand information as a node attribute.
    if (!inlined_operands.empty()) {
      builder.AppendNodeAttribute(kInlinedOperands,
                                  absl::StrJoin(inlined_operands, "\n"));
    }
  }

  // Attach get-tuple-element index if its define is a GTE and folded.
  if (options.get_tuple_element_folding) {
    std::vector<std::string> tuple_elements;
    for (int i = 0; i < instruction->operand_count(); ++i) {
      const xla::HloInstruction* operand = instruction->operand(i);
      if (IsGetTupleElement(options, operand)) {
        tuple_elements.push_back(absl::StrFormat(
            "operand %d: tuple-element %d of %s %s", i, operand->tuple_index(),
            operand->operand(0)->name(), GetShapeString(operand)));
      }
    }
    if (instruction->opcode() == xla::HloOpcode::kParameter &&
        instruction->IsFused()) {
      const xla::HloInstruction* param_input =
          instruction->parent()->FusionInstruction()->operand(
              instruction->parameter_number());
      if (param_input->opcode() == xla::HloOpcode::kGetTupleElement) {
        tuple_elements.push_back(absl::StrFormat(
            "tuple-element %d of %s %s", param_input->tuple_index(),
            param_input->operand(0)->name(), GetShapeString(param_input)));
      }
    }

    if (!tuple_elements.empty()) {
      builder.AppendNodeAttribute(kGetTupleElementIndex,
                                  absl::StrJoin(tuple_elements, "\n"));
    }
  }

  // Constant literal.
  if (instruction->IsConstant() &&
      xla::Cast<xla::HloConstantInstruction>(instruction)->HasLiteral()) {
    builder.AppendNodeAttribute(kLiteral, instruction->literal().ToString());
  }

  if (options.hide_async_collective_fusion_parameter) {
    if (IsAcfParameter(instruction)) {
      builder.AppendNodeAttribute(kHide, "true");
    }
  }
}

absl::Status BuildHloInstructionNode(
    const xla::HloInstruction* instruction, const HloAdapterOption& options,
    std::vector<std::string>& computation_stack, GraphNodeBuilder& builder,
    OutputEdges& output_edges, const ComputationExpand& computation_expand) {
  builder.SetNodeId(GetInstructionId(instruction));

  // Set namespace.
  builder.SetNodeName(GetComputationHierarchy(computation_stack));

  // Set node label.
  SetInstructionNodeLabel(instruction, builder);

  // Add incoming edges.
  absl::Status status = AddHloInstructionIncomingEdges(
      instruction, options, builder, output_edges, computation_expand);
  if (!status.ok()) {
    return status;
  }

  // Set node attributes.
  SetInstructionNodeAttributes(instruction, options, builder);

  return absl::OkStatus();
}

// Populates the outputs metadata for the given node.
void PopulateOutputsMetadata(
    GraphNodeBuilder& builder,
    const std::vector<const xla::HloInstruction*>& output_nodes) {
  for (int i = 0; i < output_nodes.size(); ++i) {
    builder.AppendAttrToMetadata(EdgeType::kOutput, i, kShapeWithLayout,
                                 builder.GetNodeAttribute(kShapeWithLayout));
  }
}

// Recursively builds a list of GraphNodeBuilders from a HLO computation and its
// subcomputations. The subcomputations, including the fusion computations, are
// represented with "namespace" feature in the Model Explorer.
//
// `instruction_node_builders`: the object that holds all GraphNodeBuilder.
//
// `computation`: the computation that is being built.
//
// `built_computations`: the computations that have been built.
//
// `computation_stack`: track current computation hierarchy.
//
// `node_filter`: decide which instruction to include.
//
// `computation_expand`: decide which computation to expand.
//
// `output_edges`: record all the edges from the instruction to its users.
//
// `group_node_attributes`: record all the attributes from the group node.
absl::Status HloComputationToGraphImpl(
    const xla::HloComputation& computation, const NodeFilter& node_filter,
    const ComputationExpand& computation_expand,
    const HloAdapterOption& options,
    absl::flat_hash_set<const xla::HloComputation*>& built_computations,
    std::vector<std::string>& computation_stack,
    std::vector<GraphNodeBuilder>& instruction_node_builders,
    OutputEdges& output_edges, GroupNodeAttributes& group_node_attributes) {
  if (built_computations.contains(&computation)) {
    return absl::OkStatus();
  }
  built_computations.insert(&computation);

  // Create a pinned node for the computation layer.
  GraphNodeBuilder builder;
  // Fusion computation is merged with its caller fusion instruction, make the
  // pinned node represents caller instruction instead of the computation.
  if (computation.FusionInstruction() != nullptr) {
    absl::Status status = BuildHloInstructionNode(
        computation.FusionInstruction(), options, computation_stack, builder,
        output_edges, computation_expand);
    if (!status.ok()) {
      return status;
    }
    builder.SetNodeLabel(GetInstructionId(computation.FusionInstruction()));
    builder.AppendNodeAttribute(kFusionComputation, computation.name());

    // Populate group node attributes from the pinned node.
    const std::string& group_name = builder.GetNodeName();
    for (const auto& attr : builder.GetNodeAttributes()) {
      group_node_attributes[group_name][attr.key] = attr.value;
    }
  } else {
    // Build the pinned node representing the computation.
    builder.SetNodeId(GetComputationId(&computation));
    builder.SetNodeName(GetComputationHierarchy(computation_stack));
    builder.SetNodeLabel(absl::StrCat("Computation \n", computation.name()));
  }
  builder.SetPinToGroupTop(true);
  instruction_node_builders.push_back(builder);

  for (const xla::HloInstruction* instruction :
       computation.MakeInstructionPostOrder()) {
    if (!node_filter(instruction)) {
      continue;
    }

    if (options.constant_folding && ShouldFoldConstant(instruction)) {
      continue;
    }

    if (instruction->opcode() == xla::HloOpcode::kFusion &&
        computation_expand(instruction,
                           instruction->fused_instructions_computation())) {
      // We do not construct a dedicated node for the variable assigned by
      // fusion op. Instead, we (1) build the fusion computation, (2) connect
      // the operands of the fusion computation to its parameters, and (3)
      // connect the ROOT of the fusion computation to its users.
      // (1) is done at this scope, (2) and (3) is done at
      // `AddHloInstructionIncomingEdges`.
      computation_stack.push_back(std::string(instruction->name()));
      absl::Status status = HloComputationToGraphImpl(
          *(instruction->fused_instructions_computation()), node_filter,
          computation_expand, options, built_computations, computation_stack,
          instruction_node_builders, output_edges, group_node_attributes);
      if (!status.ok()) {
        return status;
      }
      computation_stack.pop_back();
    } else if (IsGetTupleElement(options, instruction)) {
      continue;
    } else {
      // Build the hlo instruction node.
      GraphNodeBuilder builder;
      absl::Status status =
          BuildHloInstructionNode(instruction, options, computation_stack,
                                  builder, output_edges, computation_expand);
      if (!status.ok()) {
        return status;
      }

      // Convert subcomputations within the instruction to subgraphs.
      for (const xla::HloComputation* subcomputation :
           instruction->called_computations()) {
        if (!computation_expand(instruction, subcomputation)) {
          continue;
        }
        computation_stack.push_back(std::string(subcomputation->name()));
        int prev_node_count = instruction_node_builders.size();
        absl::Status status = HloComputationToGraphImpl(
            *subcomputation, node_filter, computation_expand, options,
            built_computations, computation_stack, instruction_node_builders,
            output_edges, group_node_attributes);
        if (!status.ok()) {
          return status;
        }
        int cur_node_count = instruction_node_builders.size();
        computation_stack.pop_back();

        // If some of the nodes from subcomputation are included, connect the
        // root of subcomputation to the caller instruction. Since the
        // subcomputation is traversed in post order, the last instruction is
        // the root of the subcomputation, and must be included.
        if (cur_node_count > prev_node_count) {
          const std::string last_node_id =
              instruction_node_builders.back().GetNodeId();
          builder.AppendEdgeInfo(last_node_id, "0", "0");
          output_edges[last_node_id].push_back(instruction);
        }
      }

      instruction_node_builders.push_back(builder);
    }
  }

  return absl::OkStatus();
}

// Convert to json.
std::string GraphCollectionToJson(GraphCollection& collection) {
  llvm::json::Value json_result(collection.Json());
  std::string json_output;
  llvm::raw_string_ostream json_ost(json_output);
  json_ost << llvm::formatv("{0:2}", json_result);
  return json_output;
}

}  // namespace

std::string GetInstructionId(const xla::HloInstruction* instruction) {
  return absl::StrCat(instruction->name());
}

std::string GetComputationId(const xla::HloComputation* computation) {
  return absl::StrCat(computation->name());
}

absl::StatusOr<GraphCollection> HloToGraph(
    const xla::HloComputation& computation, const NodeFilter& node_filter,
    const ComputationExpand& computation_expand,
    const HloAdapterOption& options) {
  Graph graph;
  Subgraph subgraph(std::string(computation.name()));
  graph.subgraphs.push_back(std::move(subgraph));
  absl::flat_hash_set<const xla::HloComputation*> built_computations;
  std::vector<std::string> computation_stack;
  computation_stack.push_back(std::string(computation.name()));
  OutputEdges output_edges;
  std::vector<GraphNodeBuilder> instruction_node_builders;

  absl::Status status = HloComputationToGraphImpl(
      computation, node_filter, computation_expand, options, built_computations,
      computation_stack, instruction_node_builders, output_edges,
      graph.subgraphs.back().group_node_attributes);
  if (!status.ok()) {
    return status;
  }

  for (GraphNodeBuilder& builder : instruction_node_builders) {
    if (const auto& it = output_edges.find(builder.GetNodeId());
        it != output_edges.end()) {
      PopulateOutputsMetadata(builder, it->second);
    }
    graph.subgraphs.back().nodes.push_back(std::move(builder).Build());
  }

  GraphCollection collection;
  collection.graphs.push_back(std::move(graph));
  return collection;
}

absl::StatusOr<std::string> HloGraphAdapter(
    const xla::HloComputation& computation, const HloAdapterOption& options) {
  const NodeFilter node_filter = [&](const xla::HloInstruction* instruction) {
    return true;
  };

  absl::StatusOr<GraphCollection> graph_collection_or =
      HloToGraph(computation, node_filter, options);
  if (!graph_collection_or.ok()) {
    return graph_collection_or.status();
  }
  GraphCollection graph_collection = std::move(graph_collection_or).value();

  return GraphCollectionToJson(graph_collection);
}

absl::StatusOr<std::string> HloGraphAdapter(
    const xla::HloInstruction& instruction, const int radius,
    const HloAdapterOption& options) {
  const NodeFilter node_filter =
      MakeInstructionRadiusAroundFilter(&instruction, radius);

  absl::StatusOr<GraphCollection> graph_collection_or =
      HloToGraph(*instruction.parent(), node_filter, options);
  if (!graph_collection_or.ok()) {
    return graph_collection_or.status();
  }
  GraphCollection graph_collection = std::move(graph_collection_or).value();

  return GraphCollectionToJson(graph_collection);
}

}  // namespace visualization_client
}  // namespace tooling
