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

#include "tensorflow/compiler/plugin/poplar/driver/entry_visitor.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_resources.h"
#include "tensorflow/compiler/plugin/poplar/driver/executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"

namespace xla {
namespace poplarplugin {

// Arrange for a FIFO and Copy for outputs which can be streamed
Status EntryVisitor::StreamOutputs(HloInstruction* inst, uint64 start_idx,
                                   OutVector outputs) {
  for (int o = 0; o < outputs.size(); o++) {
    int64 index = start_idx + o;
    output_streamed[index] = true;

    HloComputation* comp = inst->parent();
    HloModule* mod = comp->parent();
    auto* layout = mod->mutable_entry_computation_layout();
    auto shapes = FlattenedXlaShape(layout->result_shape());

    poplar::Tensor out = ConvertFromDeviceLayout(shapes[index], outputs[o]);

    auto fifo = graph_.addDeviceToHostFIFO(
        GetOutputCopyHandle(index), out.elementType(), out.numElements());

    sequence.add(poplar::program::Copy(out, fifo));
  }
  return Status::OK();
}

Status EntryVisitor::HandleParameter(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();

  auto num_streaming = inst->parent()->num_parameters() -
                       resources_.annotations.num_resource_inputs;

  parameter_streamed[inst->parameter_number()] =
      (inst->parameter_number() < num_streaming);

  std::vector<Shape> shapes = FlattenedXlaShape(inst->shape());
  std::vector<Shape> module_shapes;

  HloModule* module = inst->parent()->parent();
  ComputationLayout* layout = module->mutable_entry_computation_layout();
  if (layout->parameter_count() > inst->parameter_number()) {
    const Shape& mod_shape = layout->parameter_shape(inst->parameter_number());
    module_shapes = FlattenedXlaShape(mod_shape);
  }

  for (unsigned i = 0; i < shapes.size(); i++) {
    poplar::program::Sequence& seq =
        parameter_streamed[inst->parameter_number()] ? sequence
                                                     : host_to_device;

    poplar::Tensor out;
    TF_ASSIGN_OR_RETURN(
        out, AddTensor(graph_, std::make_pair(inst, i), shapes[i], resources_));
    TF_ASSIGN_OR_RETURN(out, AddOutputTensor(graph_, resources_, seq,
                                             tensor_map, inst, i, out));

    if (!UseSyntheticData()) {
      if (module_shapes.size() > i) {
        if (!LayoutUtil::IsMonotonicWithDim0Major(module_shapes[i].layout())) {
          // Host tensor needs to be host layout
          out = ConvertFromDeviceLayout(module_shapes[i], out);
          non_standard_parameter_layout.insert(inst);
        }
      }
      auto fifo = graph_.addHostToDeviceFIFO(
          GetInputCopyHandle(inst->parameter_number(), i), out.elementType(),
          out.numElements());

      seq.add(poplar::program::Copy(fifo, out));
    }
  }
  return Status::OK();
}

Status EntryVisitor::FinishVisit(HloInstruction* inst) {
  HloComputation* comp = inst->parent();

  auto outputs = FindInstructionOutputs(tensor_map, inst);

  auto* layout = comp->parent()->mutable_entry_computation_layout();
  std::vector<Shape> shapes = FlattenedXlaShape(layout->result_shape());

  for (size_t o = 0; o < outputs.size(); o++) {
    // For each output, if there is an identical input, put it into the map
    for (int64 i = 0; i < comp->num_parameters(); i++) {
      HloInstruction* param = comp->parameter_instruction(i);
      if (non_standard_parameter_layout.count(inst) == 0) {
        auto in = FindInstructionOutputs(tensor_map, param);

        // Only non-tuple inputs are considered for input<->output mapping
        if (in.size() == 1 && in[0] == outputs[o]) {
          output_map[o] = i;
        }
      }
    }

    if (!output_streamed[o] && !UseSyntheticData()) {
      poplar::Tensor out = ConvertFromDeviceLayout(shapes[o], outputs[o]);

      auto fifo = graph_.addDeviceToHostFIFO(
          GetOutputCopyHandle(o), out.elementType(), out.numElements());

      device_to_host.add(poplar::program::Copy(out, fifo, true));
    }
  }

  if (inst->opcode() == HloOpcode::kParameter) {
    all_outputs_are_parameters = true;
  } else if (inst->opcode() == HloOpcode::kTuple) {
    all_outputs_are_parameters = true;
    for (auto op : inst->operands()) {
      all_outputs_are_parameters &= (op->opcode() == HloOpcode::kParameter);
    }
  }

  // Streamed inputs<->outputs cannot be inplace updates
  if (!all_outputs_are_parameters) {
    for (size_t o = 0; o < outputs.size(); o++) {
      if (output_streamed[o]) {
        output_map.erase(o);
      }
    }
  }

  PrintTensorMapping(graph_, tensor_map);
  tensor_map.clear();

  all_outputs_are_parameters &= (non_standard_parameter_layout.size() == 0);

  return Status::OK();
}

Status EntryVisitor::Postprocess(HloInstruction* inst) {
  // After processing each instruction, check if its output can be streamed
  // off the device
  if (UseSyntheticData()) {
    return Status::OK();
  }
  const auto* root = inst->parent()->root_instruction();
  auto num_streaming = FlattenedXlaShape(root->shape()).size() -
                       resources_.annotations.num_resource_outputs;
  const auto& outputs = FindInstructionOutputs(tensor_map, inst);
  if (root->opcode() == HloOpcode::kTuple) {
    for (int i = 0; i < root->operand_count(); i++) {
      if (root->operand(i) == inst) {
        if (i < num_streaming) {
          auto pair = FindTupleInputIndices(root, i);
          if (pair.second - pair.first == outputs.size()) {
            StreamOutputs(inst, pair.first, outputs);
          }
        }
      }
    }
  } else if (inst == inst->parent()->root_instruction() && num_streaming) {
    StreamOutputs(inst, 0, outputs);
  }
  return Status::OK();
}

const OutputMap& EntryVisitor::GetOutputMap() { return output_map; }

const std::vector<bool>& EntryVisitor::GetParameterStreamed() {
  return parameter_streamed;
}

const std::vector<bool>& EntryVisitor::GetOutputStreamed() {
  return output_streamed;
}

const bool EntryVisitor::AreAllOutputsParameters() {
  return all_outputs_are_parameters;
}

const poplar::program::Sequence& EntryVisitor::GetHostToDevice() {
  return host_to_device;
}
const poplar::program::Sequence& EntryVisitor::GetDeviceToHost() {
  return device_to_host;
}

}  // namespace poplarplugin
}  // namespace xla
