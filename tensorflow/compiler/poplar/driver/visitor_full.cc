/* Copyright 2017 Graphcore Ltd
 */

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/poplar/driver/ops.h"
#include "tensorflow/compiler/poplar/driver/tensor.h"
#include "tensorflow/compiler/poplar/driver/visitor_full.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/stream_executor/lib/strcat.h"
#include "tensorflow/core/lib/core/errors.h"

#include <stddef.h>
#include <string.h>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/stream_executor/lib/initialize.h"

#include <poplar/Tensor.hpp>
#include <poplar/GraphElements.hpp>
#include <poplar/Engine.hpp>
#include <poplar/exceptions.hpp>

namespace se = ::perftools::gputools;

namespace xla {
namespace poplarplugin {

PoplarFullVisitor::PoplarFullVisitor(poplar::Graph* graph)
        : PoplarBaseVisitor(graph) {}

Status PoplarFullVisitor::HandleConcatenate(
        HloInstruction* inst,
        tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  VLOG(1) << inst->ToString();
  int64 dimension(inst->concatenate_dimension());
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, FindInstructionInput(tensor_map, inst, 0, 0));
  out = out.slice(0, 0, dimension);
  for (auto op : operands) {
    poplar::Tensor t;
    TF_ASSIGN_OR_RETURN(t, FindInstructionOutput(tensor_map, op, 0));
    out = poplar::concat(out, t, dimension);
  }
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status PoplarFullVisitor::HandleDot(
        HloInstruction* inst,
        HloInstruction* lhs,
        HloInstruction* rhs) {
  VLOG(1) << inst->ToString();
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(prog,
                      CreateMatMulOp(*graph_,
                                     inst,
                                     GetOutputShape(inst),
                                     tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status PoplarFullVisitor::HandleConvolution(
        HloInstruction* inst,
        HloInstruction* lhs,
        HloInstruction* rhs,
        const Window& window) {
  VLOG(1) << inst->ToString();
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(prog,
                      CreateConv2D(*graph_,
                                   inst,
                                   GetOutputShape(inst),
                                   tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status PoplarFullVisitor::HandleCrossReplicaSum(HloInstruction* inst) {
  VLOG(1) << inst->ToString();
  return Unimplemented(inst);
}

Status PoplarFullVisitor::HandleReverse(
        HloInstruction* inst,
        HloInstruction* operand) {
  poplar::Tensor t;
  TF_ASSIGN_OR_RETURN(t, FindInstructionInput(tensor_map, inst, 0, 0));
  TF_ASSIGN_OR_RETURN(t, ReverseTensor(t, inst->dimensions()));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, t));
  return Status::OK();
}

Status PoplarFullVisitor::HandleSort(
        HloInstruction* inst,
        HloInstruction* operand) {
  return Unimplemented(inst);
}

Status PoplarFullVisitor::HandleGetTupleElement(
        HloInstruction* inst,
        HloInstruction* operand) {
  poplar::Tensor t;
  TF_ASSIGN_OR_RETURN(t, FindInstructionInput(tensor_map,
                                              inst,
                                              0,
                                              inst->tuple_index()));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, t));
  return Status::OK();
}

Status PoplarFullVisitor::HandleReduce(
        HloInstruction* inst,
        HloInstruction* arg,
        HloInstruction* init_value,
        tensorflow::gtl::ArraySlice<int64> dimensions,
        HloComputation* function) {
  bool simple_reduction;
  TF_ASSIGN_OR_RETURN(simple_reduction,
                      IsComputationReducableArtithmetic(function));
  if (simple_reduction) {
    poplar::program::Program prog;
    TF_ASSIGN_OR_RETURN(prog,
                        CreateSimpleReduction(*graph_,
                                              inst,
                                              GetOutputShape(inst),
                                              tensor_map));
    sequence.add(prog);
    return Status::OK();
  }
  return Unimplemented(inst);
}

Status PoplarFullVisitor::HandleBitcast(HloInstruction* inst) {
  if (LayoutUtil::LayoutsInShapesEqual(inst->operand(0)->shape(),
                                       GetOutputShape(inst))) {
    return HandleReshape(inst);
  } else {
    return HandleTranspose(inst);
  }
}

Status PoplarFullVisitor::HandleBroadcast(HloInstruction* inst) {
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, FindInstructionInput(tensor_map, inst, 0, 0));
  TF_ASSIGN_OR_RETURN(out, BroadcastTensor(out,
                                           GetOutputShape(inst),
                                           inst->dimensions()));
  std::vector<size_t> dims(PoplarShapeFromXlaShape(GetOutputShape(inst)));
  out = out.reshape(dims);
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status PoplarFullVisitor::HandleReshape(HloInstruction* inst) {
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, FindInstructionInput(tensor_map, inst, 0, 0));
  std::vector<size_t> dims(PoplarShapeFromXlaShape(GetOutputShape(inst)));
  out = out.reshape(dims);
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status PoplarFullVisitor::HandleTranspose(HloInstruction* inst) {
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, FindInstructionInput(tensor_map, inst, 0, 0));
  std::vector<unsigned> permutation(
          convert_array<std::vector<unsigned>>(inst->dimensions()));
  out = out.dimShuffle(permutation);
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status PoplarFullVisitor::HandleFusion(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status PoplarFullVisitor::HandleCall(
        HloInstruction* inst) {
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(prog,
                      CreateCallOp(*graph_,
                                   inst,
                                   GetOutputShape(inst),
                                   tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status PoplarFullVisitor::HandleCustomCall(
        HloInstruction* inst,
        tensorflow::gtl::ArraySlice<HloInstruction*> operands,
        tensorflow::StringPiece custom_call_target) {
  return Unimplemented(inst);

}

Status PoplarFullVisitor::HandleSlice(
        HloInstruction* inst,
        HloInstruction* operand) {
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, FindInstructionInput(tensor_map, inst, 0, 0));
  std::vector<std::size_t> begin(
          convert_array<std::vector<std::size_t>>(inst->slice_starts()));
  std::vector<std::size_t> end(
          convert_array<std::vector<std::size_t>>(inst->slice_limits()));
  out = out.slice(begin, end);
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status PoplarFullVisitor::HandleDynamicSlice(
        HloInstruction* inst,
        tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  return Unimplemented(inst);
}

Status PoplarFullVisitor::HandleDynamicUpdateSlice(
        HloInstruction* inst,
        HloInstruction* operand,
        HloInstruction* update,
        HloInstruction* start_indices) {
  return Unimplemented(inst);
}

Status PoplarFullVisitor::HandleTuple(
        HloInstruction* inst,
        tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  uint64 operand_count(inst->operand_count());
  for (uint64 i=0; i<operand_count; i++) {
    poplar::Tensor t;
    TF_ASSIGN_OR_RETURN(t, FindInstructionInput(tensor_map, inst, i, 0));
    TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, i, t));
  }
  return Status::OK();
}

Status PoplarFullVisitor::HandleMap(
        HloInstruction* inst,
        tensorflow::gtl::ArraySlice<HloInstruction*> operands,
        HloComputation* function,
        tensorflow::gtl::ArraySlice<HloInstruction*> static_operands) {
  bool simple_parallel;
  TF_ASSIGN_OR_RETURN(simple_parallel,
                      IsComputationParallelMap(function));
  if (simple_parallel) {
    poplar::program::Program prog;
    TF_ASSIGN_OR_RETURN(prog,
                        CreateParallelMap(*graph_,
                                          inst,
                                          GetOutputShape(inst),
                                          tensor_map));
    sequence.add(prog);
    return Status::OK();
  }
  return port::Status(port::error::UNIMPLEMENTED,
                      port::StrCat(inst->name(),
                                   " not supported by poplar"));
}

Status PoplarFullVisitor::HandleReduceWindow(
        HloInstruction* inst,
        HloInstruction* operand,
        const Window& window,
        HloComputation* function) {
  bool simple_reduction;
  TF_ASSIGN_OR_RETURN(simple_reduction,
                      IsComputationReducableArtithmetic(function));
  if (simple_reduction) {
    poplar::program::Program prog;
    TF_ASSIGN_OR_RETURN(prog,
                        CreateSimpleWindowReduction(*graph_,
                                                    inst,
                                                    GetOutputShape(inst),
                                                    tensor_map));
    sequence.add(prog);
    return Status::OK();
  }
  return Unimplemented(inst);
}

Status PoplarFullVisitor::HandleSelectAndScatter(HloInstruction* inst) {
  bool simple_selection;
  TF_ASSIGN_OR_RETURN(simple_selection,
                      IsComputationSimpleSelection(inst->select()));
  bool simple_reduction;
  TF_ASSIGN_OR_RETURN(simple_reduction,
                      IsComputationReducableArtithmetic(inst->scatter()));
  if (simple_selection && simple_reduction) {
    poplar::program::Program prog;
    TF_ASSIGN_OR_RETURN(prog,
                        CreateSimpleSelectAndScatter(*graph_,
                                                     inst,
                                                     GetOutputShape(inst),
                                                     tensor_map));
    sequence.add(prog);
    return Status::OK();
  }
  return Unimplemented(inst);
}

Status PoplarFullVisitor::HandleWhile(HloInstruction* inst) {
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(prog,
                      CreateWhileOp(*graph_,
                                    inst,
                                    GetOutputShape(inst),
                                    tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status PoplarFullVisitor::HandlePad(HloInstruction* inst) {
  poplar::Tensor out;
  poplar::Tensor pad;
  TF_ASSIGN_OR_RETURN(out, FindInstructionInput(tensor_map, inst, 0, 0));
  TF_ASSIGN_OR_RETURN(pad, FindInstructionInput(tensor_map, inst, 1, 0));
  TF_ASSIGN_OR_RETURN(out, PadTensor(inst->padding_config(), out, pad));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}


}  // namespace poplarplugin
}  // namespace xla
