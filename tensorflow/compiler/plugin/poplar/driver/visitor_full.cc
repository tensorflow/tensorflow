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

#include "tensorflow/compiler/plugin/poplar/driver/visitor_full.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/util.h"

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/lib/strcat.h"

#include <stddef.h>
#include <string.h>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/stream_executor/lib/initialize.h"

#include <poplar/Engine.hpp>
#include <poplar/GraphElements.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/exceptions.hpp>

namespace se = ::stream_executor;

namespace xla {
namespace poplarplugin {

FullVisitor::FullVisitor(poplar::Graph& graph, CompilerResources& res)
    : BaseVisitor(graph, res) {}

Status FullVisitor::HandleConcatenate(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  int64 dimension(inst->concatenate_dimension());
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, FindInstructionInput(tensor_map, inst, 0));
  for (int i = 1; i < inst->operand_count(); i++) {
    poplar::Tensor t;
    TF_ASSIGN_OR_RETURN(t, FindInstructionInput(tensor_map, inst, i));
    out = poplar::concat(out, t, dimension);
  }
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status FullVisitor::HandleDot(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(
      prog, CreateMatMulForDotOp(graph_, resources_, inst, GetOutputShape(inst),
                                 tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status FullVisitor::HandleConvolution(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(prog, CreateConv2D(graph_, resources_, inst,
                                         GetOutputShape(inst), tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status FullVisitor::HandleReverse(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::Tensor t;
  TF_ASSIGN_OR_RETURN(t, FindInstructionInput(tensor_map, inst, 0));
  TF_ASSIGN_OR_RETURN(t, ReverseTensor(t, inst->dimensions()));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, t));
  return Status::OK();
}

Status FullVisitor::HandleReduce(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  if (IsReducableArtithmetic(inst->to_apply())) {
    poplar::program::Program prog;
    TF_ASSIGN_OR_RETURN(
        prog, CreateSimpleReduction(graph_, resources_, inst,
                                    GetOutputShape(inst), tensor_map));
    sequence.add(prog);
    return Status::OK();
  }
  return Unimplemented(inst);
}

Status FullVisitor::HandleBitcast(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  if (LayoutUtil::LayoutsInShapesEqual(inst->operand(0)->shape(),
                                       GetOutputShape(inst))) {
    return HandleReshape(inst);
  } else {
    return HandleTranspose(inst);
  }
}

Status FullVisitor::HandleBroadcast(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, FindInstructionInput(tensor_map, inst, 0));
  TF_ASSIGN_OR_RETURN(
      out, BroadcastTensor(out, GetOutputShape(inst), inst->dimensions()));
  std::vector<size_t> dims(PoplarShapeFromXlaShape(GetOutputShape(inst)));
  out = out.reshape(dims);
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status FullVisitor::HandleReshape(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, FindInstructionInput(tensor_map, inst, 0));
  std::vector<size_t> dims(PoplarShapeFromXlaShape(GetOutputShape(inst)));
  out = out.reshape(dims);
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status FullVisitor::HandleTranspose(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, FindInstructionInput(tensor_map, inst, 0));
  std::vector<unsigned> permutation(
      convert_array<std::vector<unsigned>>(inst->dimensions()));
  out = out.dimShuffle(permutation);
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status FullVisitor::HandleSlice(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, FindInstructionInput(tensor_map, inst, 0));
  std::vector<std::size_t> begin(
      convert_array<std::vector<std::size_t>>(inst->slice_starts()));
  std::vector<std::size_t> end(
      convert_array<std::vector<std::size_t>>(inst->slice_limits()));
  std::vector<int64> strides(inst->slice_strides());
  bool simple(true);
  for (std::size_t s : strides) {
    simple &= (s == 1);
  }
  if (simple) {
    out = out.slice(begin, end);
  } else {
    for (size_t d = 0; d < strides.size(); d++) {
      int64 s = strides[d];
      if (s > 0) {
        out = out.slice(begin[d], end[d], d);
        out = out.subSample(strides[d], d);
      } else {
        out = out.slice(end[d] + 1, begin[d] + 1, d);
        out = out.reverse(d);
        out = out.subSample(-strides[d], d);
      }
    }
  }
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status FullVisitor::HandleDynamicSlice(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(
      prog, CreateDynamicSliceOp(graph_, resources_, inst, GetOutputShape(inst),
                                 tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status FullVisitor::HandleDynamicUpdateSlice(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(
      prog, CreateDynamicSliceUpdateOp(graph_, resources_, inst,
                                       GetOutputShape(inst), tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status FullVisitor::HandleReduceWindow(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  if (IsPoplibsPool(inst, inst->to_apply())) {
    poplar::program::Program prog;
    TF_ASSIGN_OR_RETURN(
        prog, CreatePoplibsWindowReduction(graph_, resources_, inst,
                                           GetOutputShape(inst), tensor_map));
    sequence.add(prog);
    return Status::OK();
  }
  if (IsReducableArtithmetic(inst->to_apply())) {
    poplar::program::Program prog;
    TF_ASSIGN_OR_RETURN(
        prog, CreateSimpleWindowReduction(graph_, resources_, inst,
                                          GetOutputShape(inst), tensor_map));
    sequence.add(prog);
    return Status::OK();
  }
  return Unimplemented(inst);
}

Status FullVisitor::HandleSelectAndScatter(HloInstruction* inst) {
  if (IsSimpleSelection(inst->select()) &&
      IsReducableArtithmetic(inst->scatter())) {
    VLOG(1) << "Processing " << inst->name();
    poplar::program::Program prog;
    TF_ASSIGN_OR_RETURN(
        prog, CreateSimpleSelectAndScatter(graph_, resources_, inst,
                                           GetOutputShape(inst), tensor_map));
    sequence.add(prog);
    return Status::OK();
  }
  return Unimplemented(inst);
}

Status FullVisitor::HandleWhile(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(prog, CreateWhileOp(graph_, resources_, inst,
                                          GetOutputShape(inst), tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status FullVisitor::HandlePad(HloInstruction* inst) {
  VLOG(1) << "Processing " << inst->name();
  poplar::Tensor out;
  poplar::Tensor pad;
  TF_ASSIGN_OR_RETURN(out, FindInstructionInput(tensor_map, inst, 0));
  TF_ASSIGN_OR_RETURN(pad, FindInstructionInput(tensor_map, inst, 1));
  TF_ASSIGN_OR_RETURN(out, PadTensor(inst->padding_config(), out, pad));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

}  // namespace poplarplugin
}  // namespace xla
