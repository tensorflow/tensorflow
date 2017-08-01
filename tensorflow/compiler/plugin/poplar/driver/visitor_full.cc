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

#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/fuse_ops.h"
#include "tensorflow/compiler/plugin/poplar/driver/tensor.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitor_full.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
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

FullVisitor::FullVisitor(poplar::Graph* graph,
                         CompilerResources& res)
        : BaseVisitor(graph, res) {}

Status FullVisitor::HandleConcatenate(
        HloInstruction* inst,
        tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
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

Status FullVisitor::HandleDot(
        HloInstruction* inst,
        HloInstruction* lhs,
        HloInstruction* rhs) {
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(prog,
                      CreateMatMulOp(*graph_,
                                     resources_,
                                     inst,
                                     GetOutputShape(inst),
                                     tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status FullVisitor::HandleConvolution(
        HloInstruction* inst,
        HloInstruction* lhs,
        HloInstruction* rhs,
        const Window& window) {
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(prog,
                      CreateConv2D(*graph_,
                                   resources_,
                                   inst,
                                   GetOutputShape(inst),
                                   tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status FullVisitor::HandleReverse(
        HloInstruction* inst,
        HloInstruction* operand) {
  poplar::Tensor t;
  TF_ASSIGN_OR_RETURN(t, FindInstructionInput(tensor_map, inst, 0, 0));
  TF_ASSIGN_OR_RETURN(t, ReverseTensor(t, inst->dimensions()));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, t));
  return Status::OK();
}

Status FullVisitor::HandleGetTupleElement(
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

Status FullVisitor::HandleReduce(
        HloInstruction* inst,
        HloInstruction* arg,
        HloInstruction* init_value,
        tensorflow::gtl::ArraySlice<int64> dimensions,
        HloComputation* function) {
  if (IsReducableArtithmetic(inst, function)) {
    poplar::program::Program prog;
    TF_ASSIGN_OR_RETURN(prog,
                        CreateSimpleReduction(*graph_,
                                              resources_,
                                              inst,
                                              GetOutputShape(inst),
                                              tensor_map));
    sequence.add(prog);
    return Status::OK();
  }
  return Unimplemented(inst);
}

Status FullVisitor::HandleBitcast(HloInstruction* inst) {
  if (LayoutUtil::LayoutsInShapesEqual(inst->operand(0)->shape(),
                                       GetOutputShape(inst))) {
    return HandleReshape(inst);
  } else {
    return HandleTranspose(inst);
  }
}

Status FullVisitor::HandleBroadcast(HloInstruction* inst) {
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

Status FullVisitor::HandleReshape(HloInstruction* inst) {
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, FindInstructionInput(tensor_map, inst, 0, 0));
  std::vector<size_t> dims(PoplarShapeFromXlaShape(GetOutputShape(inst)));
  out = out.reshape(dims);
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status FullVisitor::HandleTranspose(HloInstruction* inst) {
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, FindInstructionInput(tensor_map, inst, 0, 0));
  std::vector<unsigned> permutation(
          convert_array<std::vector<unsigned>>(inst->dimensions()));
  out = out.dimShuffle(permutation);
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status FullVisitor::HandleFusion(HloInstruction* inst) {
  switch (static_cast<int>(inst->fusion_kind())) {
    case FUSED_SLICE_UPDATE:
    {
      poplar::program::Program prog;
      TF_ASSIGN_OR_RETURN(prog,
                          CreateSliceUpdateOp(*graph_,
                                              resources_,
                                              inst,
                                              GetOutputShape(inst),
                                              tensor_map));
      sequence.add(prog);
      return Status::OK();
    }
    case FUSED_SLICE:
    {
      poplar::program::Program prog;
      TF_ASSIGN_OR_RETURN(prog,
                          CreateSliceOp(*graph_,
                                        resources_,
                                        inst,
                                        GetOutputShape(inst),
                                        tensor_map));
      sequence.add(prog);
      return Status::OK();
    }
    case FUSED_RELU:
    {
      poplar::program::Program prog;
      TF_ASSIGN_OR_RETURN(prog,
                          CreateReluOp(*graph_,
                                       resources_,
                                       inst,
                                       GetOutputShape(inst),
                                       tensor_map));
      sequence.add(prog);
      return Status::OK();
    }
    case FUSED_SIGMOID:
    {
      poplar::program::Program prog;
      TF_ASSIGN_OR_RETURN(prog,
                          CreateSigmoidOp(*graph_,
                                          resources_,
                                          inst,
                                          GetOutputShape(inst),
                                          tensor_map));
      sequence.add(prog);
      return Status::OK();
    }
    case FUSED_BIASADD_BROADCAST:
    case FUSED_BIASADD:
    {
      poplar::program::Program prog;
      TF_ASSIGN_OR_RETURN(prog,
                          CreateBiasAddOp(*graph_,
                                          resources_,
                                          inst,
                                          GetOutputShape(inst),
                                          tensor_map));
      sequence.add(prog);
      return Status::OK();
    }
    case FUSED_ZERO_PAD:
    {
      const HloInstruction* root = inst->fused_expression_root();
      const PaddingConfig& cfg(root->padding_config());
      poplar::Tensor out;
      TF_ASSIGN_OR_RETURN(out, FindInstructionInput(tensor_map, inst, 0, 0));
      TF_ASSIGN_OR_RETURN(out, PadWithConstantZero(*graph_, cfg, out));
      TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
      return Status::OK();
    }
    case FUSED_TRUNCATED_NORMAL_WITH_SCALE:
    case FUSED_TRUNCATED_NORMAL:
    case FUSED_RANDOM_NORMAL_WITH_SCALE:
    case FUSED_RANDOM_UNIFORM_WITH_SCALE:
    case FUSED_RANDOM_NORMAL:
    case FUSED_RANDOM_UNIFORM:
    case FUSED_BERNOULLI:
    {
      poplar::program::Program prog;
      TF_ASSIGN_OR_RETURN(prog,
                          CreateRandomOp(*graph_,
                                         resources_,
                                         inst,
                                         GetOutputShape(inst),
                                         tensor_map));
      sequence.add(prog);
      return Status::OK();
    }
    case FUSED_AVG_POOL_SAME:
    case FUSED_AVG_POOL_VALID:
    {
      poplar::program::Program prog;
      TF_ASSIGN_OR_RETURN(prog,
                          CreatePoplibsWindowReduction(*graph_,
                                                       resources_,
                                                       inst,
                                                       GetOutputShape(inst),
                                                       tensor_map));
      sequence.add(prog);
      return Status::OK();
    }
    default:
      return Unimplemented(inst);
  }
};

Status FullVisitor::HandleCall(HloInstruction* inst) {
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(prog,
                      CreateCallOp(*graph_,
                                   resources_,
                                   inst,
                                   GetOutputShape(inst),
                                   tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status FullVisitor::HandleSlice(
        HloInstruction* inst,
        HloInstruction* operand) {
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(out, FindInstructionInput(tensor_map, inst, 0, 0));
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
        out = out.slice(end[d]+1, begin[d]+1, d);
        out = out.reverse(d);
        out = out.subSample(-strides[d], d);
      }
    }
  }
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status FullVisitor::HandleDynamicSlice(
        HloInstruction* inst,
        HloInstruction* operand,
        HloInstruction* start_indices) {
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(prog,
                      CreateDynamicSliceOp(*graph_,
                                           resources_,
                                           inst,
                                           GetOutputShape(inst),
                                           tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status FullVisitor::HandleDynamicUpdateSlice(
        HloInstruction* inst,
        HloInstruction* operand,
        HloInstruction* update,
        HloInstruction* start_indices) {
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(prog,
                      CreateDynamicSliceUpdateOp(*graph_,
                                                 resources_,
                                                 inst,
                                                 GetOutputShape(inst),
                                                 tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status FullVisitor::HandleTuple(
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

Status FullVisitor::HandleMap(
        HloInstruction* inst,
        tensorflow::gtl::ArraySlice<HloInstruction*> operands,
        HloComputation* function,
        tensorflow::gtl::ArraySlice<HloInstruction*> static_operands) {
  bool simple_parallel;
  TF_ASSIGN_OR_RETURN(simple_parallel,
                      IsParallelMap(inst, function));
  if (simple_parallel) {
    poplar::program::Program prog;
    TF_ASSIGN_OR_RETURN(prog,
                        CreateParallelMap(*graph_,
                                          resources_,
                                          inst,
                                          GetOutputShape(inst),
                                          tensor_map));
    sequence.add(prog);
    return Status::OK();
  }
  return Unimplemented(inst);
}

Status FullVisitor::HandleReduceWindow(
        HloInstruction* inst,
        HloInstruction* operand,
        const Window& window,
        HloComputation* function) {
  if (IsPoplibsPool(inst, function)) {
    poplar::program::Program prog;
    TF_ASSIGN_OR_RETURN(prog,
                        CreatePoplibsWindowReduction(*graph_,
                                                     resources_,
                                                     inst,
                                                     GetOutputShape(inst),
                                                     tensor_map));
    sequence.add(prog);
    return Status::OK();
  }
  if (IsReducableArtithmetic(inst, function)) {
    poplar::program::Program prog;
    TF_ASSIGN_OR_RETURN(prog,
                        CreateSimpleWindowReduction(*graph_,
                                                    resources_,
                                                    inst,
                                                    GetOutputShape(inst),
                                                    tensor_map));
    sequence.add(prog);
    return Status::OK();
  }
  return Unimplemented(inst);
}

Status FullVisitor::HandleSelectAndScatter(HloInstruction* inst) {
  if (IsSimpleSelection(inst, inst->select()) &&
      IsReducableArtithmetic(inst, inst->scatter())) {
    poplar::program::Program prog;
    TF_ASSIGN_OR_RETURN(prog,
                        CreateSimpleSelectAndScatter(*graph_,
                                                     resources_,
                                                     inst,
                                                     GetOutputShape(inst),
                                                     tensor_map));
    sequence.add(prog);
    return Status::OK();
  }
  return Unimplemented(inst);
}

Status FullVisitor::HandleWhile(HloInstruction* inst) {
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(prog,
                      CreateWhileOp(*graph_,
                                    resources_,
                                    inst,
                                    GetOutputShape(inst),
                                    tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status FullVisitor::HandlePad(HloInstruction* inst) {
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
