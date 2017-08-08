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
#include "tensorflow/compiler/plugin/poplar/driver/visitor_base.h"

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

BaseVisitor::BaseVisitor(poplar::Graph* graph,
                         CompilerResources& res)
        : graph_(graph),
          resources_(res) {}

const Shape&
BaseVisitor::GetOutputShape(HloInstruction* inst) const {
  return inst->shape();
}

Status BaseVisitor::Unimplemented(HloInstruction* inst) {
  return port::Status(port::error::UNIMPLEMENTED,
                      port::StrCat(inst->name(),
                                   " not implemented"));
}

Status BaseVisitor::HandleElementwiseUnary(
        HloInstruction* inst,
        HloOpcode opcode) {
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(prog,
                      CreateUnaryElementwiseOp(*graph_,
                                               resources_,
                                               inst,
                                               GetOutputShape(inst),
                                               tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status BaseVisitor::HandleElementwiseBinary(
        HloInstruction* inst,
        HloOpcode opcode) {
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(prog,
                      CreateBinaryElementwiseOp(*graph_,
                                                resources_,
                                                inst,
                                                GetOutputShape(inst),
                                                tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status BaseVisitor::HandleConvert(
        HloInstruction* inst) {
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(prog,
                      CreateCastOp(*graph_,
                                   resources_,
                                   inst,
                                   GetOutputShape(inst),
                                   tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status BaseVisitor::HandleCopy(HloInstruction* inst) {
  poplar::Tensor in;
  poplar::Tensor out;
  TF_ASSIGN_OR_RETURN(in, FindInstructionInput(tensor_map, inst, 0));

  sequence.add(poplar::program::Copy(in, out));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
  return Status::OK();
}

Status BaseVisitor::HandleClamp(
        HloInstruction* inst,
        HloInstruction* min,
        HloInstruction* arg,
        HloInstruction* max) {
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(prog,
                      CreateClampOp(*graph_,
                                    resources_,
                                    inst,
                                    GetOutputShape(inst),
                                    tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status BaseVisitor::HandleSelect(
        HloInstruction* inst,
        HloInstruction* pred,
        HloInstruction* on_true,
        HloInstruction* on_false) {
  poplar::program::Program prog;
  TF_ASSIGN_OR_RETURN(prog,
                      CreateSelectOp(*graph_,
                                     resources_,
                                     inst,
                                     GetOutputShape(inst),
                                     tensor_map));
  sequence.add(prog);
  return Status::OK();
}

Status BaseVisitor::HandleConcatenate(
        HloInstruction* inst,
        tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleDot(
        HloInstruction* inst,
        HloInstruction* lhs,
        HloInstruction* rhs) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleConvolution(
        HloInstruction* inst,
        HloInstruction* lhs,
        HloInstruction* rhs,
        const Window& window) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleCrossReplicaSum(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleRng(
        HloInstruction* inst,
        RandomDistribution distribution) {
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

Status BaseVisitor::HandleReverse(
        HloInstruction* inst,
        HloInstruction* operand) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleSort(
        HloInstruction* inst,
        HloInstruction* operand) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleConstant(
        HloInstruction* inst,
        const Literal& literal) {
  poplar::Tensor t;
  TF_ASSIGN_OR_RETURN(t, AddConstantTensor(*graph_,
                                           GetOutputShape(inst),
                                           inst->literal(),
                                           resources_));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, t));
  return Status::OK();
}

Status BaseVisitor::HandleGetTupleElement(
        HloInstruction* inst,
        HloInstruction* operand) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleReduce(
        HloInstruction* inst,
        HloInstruction* arg,
        HloInstruction* init_value,
        tensorflow::gtl::ArraySlice<int64> dimensions,
        HloComputation* function) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleBitcast(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleBroadcast(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleReshape(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleTranspose(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleFusion(HloInstruction* inst) {
  switch (static_cast<int>(inst->fusion_kind())) {
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
    case FUSED_WIDE_CONSTANT:
    {
      const HloInstruction* root = inst->fused_expression_root();
      poplar::Tensor out;
      TF_ASSIGN_OR_RETURN(out, AddConstantTensor(*graph_, inst->shape(),
                                                 root->operand(0)->literal(),
                                                 resources_));
      TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, out));
      return Status::OK();
    }
    default:
      return Unimplemented(inst);
  }
};


Status BaseVisitor::HandleCall(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleCustomCall(
        HloInstruction* inst,
        tensorflow::gtl::ArraySlice<HloInstruction*> operands,
        tensorflow::StringPiece custom_call_target) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleSlice(
        HloInstruction* inst,
        HloInstruction* operand) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleDynamicSlice(
        HloInstruction* inst,
        HloInstruction* operand,
        HloInstruction* start_indices) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleDynamicUpdateSlice(
        HloInstruction* inst,
        HloInstruction* operand,
        HloInstruction* update,
        HloInstruction* start_indices) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleTuple(
        HloInstruction* inst,
        tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleMap(
        HloInstruction* inst,
        tensorflow::gtl::ArraySlice<HloInstruction*> operands,
        HloComputation* function,
        tensorflow::gtl::ArraySlice<HloInstruction*> static_operands) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleReduceWindow(
        HloInstruction* inst,
        HloInstruction* operand,
        const Window& window,
        HloComputation* function) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleSelectAndScatter(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleWhile(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandlePad(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleReducePrecision(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleInfeed(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleOutfeed(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleSend(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleRecv(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleBatchNormTraining(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status BaseVisitor::HandleBatchNormGrad(HloInstruction* inst) {
  return Unimplemented(inst);
}

}  // namespace poplarplugin
}  // namespace xla
