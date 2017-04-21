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
#include "tensorflow/compiler/poplar/driver/visitor_base.h"

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

PoplarBaseVisitor::PoplarBaseVisitor(poplar::Graph* graph,
                                     CompilerResources& res)
        : graph_(graph),
          resources_(res) {}

const Shape&
PoplarBaseVisitor::GetOutputShape(HloInstruction* inst) const {
  return inst->shape();
}

Status PoplarBaseVisitor::Unimplemented(HloInstruction* inst) {
  return port::Status(port::error::UNIMPLEMENTED,
                      port::StrCat(inst->name(),
                                   " not implemented"));
}

Status PoplarBaseVisitor::HandleElementwiseUnary(
        HloInstruction* inst,
        HloOpcode opcode,
        HloInstruction* operand) {
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

Status PoplarBaseVisitor::HandleElementwiseBinary(
        HloInstruction* inst,
        HloOpcode opcode,
        HloInstruction* lhs,
        HloInstruction* rhs) {
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

Status PoplarBaseVisitor::HandleConvert(HloInstruction* inst,
                                        HloInstruction* operand) {
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

Status PoplarBaseVisitor::HandleClamp(
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

Status PoplarBaseVisitor::HandleSelect(
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

Status PoplarBaseVisitor::HandleConcatenate(
        HloInstruction* inst,
        tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandleDot(
        HloInstruction* inst,
        HloInstruction* lhs,
        HloInstruction* rhs) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandleConvolution(
        HloInstruction* inst,
        HloInstruction* lhs,
        HloInstruction* rhs,
        const Window& window) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandleCrossReplicaSum(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandleRng(
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

Status PoplarBaseVisitor::HandleReverse(
        HloInstruction* inst,
        HloInstruction* operand) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandleSort(
        HloInstruction* inst,
        HloInstruction* operand) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandleConstant(
        HloInstruction* inst,
        const Literal& literal) {
  poplar::Tensor t;
  TF_ASSIGN_OR_RETURN(t, AddConstantTensor(*graph_,
                                           inst->name(),
                                           GetOutputShape(inst),
                                           inst->literal()));
  TF_RETURN_IF_ERROR(AddOutputTensor(tensor_map, inst, 0, t));
  return Status::OK();
}

Status PoplarBaseVisitor::HandleGetTupleElement(
        HloInstruction* inst,
        HloInstruction* operand) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandleReduce(
        HloInstruction* inst,
        HloInstruction* arg,
        HloInstruction* init_value,
        tensorflow::gtl::ArraySlice<int64> dimensions,
        HloComputation* function) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandleBitcast(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandleBroadcast(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandleReshape(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandleTranspose(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandleFusion(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandleCall(
        HloInstruction* inst) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandleCustomCall(
        HloInstruction* inst,
        tensorflow::gtl::ArraySlice<HloInstruction*> operands,
        tensorflow::StringPiece custom_call_target) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandleSlice(
        HloInstruction* inst,
        HloInstruction* operand) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandleDynamicSlice(
        HloInstruction* inst,
        tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandleDynamicUpdateSlice(
        HloInstruction* inst,
        HloInstruction* operand,
        HloInstruction* update,
        HloInstruction* start_indices) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandleTuple(
        HloInstruction* inst,
        tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandleMap(
        HloInstruction* inst,
        tensorflow::gtl::ArraySlice<HloInstruction*> operands,
        HloComputation* function,
        tensorflow::gtl::ArraySlice<HloInstruction*> static_operands) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandleReduceWindow(
        HloInstruction* inst,
        HloInstruction* operand,
        const Window& window,
        HloComputation* function) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandleSelectAndScatter(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandleWhile(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandlePad(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandleInfeed(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandleOutfeed(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandleSend(HloInstruction* inst) {
  return Unimplemented(inst);
}

Status PoplarBaseVisitor::HandleRecv(HloInstruction* inst) {
  return Unimplemented(inst);
}


}  // namespace poplarplugin
}  // namespace xla
