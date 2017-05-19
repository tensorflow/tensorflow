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

#include "tensorflow/compiler/plugin/example/visitor.h"

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

namespace se = ::perftools::gputools;

namespace xla {
namespace exampleplugin {

ExampleVisitor::ExampleVisitor() {}

Status 
ExampleVisitor::HandleElementwiseUnary(
        HloInstruction* inst,
        HloOpcode opcode,
        HloInstruction* operand) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleElementwiseBinary(
        HloInstruction* inst,
        HloOpcode opcode,
        HloInstruction* lhs,
        HloInstruction* rhs) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleClamp(HloInstruction* inst, HloInstruction* min,
                   HloInstruction* arg, HloInstruction* max) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleSelect(HloInstruction* inst, HloInstruction* pred,
                    HloInstruction* on_true,
                    HloInstruction* on_false) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleConcatenate(
        HloInstruction* inst,
        tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleDot(HloInstruction* inst,
                          HloInstruction* lhs,
                          HloInstruction* rhs) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleConvolution(
        HloInstruction* inst,
        HloInstruction* lhs, HloInstruction* rhs,
        const Window& window) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleCrossReplicaSum(HloInstruction* inst) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleInfeed(HloInstruction* inst) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleOutfeed(HloInstruction* inst) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleRng(HloInstruction* inst,
                          RandomDistribution distribution) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleReverse(HloInstruction* inst,
                              HloInstruction* operand) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleSort(HloInstruction* inst, HloInstruction* operand) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleConstant(HloInstruction* inst,
                      const Literal& literal) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleGetTupleElement(HloInstruction* inst,
                             HloInstruction* operand) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleReduce(
        HloInstruction* inst,
        HloInstruction* arg,
        HloInstruction* init_value,
        tensorflow::gtl::ArraySlice<int64> dimensions,
        HloComputation* function) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleBitcast(HloInstruction* inst) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleBroadcast(HloInstruction* inst) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleReshape(HloInstruction* inst) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleTranspose(HloInstruction* inst) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleParameter(HloInstruction* inst) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleFusion(HloInstruction* inst) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleCall(HloInstruction* inst) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleCustomCall(
        HloInstruction* inst,
        tensorflow::gtl::ArraySlice<HloInstruction*> operands,
        tensorflow::StringPiece custom_call_target) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleSlice(
        HloInstruction* inst,
        HloInstruction* operand) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleDynamicSlice(
        HloInstruction* inst,
        tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status ExampleVisitor::HandleDynamicUpdateSlice(
        HloInstruction* inst,
        HloInstruction* operand,
        HloInstruction* update,
        HloInstruction* start_indices) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleTuple(
        HloInstruction* inst,
        tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleMap(
        HloInstruction* inst,
        tensorflow::gtl::ArraySlice<HloInstruction*> operands,
        HloComputation* function,
        tensorflow::gtl::ArraySlice<HloInstruction*> static_operands) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleReduceWindow(
        HloInstruction* inst,
        HloInstruction* operand,
        const Window& window,
        HloComputation* function) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleSelectAndScatter(HloInstruction* inst) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleWhile(HloInstruction* inst) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandlePad(HloInstruction* inst) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleSend(HloInstruction* inst) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::HandleRecv(HloInstruction* inst) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}

Status
ExampleVisitor::FinishVisit(HloInstruction* inst) {
  VLOG(1) << inst->ToString();
  return Status::OK();
}


}  // namespace exampleplugin
}  // namespace xla
