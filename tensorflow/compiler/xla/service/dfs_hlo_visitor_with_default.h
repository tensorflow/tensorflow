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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_DFS_HLO_VISITOR_WITH_DEFAULT_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_DFS_HLO_VISITOR_WITH_DEFAULT_H_

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

class HloComputation;
class HloInstruction;

// DfsHloVisitor with default action based on the HloInstruction being visited.
// Users should not use this class directly, but use the type aliases
// DfsHloVisitorWithDefault/ConstDfsHloVisitorWithDefault instead.
//
// Do *not* add an override to this class if the opcode is covered by
// HandleElementwiseUnary/Binary. These opcode handlers dispatch to
// HandleElementwiseUnary/Binary in DfsHloVisitorBase. Adding such a handler
// here will break passes which rely on the HandleElementwiseUnary/Binary
// handling these opcodes.
template <typename HloInstructionPtr>
class DfsHloVisitorWithDefaultBase
    : public DfsHloVisitorBase<HloInstructionPtr> {
 public:
  DfsHloVisitorWithDefaultBase() {}
  ~DfsHloVisitorWithDefaultBase() override {}

  // Default action performed on HloInstruction.
  virtual Status DefaultAction(HloInstructionPtr hlo_instruction) = 0;

  Status HandleElementwiseUnary(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }
  Status HandleElementwiseBinary(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }

  Status HandleBatchNormTraining(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }

  Status HandleBatchNormInference(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }

  Status HandleBatchNormGrad(HloInstructionPtr hlo) override {
    return DefaultAction(hlo);
  }

  Status HandleClamp(HloInstructionPtr clamp) override {
    return DefaultAction(clamp);
  }
  Status HandleConcatenate(HloInstructionPtr concatenate) override {
    return DefaultAction(concatenate);
  }
  Status HandleSelect(HloInstructionPtr select) override {
    return DefaultAction(select);
  }
  Status HandleTupleSelect(HloInstructionPtr tuple_select) override {
    return DefaultAction(tuple_select);
  }
  Status HandleDot(HloInstructionPtr dot) override {
    return DefaultAction(dot);
  }
  Status HandleConvolution(HloInstructionPtr convolution) override {
    return DefaultAction(convolution);
  }
  Status HandleFft(HloInstructionPtr fft) override {
    return DefaultAction(fft);
  }
  Status HandleCrossReplicaSum(HloInstructionPtr crs) override {
    return DefaultAction(crs);
  }
  Status HandleRng(HloInstructionPtr random) override {
    return DefaultAction(random);
  }
  Status HandleInfeed(HloInstructionPtr infeed) override {
    return DefaultAction(infeed);
  }
  Status HandleOutfeed(HloInstructionPtr outfeed) override {
    return DefaultAction(outfeed);
  }
  Status HandleHostCompute(HloInstructionPtr host_compute) override {
    return DefaultAction(host_compute);
  }
  Status HandleReverse(HloInstructionPtr reverse) override {
    return DefaultAction(reverse);
  }
  Status HandleSort(HloInstructionPtr sort) override {
    return DefaultAction(sort);
  }
  Status HandleConstant(HloInstructionPtr constant) override {
    return DefaultAction(constant);
  }
  Status HandleIota(HloInstructionPtr iota) override {
    return DefaultAction(iota);
  }
  Status HandleGetTupleElement(HloInstructionPtr get_tuple_element) override {
    return DefaultAction(get_tuple_element);
  }
  Status HandleParameter(HloInstructionPtr parameter) override {
    return DefaultAction(parameter);
  }
  Status HandleFusion(HloInstructionPtr fusion) override {
    return DefaultAction(fusion);
  }
  Status HandleCall(HloInstructionPtr call) override {
    return DefaultAction(call);
  }
  Status HandleCustomCall(HloInstructionPtr custom_call) override {
    return DefaultAction(custom_call);
  }
  Status HandleSlice(HloInstructionPtr slice) override {
    return DefaultAction(slice);
  }
  Status HandleDynamicSlice(HloInstructionPtr dynamic_slice) override {
    return DefaultAction(dynamic_slice);
  }
  Status HandleDynamicUpdateSlice(
      HloInstructionPtr dynamic_update_slice) override {
    return DefaultAction(dynamic_update_slice);
  }
  Status HandleTuple(HloInstructionPtr tuple) override {
    return DefaultAction(tuple);
  }
  Status HandleMap(HloInstructionPtr map) override {
    return DefaultAction(map);
  }
  Status HandleReduce(HloInstructionPtr reduce) override {
    return DefaultAction(reduce);
  }
  Status HandleReduceWindow(HloInstructionPtr reduce_window) override {
    return DefaultAction(reduce_window);
  }
  Status HandleSelectAndScatter(HloInstructionPtr select_and_scatter) override {
    return DefaultAction(select_and_scatter);
  }
  Status HandleBitcast(HloInstructionPtr bitcast) override {
    return DefaultAction(bitcast);
  }
  Status HandleBroadcast(HloInstructionPtr broadcast) override {
    return DefaultAction(broadcast);
  }
  Status HandlePad(HloInstructionPtr pad) override {
    return DefaultAction(pad);
  }
  Status HandleReshape(HloInstructionPtr reshape) override {
    return DefaultAction(reshape);
  }
  Status HandleTranspose(HloInstructionPtr transpose) override {
    return DefaultAction(transpose);
  }
  Status HandleWhile(HloInstructionPtr xla_while) override {
    return DefaultAction(xla_while);
  }
  Status HandleConditional(HloInstructionPtr conditional) override {
    return DefaultAction(conditional);
  }
  Status HandleRecv(HloInstructionPtr recv) override {
    return DefaultAction(recv);
  }
  Status HandleRecvDone(HloInstructionPtr recv_done) override {
    return DefaultAction(recv_done);
  }
  Status HandleSend(HloInstructionPtr send) override {
    return DefaultAction(send);
  }
  Status HandleSendDone(HloInstructionPtr send_done) override {
    return DefaultAction(send_done);
  }
  Status HandleGather(HloInstructionPtr gather) override {
    return DefaultAction(gather);
  }
  Status HandleAfterAll(HloInstructionPtr token) override {
    return DefaultAction(token);
  }

  // Invoked to inform the visitor that the traversal has completed, and that
  // the root was "root".
  Status FinishVisit(HloInstructionPtr /*root*/) override {
    return Status::OK();
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(DfsHloVisitorWithDefaultBase);
};

// Users should use these type aliases which are only two valid instantiations.
using DfsHloVisitorWithDefault = DfsHloVisitorWithDefaultBase<HloInstruction*>;
using ConstDfsHloVisitorWithDefault =
    DfsHloVisitorWithDefaultBase<const HloInstruction*>;

// (Const)FunctionVisitor lets you transform an
// std::function<Status((const) HloInstruction*)> into a (Const)DfsHloVisitor.
//
// This is useful if you have code that needs to handle visitors in the form of
// both std::function and DfsHloVisitor.  You can wrap the function in a
// FunctionVisitor and then treat it like any other DfsHloVisitor.
template <typename HloInstructionPtr>
class FunctionVisitorBase
    : public DfsHloVisitorWithDefaultBase<HloInstructionPtr> {
 public:
  explicit FunctionVisitorBase(
      std::function<Status(HloInstructionPtr)> visitor_func)
      : visitor_func_(std::move(visitor_func)) {}

  Status DefaultAction(HloInstructionPtr hlo_instruction) override {
    return visitor_func_(hlo_instruction);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(FunctionVisitorBase);

  std::function<Status(HloInstructionPtr)> visitor_func_;
};

using FunctionVisitor = FunctionVisitorBase<HloInstruction*>;
using ConstFunctionVisitor = FunctionVisitorBase<const HloInstruction*>;

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_DFS_HLO_VISITOR_WITH_DEFAULT_H_
