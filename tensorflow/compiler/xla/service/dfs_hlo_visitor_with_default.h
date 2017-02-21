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
class DfsHloVisitorWithDefault : public DfsHloVisitor {
 public:
  DfsHloVisitorWithDefault() {}
  ~DfsHloVisitorWithDefault() override {}

  // Default action performed on HloInstruction.
  virtual Status DefaultAction(HloInstruction* hlo_instruction) = 0;

  Status HandleElementwiseUnary(HloInstruction* hlo, HloOpcode opcode,
                                HloInstruction* operand) override {
    return DefaultAction(hlo);
  }
  Status HandleElementwiseBinary(HloInstruction* hlo, HloOpcode opcode,
                                 HloInstruction* lhs,
                                 HloInstruction* rhs) override {
    return DefaultAction(hlo);
  }
  Status HandleClamp(HloInstruction* clamp, HloInstruction* /*min*/,
                     HloInstruction* /*arg*/,
                     HloInstruction* /*max*/) override {
    return DefaultAction(clamp);
  }
  Status HandleConcatenate(
      HloInstruction* concatenate,
      tensorflow::gtl::ArraySlice<HloInstruction*> /*operands*/) override {
    return DefaultAction(concatenate);
  }
  Status HandleConvert(HloInstruction* convert,
                       HloInstruction* /*operand*/) override {
    return DefaultAction(convert);
  }
  Status HandleCopy(HloInstruction* copy,
                    HloInstruction* /*operand*/) override {
    return DefaultAction(copy);
  }
  Status HandleSelect(HloInstruction* select, HloInstruction* /*pred*/,
                      HloInstruction* /*on_true*/,
                      HloInstruction* /*on_false*/) override {
    return DefaultAction(select);
  }
  Status HandleDot(HloInstruction* dot, HloInstruction* /*lhs*/,
                   HloInstruction* /*rhs*/) override {
    return DefaultAction(dot);
  }
  Status HandleConvolution(HloInstruction* convolution, HloInstruction* /*lhs*/,
                           HloInstruction* /*rhs*/,
                           const Window& /*window*/) override {
    return DefaultAction(convolution);
  }
  Status HandleCrossReplicaSum(HloInstruction* crs) override {
    return DefaultAction(crs);
  }
  Status HandleCompare(HloInstruction* compare, HloOpcode /*opcode*/,
                       HloInstruction* /*lhs*/,
                       HloInstruction* /*rhs*/) override {
    return DefaultAction(compare);
  }
  Status HandleRng(HloInstruction* random,
                   RandomDistribution /*distribution*/) override {
    return DefaultAction(random);
  }
  Status HandleInfeed(HloInstruction* infeed) override {
    return DefaultAction(infeed);
  }
  Status HandleOutfeed(HloInstruction* outfeed) override {
    return DefaultAction(outfeed);
  }
  Status HandleReverse(HloInstruction* reverse,
                       HloInstruction* /*operand*/) override {
    return DefaultAction(reverse);
  }
  Status HandleSort(HloInstruction* sort,
                    HloInstruction* /*operand*/) override {
    return DefaultAction(sort);
  }
  Status HandleConstant(HloInstruction* constant,
                        const Literal& /*literal*/) override {
    return DefaultAction(constant);
  }
  Status HandleGetTupleElement(HloInstruction* get_tuple_element,
                               HloInstruction* /*operand*/) override {
    return DefaultAction(get_tuple_element);
  }
  Status HandleParameter(HloInstruction* parameter) override {
    return DefaultAction(parameter);
  }
  Status HandleFusion(HloInstruction* fusion) override {
    return DefaultAction(fusion);
  }
  Status HandleCall(HloInstruction* call,
                    tensorflow::gtl::ArraySlice<HloInstruction*> /*operands*/,
                    HloComputation* /*computation*/) override {
    return DefaultAction(call);
  }
  Status HandleCustomCall(
      HloInstruction* custom_call,
      tensorflow::gtl::ArraySlice<HloInstruction*> /*operands*/,
      tensorflow::StringPiece /*custom_call_target*/) override {
    return DefaultAction(custom_call);
  }
  Status HandleSlice(HloInstruction* slice,
                     HloInstruction* /*operand*/) override {
    return DefaultAction(slice);
  }
  Status HandleDynamicSlice(
      HloInstruction* slice,
      tensorflow::gtl::ArraySlice<HloInstruction*> /*operands*/) override {
    return DefaultAction(slice);
  }
  Status HandleDynamicUpdateSlice(HloInstruction* dynamic_update_slice,
                                  HloInstruction* /*operand*/,
                                  HloInstruction* /*update*/,
                                  HloInstruction* /*start_indices*/) override {
    return DefaultAction(dynamic_update_slice);
  }
  Status HandleTuple(
      HloInstruction* tuple,
      tensorflow::gtl::ArraySlice<HloInstruction*> /*operands*/) override {
    return DefaultAction(tuple);
  }
  Status HandleMap(
      HloInstruction* map,
      tensorflow::gtl::ArraySlice<HloInstruction*> /*operands*/,
      HloComputation* /*function*/,
      tensorflow::gtl::ArraySlice<HloInstruction*> /*static_operands*/)
      override {
    return DefaultAction(map);
  }
  Status HandleReduce(HloInstruction* reduce, HloInstruction* /*arg*/,
                      HloInstruction* /*init_value*/,
                      tensorflow::gtl::ArraySlice<int64> /*dimensions*/,
                      HloComputation* /*function*/) override {
    return DefaultAction(reduce);
  }
  Status HandleReduceWindow(HloInstruction* reduce_window,
                            HloInstruction* /*operand*/,
                            const Window& /*window*/,
                            HloComputation* /*function*/) override {
    return DefaultAction(reduce_window);
  }
  Status HandleSelectAndScatter(HloInstruction* select_and_scatter) override {
    return DefaultAction(select_and_scatter);
  }
  Status HandleBitcast(HloInstruction* bitcast) override {
    return DefaultAction(bitcast);
  }
  Status HandleBroadcast(HloInstruction* broadcast) override {
    return DefaultAction(broadcast);
  }
  Status HandlePad(HloInstruction* pad) override { return DefaultAction(pad); }
  Status HandleReshape(HloInstruction* reshape) override {
    return DefaultAction(reshape);
  }
  Status HandleTranspose(HloInstruction* transpose) override {
    return DefaultAction(transpose);
  }
  Status HandleWhile(HloInstruction* xla_while, HloInstruction* /*init*/,
                     HloComputation* /*condition*/,
                     HloComputation* /*body*/) override {
    return DefaultAction(xla_while);
  }
  Status HandleSend(HloInstruction* send) override {
    return DefaultAction(send);
  }
  Status HandleRecv(HloInstruction* recv) override {
    return DefaultAction(recv);
  }

  // Invoked to inform the visitor that the traversal has completed, and that
  // the root was "root".
  Status FinishVisit(HloInstruction* /*root*/) override { return Status::OK(); }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(DfsHloVisitorWithDefault);
};

// Helper class for Accept(VisitorFunction) which visits instructions in DFS
// order calling the given function at each instruction.
class FunctionVisitor : public DfsHloVisitorWithDefault {
 public:
  using VisitorFunction = std::function<Status(HloInstruction*)>;
  explicit FunctionVisitor(VisitorFunction visitor_func)
      : visitor_func_(std::move(visitor_func)) {}

  Status DefaultAction(HloInstruction* hlo_instruction) override {
    return visitor_func_(hlo_instruction);
  }

 private:
  VisitorFunction visitor_func_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_DFS_HLO_VISITOR_WITH_DEFAULT_H_
