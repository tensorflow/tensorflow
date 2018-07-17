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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITOR_BASE_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITOR_BASE_H_

#include "tensorflow/compiler/plugin/poplar/driver/ops.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor.h"

namespace xla {
namespace poplarplugin {

struct CompilerResources;

/*
 * The base visitor handles all operations that are element-wise.
 * This includes all explicitly element-wise ops, and also operations
 * Select, Convert, Clamp, Rng, Constant.  All of these have no element
 * to element dependencies.
 */
class BaseVisitor : public DfsHloVisitor {
 public:
  BaseVisitor(poplar::Graph& graph, CompilerResources&);

  virtual const Shape& GetOutputShape(HloInstruction*) const;

  Status HandleElementwiseUnary(HloInstruction* inst) override;

  Status HandleElementwiseBinary(HloInstruction* inst) override;

  Status HandleConvert(HloInstruction* inst) override;

  Status HandleClamp(HloInstruction* inst) override;

  Status HandleSelect(HloInstruction* inst) override;

  Status HandleTupleSelect(HloInstruction* inst) override;

  Status HandleConcatenate(HloInstruction* inst) override;

  Status HandleBitcastConvert(HloInstruction* inst) override;

  Status HandleCopy(HloInstruction* inst) override;

  Status HandleDot(HloInstruction* inst) override;

  Status HandleConvolution(HloInstruction* inst) override;

  Status HandleCrossReplicaSum(HloInstruction* crs) override;

  Status HandleRng(HloInstruction* inst) override;

  Status HandleReverse(HloInstruction* inst) override;

  Status HandleSort(HloInstruction* inst) override;

  Status HandleConstant(HloInstruction* inst) override;

  Status HandleGetTupleElement(HloInstruction* inst) override;

  Status HandleReduce(HloInstruction* inst) override;

  Status HandleBitcast(HloInstruction* inst) override;

  Status HandleBroadcast(HloInstruction* inst) override;

  Status HandleReshape(HloInstruction* inst) override;

  Status HandleTranspose(HloInstruction* inst) override;

  Status HandleFusion(HloInstruction* inst) override;

  Status HandleCall(HloInstruction* inst) override;

  Status HandleCustomCall(HloInstruction* inst) override;

  Status HandleSlice(HloInstruction* inst) override;

  Status HandleDynamicSlice(HloInstruction* inst) override;

  Status HandleDynamicUpdateSlice(HloInstruction* inst) override;

  Status HandleTuple(HloInstruction* inst) override;

  Status HandleMap(HloInstruction* inst) override;

  Status HandleReduceWindow(HloInstruction* inst) override;

  Status HandleSelectAndScatter(HloInstruction* inst) override;

  Status HandleWhile(HloInstruction* inst) override;

  Status HandleConditional(HloInstruction* inst) override;

  Status HandlePad(HloInstruction* inst) override;

  Status HandleReducePrecision(HloInstruction* inst) override;

  Status HandleInfeed(HloInstruction* inst) override;

  Status HandleOutfeed(HloInstruction* inst) override;

  Status HandleSend(HloInstruction* inst) override;

  Status HandleSendDone(HloInstruction* inst) override;

  Status HandleRecv(HloInstruction* inst) override;

  Status HandleRecvDone(HloInstruction* inst) override;

  Status HandleBatchNormInference(HloInstruction* inst) override;

  Status HandleBatchNormTraining(HloInstruction* inst) override;

  Status HandleBatchNormGrad(HloInstruction* inst) override;

  Status HandleFft(HloInstruction* inst) override;

  Status HandleHostCompute(HloInstruction* inst) override;

  Status HandleGather(HloInstruction* inst) override;

  Status HandleAfterAll(HloInstruction* inst) override;

  Status HandleReal(HloInstruction* inst) override;

  TensorMap tensor_map;

  poplar::program::Sequence sequence;

 protected:
  Status Unimplemented(HloInstruction* inst);

  poplar::Graph& graph_;

  CompilerResources& resources_;
};

}  // namespace poplarplugin
}  // namespace xla

#endif
