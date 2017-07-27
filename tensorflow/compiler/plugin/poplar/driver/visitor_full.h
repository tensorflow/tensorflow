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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITOR_FULL_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITOR_FULL_H_

#include "tensorflow/compiler/plugin/poplar/driver/visitor_base.h"
#include "tensorflow/compiler/plugin/poplar/driver/ops.h"

namespace xla {
namespace poplarplugin {

/*
 * The full visitor is an extension of the base visitor
 * that adds other operations which do element to element
 * mixing, for instance convolution.  It also adds ops
 * that change the shape of the tensor, for instance Reverse
 * or Concatinate.
 */
class FullVisitor : public BaseVisitor {
public:
  FullVisitor(poplar::Graph* graph, CompilerResources&);

  Status HandleConcatenate(
          HloInstruction* inst,
          tensorflow::gtl::ArraySlice<HloInstruction*> operands) override;

  Status HandleDot(HloInstruction* inst,
                   HloInstruction* lhs,
                   HloInstruction* rhs) override;

  Status HandleConvolution(HloInstruction* inst,
                           HloInstruction* lhs,
                           HloInstruction* rhs,
                           const Window& window) override;

  Status HandleReverse(HloInstruction* inst,
                       HloInstruction* operand) override;

  Status HandleGetTupleElement(HloInstruction* inst,
                               HloInstruction* operand) override;

  Status HandleReduce(HloInstruction* inst,
                      HloInstruction* arg,
                      HloInstruction* init_value,
                      tensorflow::gtl::ArraySlice<int64> dimensions,
                      HloComputation* function) override;

  Status HandleBitcast(HloInstruction* inst) override;

  Status HandleBroadcast(HloInstruction* inst) override;

  Status HandleReshape(HloInstruction* inst) override;

  Status HandleTranspose(HloInstruction* inst) override;

  Status HandleFusion(HloInstruction* fusion) override;

  Status HandleCall(HloInstruction* inst) override;

  Status HandleSlice(HloInstruction* inst,
                     HloInstruction* operand) override;

  Status HandleDynamicSlice(HloInstruction* dynamic_slice,
                            HloInstruction* operand,
                            HloInstruction* start_indices) override;

  Status HandleDynamicUpdateSlice(HloInstruction* inst,
                                  HloInstruction* operand,
                                  HloInstruction* update,
                                  HloInstruction* start_indices) override;

  Status HandleTuple(HloInstruction* inst,
                     tensorflow::gtl::ArraySlice<HloInstruction*> operands) override;

  Status HandleMap(HloInstruction* inst,
                   tensorflow::gtl::ArraySlice<HloInstruction*> operands,
                   HloComputation* function,
                   tensorflow::gtl::ArraySlice<HloInstruction*> static_operands) override;

  Status HandleReduceWindow(HloInstruction* inst,
                            HloInstruction* operand,
                            const Window& window,
                            HloComputation* function) override;

  Status HandleSelectAndScatter(HloInstruction* inst) override;

  Status HandleWhile(HloInstruction* inst) override;

  Status HandlePad(HloInstruction* inst) override;

};

}  // namespace poplarplugin
}  // namespace xla

#endif
