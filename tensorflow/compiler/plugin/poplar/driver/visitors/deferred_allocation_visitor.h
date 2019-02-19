/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_DEFERRED_ALLOCATION_VISITOR_H_
#define TENSORFLOW_COMPILER_PLUGIN_POPLAR_DRIVER_VISITORS_DEFERRED_ALLOCATION_VISITOR_H_

#include "tensorflow/compiler/plugin/poplar/driver/passes/allocation_finder.h"
#include "tensorflow/compiler/plugin/poplar/driver/visitors/visitor_full.h"

namespace xla {
namespace poplarplugin {

struct CompilerResources;
/*
 * This visitor uses the deferred allocation info to allocate tuple allocation
 * targets when needed.
 * This is required for forward allocations where the target and the source both
 * come from the same input instruction.
 */
class DeferredAllocationVisitor : public FullVisitor {
 public:
  DeferredAllocationVisitor(CompilerResources& resources)
      : FullVisitor(resources) {}

  // GTEs are specialised:
  // * if the GTE input is deferred and:
  //   - this is the deferred allocation place then this calls the AllocateInput
  //   - otherwise it skips all the deferred allocations in the output.
  // * Otherwise it behaves like a GTE.
  Status HandleGetTupleElement(HloInstruction* inst) override;

  Status HandleInfeed(HloInstruction* inst) override;

 protected:
  // Allocates the input and calls the post processing function - this function
  // should be called by HandleParameter and HandleInfeed. If it's allocating a
  // deferred input then it also makes sure to set the outputs of all
  // instructions between the input tuple and inst to this allocation.
  Status AllocateInput(const HloInstruction* inst, int64 flat_tuple_index,
                       const Shape& shape);

  // Called by AllocateInput when allocating an input for an infeed.
  StatusOr<poplar::Tensor> PostProcessInfeedAllocation(
      const HloInstruction* inst, int64 flat_tuple_index,
      poplar::Tensor tensor);

  // Called by AllocateInput when allocating an input for a paramter.
  virtual StatusOr<poplar::Tensor> PostProcessParameterAllocation(
      const HloInstruction* inst, int64 flat_tuple_index,
      poplar::Tensor tensor) = 0;

  bool DeferAllocation(const HloInstruction* inst, int64 flat_tuple_index);

 private:
  bool IsDeferredAllocation(const HloInstruction* inst, int64 flat_tuple_index);
};

}  // namespace poplarplugin
}  // namespace xla

#endif
