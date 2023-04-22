/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_TOOLS_HLO_CONTROL_FLOW_FLATTENING_H_
#define TENSORFLOW_COMPILER_XLA_TOOLS_HLO_CONTROL_FLOW_FLATTENING_H_

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// An HLO pass that replaces while loop conditionals to execute a known constant
// number of iterations and remove operations that are difficult to run in
// standalone tests, such as infeed/outfeed and collective operations.
class HloControlFlowFlattening : public HloModulePass {
 public:
  // While execution count specifies how many times the while loops in the
  // transformed graph will execute.
  explicit HloControlFlowFlattening(int while_execution_count,
                                    bool remove_infeed_outfeed = true,
                                    bool flatten_while_loop = true,
                                    bool remove_all_reduce = true)
      : while_execution_count_(while_execution_count),
        remove_infeed_outfeed_(remove_infeed_outfeed),
        flatten_while_loop_(flatten_while_loop),
        remove_all_reduce_(remove_all_reduce) {}
  ~HloControlFlowFlattening() override = default;
  absl::string_view name() const override { return "control-flow-flattening"; }
  StatusOr<bool> Run(HloModule* module) override;

 private:
  // Removes infeeds and replaces the infeeded values with constants.
  Status RemoveInfeed(HloInstruction* infeed_hlo) const;
  // Removes outfeeds and replaces the outfeed HLO with a side-effecting custom
  // call that ensures that XLA doesn't dead-code-eliminate the outfeeded values
  // but lowers to a no-op.
  Status RemoveOutfeed(HloInstruction* outfeed_hlo) const;
  // Flattens the while loop. Precondition: while_hlo is a while instruction.
  Status FlattenWhileLoop(HloInstruction* while_hlo) const;
  // Removes all reduce instructions.
  Status RemoveAllReduce(HloInstruction* hlo) const;

  int while_execution_count_;
  bool remove_infeed_outfeed_;
  bool flatten_while_loop_;
  bool remove_all_reduce_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_TOOLS_HLO_CONTROL_FLOW_FLATTENING_H_
