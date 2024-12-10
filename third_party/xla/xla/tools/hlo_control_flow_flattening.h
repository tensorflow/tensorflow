/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_TOOLS_HLO_CONTROL_FLOW_FLATTENING_H_
#define XLA_TOOLS_HLO_CONTROL_FLOW_FLATTENING_H_

#include <stdbool.h>

#include <limits>
#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/call_graph.h"

namespace xla {

// An HLO pass that replaces while loop conditionals to execute a known constant
// number of iterations and remove operations that are difficult to run in
// standalone tests, such as infeed/outfeed and collective operations.
class HloControlFlowFlattening : public HloModulePass {
 public:
  // While execution count specifies how many times the while loops in the
  // transformed graph will execute.
  // If remove_comm = true, remove all communication operations.
  // If remove_host_transfer = true, remove the host-transfer send and recv
  // operations.
  struct Options {
    int while_execution_count = 1;
    int max_outer_loop_count = std::numeric_limits<int>::max();
    int max_loop_count = std::numeric_limits<int>::max();
    bool remove_infeed_outfeed = true;
    bool flatten_while_loop = true;
    bool remove_comm = true;
    bool remove_host_transfer = false;
    // Removes partition-id, replica-id, and slice-id.
    bool remove_id = false;
    // Whether to flatten conditional ops by setting a default index or PRED
    // value. For indexed conditional ops, the default value is the N-1'th
    // conditional branch computation.
    bool flatten_conditional = false;
    // If flatten_conditional is true, this will behe default predicate value to
    // use for predicated conditional ops.
    bool conditional_value = false;
  };
  explicit HloControlFlowFlattening(const Options& options)
      : while_execution_count_(options.while_execution_count),
        max_outer_loop_count_(options.max_outer_loop_count),
        max_loop_count_(options.max_loop_count),
        remove_infeed_outfeed_(options.remove_infeed_outfeed),
        flatten_while_loop_(options.flatten_while_loop),
        remove_host_transfer_(options.remove_host_transfer),
        flatten_conditional_(options.flatten_conditional),
        conditional_value_(options.conditional_value),
        remove_comm_(options.remove_comm),
        remove_id_(options.remove_id) {}
  ~HloControlFlowFlattening() override = default;
  absl::string_view name() const override { return "control-flow-flattening"; }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Replaces an infeed with a custom call.
  absl::Status RemoveInfeed(HloInstruction* infeed_hlo) const;
  // Removes outfeeds and replaces the outfeed HLO with a side-effecting custom
  // call that ensures that XLA doesn't dead-code-eliminate the outfeeded values
  // but lowers to a no-op.
  absl::Status RemoveOutfeed(HloInstruction* outfeed_hlo) const;
  // Flattens the while loop. Precondition: while_hlo is a while instruction.
  absl::Status FlattenWhileLoop(HloInstruction* while_hlo,
                                const CallGraph& call_graph) const;
  // Replaces an id with a zero constant.
  absl::Status RemoveId(HloInstruction* hlo) const;
  // Sets the default branch for conditional ops to take. For indexed
  // conditionals, the index is set to the N-1'th conditional branch
  // computation. For predicated conditionals, the PRED is set to
  // conditional_value_.
  absl::Status SetConditionalValue(HloInstruction* conditional) const;

  int while_execution_count_;
  int max_outer_loop_count_;
  int max_loop_count_;
  bool remove_infeed_outfeed_;
  bool flatten_while_loop_;
  bool remove_host_transfer_;
  bool flatten_conditional_;
  bool conditional_value_;

 protected:
  // Replaces a collective op with a custom call and returns the custom call.
  virtual absl::StatusOr<HloInstruction*> RemoveCollective(
      HloInstruction* hlo) const;
  // Replaces send and send-done with a custom call. Returns the new custom
  // calls in a pair.
  virtual absl::StatusOr<std::pair<HloInstruction*, HloInstruction*>>
  RemoveSendAndSendDone(
      HloInstruction* send_done,
      absl::flat_hash_set<HloInstruction*>* additional_removed) const;
  // Replaces recv and recv-done with a custom call. Returns the new custom
  // calls in a pair
  virtual absl::StatusOr<std::pair<HloInstruction*, HloInstruction*>>
  RemoveRecvAndRecvDone(
      HloInstruction* recv_done,
      absl::flat_hash_set<HloInstruction*>* additional_removed) const;
  bool remove_comm_;
  bool remove_id_;
};

// Retrieves the original loop bound. If fail, return a default value. If bounds
// exceed a given max, returns the max. This function is more opportunistic than
// ComputeWhileLoopTripCount in the while loop analysis as it may return a
// constant found in a compare expression when it is not an actual bound.
int GetLoopBound(const HloInstruction& while_hlo, const int default_loop_count,
                 const int max_loop_count);

// Retrieves the loop bound determined by the original loop bound, the max
// outer loops count and max loop count.
int GetLoopBoundWithOuterLoopMax(const HloInstruction& while_hlo,
                                 const CallGraph& call_graph,
                                 const int default_loop_count,
                                 const int max_outer_loop_count,
                                 const int max_loop_count);
}  // namespace xla

#endif  // XLA_TOOLS_HLO_CONTROL_FLOW_FLATTENING_H_
