/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LATENCY_HIDING_SCHEDULER_PREPARATION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LATENCY_HIDING_SCHEDULER_PREPARATION_H_

#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// LatencyHidingSchedulerPreparation is a pass to linearize certain operations
// to prepare for the latency hiding scheduler (LHS). In particular, this pass
// currently does the following:
//
// Adds control prececessors/successors to ensure that a P2P Send-Recv sequence
// on a non-host device will be scheduled before other operations that use the
// Recv result and may also invoke P2P operations indirectly. Here is an example
// to illustrate the problem we address:
//
// Assume a computation with the following HLO instructions, where while-body
// invokes collective-permute operations:
//    collective-permute-start = (u32[2], u32[2])
//      collective-permute-start(data) ...
//    collective-permute-done = u32[2]
//      collective-permute-done(collective-permute-start)
//    while-init = (u32[], u32[2]) tuple(c0, collective-permute-done)
//    while-result = (u32[], u32[2]) while(while-init),
//      body=while-body, condition=while-cond
//
// Without collective-permute-decomposer transformation, LHS will Schedule
// while-result after collective-permute-start without any problem.
//
// Now assume we transform the collective-permute operations in the computation
// as well as inside the while-body into a sequence of P2P Send-Recv sequence,
// the computation will become something like this:
//    after-all = token[] after-all()
//    recv = (u32[2], token[]) recv(after-all) ...
//    send = (u32[2], token[]) send(data, after-all),
//      control-predecessors={recv} ...
//    recv-done = (u32[2], token[]) recv-done(recv),
//      control-predecessors={send} ...
//    send-done = token[] send-done(send),
//      control-predecessors={recv-done} ...
//    recv-data = u32[2] get-tuple-element(recv-done), index=0
//    while-init = (u32[], u32[2]) tuple(c0, recv-data)
//    while-result = (u32[], u32[2]) while(while_init),
//        body=while_body, condition=while_cond
//
// When scheduling this computation in a bottom up fashion, the LHS will reach a
// point where both while-result and send-done are in the ready queue. If LHS
// picks send-done over while-result, the scheduler is stuck because
// while-result can't be scheduled when the Send-Recv chain is holding the
// resources for P2P operations and recv-done cannot be scheduled as well
// because while-result depends on while-init which depends on recv-done. To
// avoid this deadlock, we make send-done a control predecessor of recv-data
// in this pass.
//
// Note that instead of making send-done a control predecessor of recv-data, we
// may make send-done a control predecessor of the instruction that contains
// the nested P2P operations, which is while-result in this example. This allows
// recv-data and while-init to be scheduled before send-done. However, doing so
// would complicate the implementation. We leave this to future improvement if
// we will find out it can actually help performance in real practice.
class LatencyHidingSchedulerPreparation : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "latency-hiding-scheduler-preparation";
  }

  using HloPassInterface::Run;
  // Runs LatencyHidingSchedulerPreparation pass on computations in 'module'.
  // Returns whether the 'module' was changed.
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LATENCY_HIDING_SCHEDULER_PREPARATION_H_
