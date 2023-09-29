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

#ifndef XLA_SERVICE_P2P_SCHEDULE_PREPARATION_H_
#define XLA_SERVICE_P2P_SCHEDULE_PREPARATION_H_

#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

// P2PSchedulePreparation is a pass to linearize point-to-point operation chain
// to prepare for any HLO scheduler. In particular, this pass currently does the
// following:
// (1) For an unpipelined P2P Send-Recv chain, add control dependence to
//     express this ordering:
//       recv => send => recv-done => send-done
//
// (2) For a pipelined P2P Send-Recv chain, add control dependence to the
//     while-body to express this ordering:
//       send => recv
//     No control dependence are added to a pipelined Send-Recv in the
// computation with the while-loop as the data dependence already expresses
// this ordering:
//       recv => recv-done => while-loop => send => send-done.
//
// (3) For a pipelined P2P Send-Recv chain, if the while-body has other P2P
// chains, we need to add control dependence to ensure that the pipelined
// Send-done is ordered before other P2P chains while the pipelined
// Recv-done is ordered after other P2P chains. In particular, we make the
// pipelined Send-done the control predecessor of other Recv and the pipelined
// Recv the control successor of other Send-done. Here is an example to
// illustrate the problem we address:
//
// Assume a while-body with the following HLO collective-permute operations:
//    collective-permute-start-1 = (u32[2], u32[2])
//      collective-permute-start(data), channel_id=1...
//    collective-permute-done-1 = u32[2], channel_id=1
//    use of collective-permute-done-1 result
//    collective-permute-start-2 = (u32[2], u32[2])
//      collective-permute-start(data), channel_id=2...
//    collective-permute-done-2 = u32[2], channel_id=2
//    use of collective-permute-done-2 result
//
// Now assume we transform the collective-permute operations into two P2P
// Send-Recv chains, the block of code will become something like this:
//    after-all-1 = token[] after-all()
//    recv-1 = (u32[2], token[]) recv(after-all-1), channel_id=1 ...
//    send-1 = (u32[2], token[]) send(data, after-all-1), channel_id=1 ...
//    recv-done-1 = (u32[2], token[]) recv-done(recv-1), channel_id=1 ...
//    send-done-1 = token[] send-done(send-1), channel_id=1 ...
//    use of recv-done-1 result
//    after-all-2 = token[] after-all()
//    recv-2 = (u32[2], token[]) recv(after-all-2), channel_id=2 ...
//    send-2 = (u32[2], token[]) send(data, after-all-2), channel_id=2 ...
//    recv-done-2 = (u32[2], token[]) recv-done(recv-2), channel_id=2 ...
//    send-done-2 = token[] send-done(send-2), channel_id=2 ...
//    use of recv-done-2 result
//
// If the while-loop is not pipelined, this pass adds control dependence to
// make sure the first Send-Recv chain finish before the second Send-Recv
// starts.
//
// If the while-loop is pipelined for the first Send-Recv chain, then the
// first Recv and the last Send of the chain are moved to the computation
// that calls the while-loop, and the block of code in the while-body will
// become something like this:

//    after-all-1 = token[] after-all()
//    send-1 = (u32[2], token[]) send(data, after-all-1), channel_id=1 ...
//    send-done-1 = token[] send-done(send-1), channel_id=1 ...
//    use of recv-done-1 result from the previous iteration or the computation
//      that calls the while-loop (for the first iteration)
//
//    after-all-2 = token[] after-all()
//    recv-2 = (u32[2], token[]) recv(after-all-2), channel_id=2 ...
//    send-2 = (u32[2], token[]) send(data, after-all-2), channel_id=2 ...
//    recv-done-2 = (u32[2], token[]) recv-done(recv-2), channel_id=2 ...
//    send-done-2 = token[] send-done(send-2), channel_id=2 ...
//    use of recv-done-2 result
//
//    recv-1 = (u32[2], token[]) recv(after-all-1), channel_id=1 ...
//    recv-done-1 = (u32[2], token[]) recv-done(recv-1), channel_id=1 ...
//
// In this case, we make send-done-1 the control predecessor of recv-2 and
// send-done-2 the control predecessor of recv-1 to ensure that the second
// Send-Recv chain is executed after the Send for the first chain finishes and
// before the Recv for the first chain starts.
//
// (4) Adds control prececessors/successors to ensure that a P2P Send-Recv
// sequence on a non-host device will be scheduled before other operations that
// use the Recv result and may also invoke P2P operations indirectly. Here is an
// example to illustrate the problem we address:
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
//
// (4) Similar to case (3), if the result of the while-loop with pipelined P2P
// chain is used by an instruction with nested P2P chains, we meed to schedule
// send-done of the pipelined while-loop before the instruction to avoid
// deadlock. We express this by making the send-done a control predecessor of
// the get-tuple-element instruction that retrieve the while-loop result.
//
class P2PSchedulePreparation : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "latency-hiding-scheduler-preparation";
  }

  using HloPassInterface::Run;
  // Runs P2PSchedulePreparation pass on computations in 'module'.
  // Returns whether the 'module' was changed.
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_P2P_SCHEDULE_PREPARATION_H_
