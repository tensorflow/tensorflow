/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_COLLECTIVE_PERMUTE_DECOMPOSER_H_
#define XLA_SERVICE_COLLECTIVE_PERMUTE_DECOMPOSER_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// CollectivePermuteDecomposer is a pass that (1) converts CollectivePermute
// operations without any cycle in their (source, target) relationship to
// Send/Recv, and (2) annotates the Send/Recv for pipelining with a frontend
// attribute. We currently restrict the decomposition to CollectivePermute with
// one input and without any context data.
//
// before transformation:
//     cp = (<rt>, <rt>) collective-permute(data),
//       source_target_pairs={...}
//
// after transformation:
//    after-all = token[] after-all()
//    recv = (<rt>, token[]) recv(after-all), channel_id=0,
//     frontend_attributes={_xla_send_recv_source_target_pairs="{...}"}
//    send = (<rt>, token[]) send(data, after-all), channel_id=0,
//      control-predecessors={recv}, frontend_attributes={
//      _xla_send_recv_source_target_pairs="{...}"}
//    recv-done = (<rt>, token[]) recv-done(recv), channel_id=0
//    send-done = token[] send-done(send), channel_id=0,
//      control-predecessors={recv-done}
//    cp = <rt> get-tuple-element(recv-done), index=0
//
// For pipelining, we first make pipelining decision on CollectivePermute
// operations, and then record the decision on the decomposed Send/Recv via
// frontend attributes. We currently only pipeline CollectivePermute operations
// that send loop input data. As a simple heuristics, we pick the first
// encountered pipelineable CollectivePermute for pipelining. Then, if there is
// another pipelineable CollectivePermute that forms a forward or backward
// cycle with the first CollectivePermute, we mark both CollectivePermute
// for pipelining. Otherwise, we only mark one CollectivePermute for pipelining.
//
class CollectivePermuteDecomposer : public HloModulePass {
 public:
  explicit CollectivePermuteDecomposer(int64_t threshold_in_bytes)
      : threshold_in_bytes_(threshold_in_bytes) {}
  absl::string_view name() const override {
    return "collective-permute-decomposer";
  }

  using HloPassInterface::Run;
  // Runs CollectivePermuteDecomposer pass on computations in 'module'.
  // Returns whether the 'module' was changed.
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Transform only if the size of the collective permute is >= threshold.
  int64_t threshold_in_bytes_;
};

}  // namespace xla

#endif  // XLA_SERVICE_COLLECTIVE_PERMUTE_DECOMPOSER_H_
