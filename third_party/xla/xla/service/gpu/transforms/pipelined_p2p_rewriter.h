/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_PIPELINED_P2P_REWRITER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_PIPELINED_P2P_REWRITER_H_

#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// PipelinedP2PRewriter is a pass that rewrites pipelined Send/Recv related
// code for point-to-point communication to rotate SendDone and RecvDone at the
// end of a loop iteration to the beginning of the next iteration. This pass
// operates on scheduled module and updates the instruction sequence.
//
// In particular, a pipelined Send/Recv chain with one channel group with this
// code pattern:
//
// main:
//    recv
//    send
//    recv-done
//    send-done
//    while-init = (recv-done, send-done, ...)
//    while-op = while(whiel-init) ...
//
// while-body:
//    ...
//    recv
//    send
//    recv-done
//    send-done
//    ROOT tuple(recv-done, send-done, ...)
//
// Will be transformed to:
//
// main:
//    recv
//    send
//    while-init = (recv, send, ...)
//    while-op = while(whiel-init) ...
//    recv-done
//    send-done
//
// while-body:
//    recv-done
//    ...
//    send-done
//    recv
//    send
//    ROOT tuple(recv, send, ...)
//
// A pipelined Send/Recv chain with two channel groups with this code pattern:
//
// main:
//    recv.0
//    send.0
//    recv.1
//    send.1
//    recv-done.0
//    send-done.0
//    recv-done.1
//    send-done.1
//    while-init = (recv-done.0, send-done.0, recv-done.1, send-done.1, ...)
//    while-op = while(whiel-init) ...
//
// while-body:
//    ...
//    recv.0
//    send.0
//    recv.1
//    send.1
//    recv-done.0
//    send-done.0
//    recv-done.1
//    send-done.1
//    ROOT = tuple(recv-done.0, send-done.0, recv-done.1, send-done.1, ...)
//
// Will be transformed to:
//
// main:
//
//    recv.0
//    send.0
//    recv.1
//    send.1
//    while-init = (recv.0, send.0, recv.1, send.1, ...)
//    while-op = while(while-init) ...
//    recv-done.0
//    send-done.0
//    recv-done.1
//    send-done.1
//
// while-body:
//    recv-done.0
//    recv-done.1
//    ...
//    send-done.0
//    send-done.1
//    recv.0
//    send.1
//    recv.1
//    send.1
//    ROOT tuple(recv.0, send.0, recv.1, send.1, ...)
//
class PipelinedP2PRewriter : public HloModulePass {
 public:
  absl::string_view name() const override { return "pipelined-p2p-rewriter"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_PIPELINED_P2P_REWRITER_H_
