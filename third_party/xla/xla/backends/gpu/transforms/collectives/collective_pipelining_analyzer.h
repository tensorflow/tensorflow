/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_PIPELINING_ANALYZER_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_PIPELINING_ANALYZER_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla::gpu {

// Analysis pass which traverses the HLO and adds metadata to collectives'
// frontend attributes if it finds that a collective is trivially pipeline-able.
// That is:
// * No heavy dependencies will be pipelined along with it.
// * Loop tuple size will not increase too much (e.g. for variadic collectives)
// if the collective gets pipelined.
//
// We define a collective to be trivially pipeline-eable if it is
// followed/preceded by dynamic-update-slice/dynamic-slice which operates only
// on induction variable, and no heavy (anything not no-op and simple converts,
// reshapes, transposes, etc.) ops follow/preceed it.
// The pass supports detecting such cases for AllReduce, AllGather, and
// ReduceScatter.
//
// Example in pseudocode:
//
// All-Reduce/Reduce-Scatter case:
// while (i < LAYERS) {
//   ...
//   ar = all-reduce(...)
//   inexpensive-op = trivial-op(ar)
//   dus = dynamic-update-slice(inexpensive-op, i)
//   inexpensive-op.2 = trivial-op(dus)
//   ROOT result = tuple(..., inexpensive-op.2)
// }
//
// All-Gather case:
// while (i < LAYERS) {
//   ds = dynamic-slice(i)
//   inexpensive-op = trivial-op(ds)
//   ag = all-gather(inexpensive-op)
//   ...
// }
//
class CollectivePipeliningAnalyzer : public HloModulePass {
 public:
  explicit CollectivePipeliningAnalyzer(int64_t pointer_size)
      : pointer_size_(pointer_size) {};

  absl::string_view name() const override {
    return "collective-pipelining-analyzer";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  int64_t pointer_size_;
};

// Determines whether the instruction is trivially pipeline-able.
//
// We define a collective to be trivially pipeline-eable if it is
// followed/preceded by dynamic-update-slice/dynamic-slice which operates only
// on induction variable, and no heavy (anything not no-op and simple converts,
// reshapes, transposes, etc.) ops follow/preceed it.
bool IsTriviallyPipelineable(const HloInstruction& instr);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_COLLECTIVES_COLLECTIVE_PIPELINING_ANALYZER_H_
