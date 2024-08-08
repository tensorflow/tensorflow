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
#ifndef XLA_SERVICE_GPU_TRANSFORMS_DOUBLE_BUFFER_LOOP_UNROLLING_H_
#define XLA_SERVICE_GPU_TRANSFORMS_DOUBLE_BUFFER_LOOP_UNROLLING_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// With `kDoubleBuffer` strategy:
//   This pass performs the unrolling-by-2 loop transformation
//   to effectively achieve double buffering between inputs and outputs
//   of previously rolled iterations.
//   This pass only runs on loops with known trip counts.
//   For even number of iterations, unrolling-by-2 will be done directly.
//   For odd number of iterations, the first iteration of the loop will be
//   peeled outside of the while loop to make the trip count an even number,
//   then proceed to unroll by 2.
//   It also updates the trip count property of the loop to the correct one
//   (n/2).
//
// With `kFullUnroll` strategy:
//   This pass will perform the full unroll of the loop with the same strategy
//   that is used with `kDoubleBuffer` but while loop trip count times.
//   It updates the trip count of the while loop to 1, and relies on other
//   passes (like `WhileLoopSimplifier`) to simplify/get rid of the while loop
//   eventually.
//
// Note that this pass will flatten the call graph if any loop has been
// unrolled.
// TODO(olechwierowicz): Rename the loop unroller to something more generic like
// 'DoubleBufferLoopUnrolling'.
class DoubleBufferLoopUnrolling : public HloModulePass {
 public:
  enum class UnrollStrategy { kDoubleBuffer, kFullUnroll };

  explicit DoubleBufferLoopUnrolling(
      UnrollStrategy unroll_strategy = UnrollStrategy::kDoubleBuffer)
      : unroll_strategy_(unroll_strategy) {};
  ~DoubleBufferLoopUnrolling() override = default;

  absl::string_view name() const override {
    return "loop-double-buffer-transformer";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  UnrollStrategy unroll_strategy_;
};

}  // end namespace gpu
}  // end namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_DOUBLE_BUFFER_LOOP_UNROLLING_H_
