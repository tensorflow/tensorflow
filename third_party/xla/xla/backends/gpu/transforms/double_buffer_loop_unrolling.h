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
#ifndef XLA_BACKENDS_GPU_TRANSFORMS_DOUBLE_BUFFER_LOOP_UNROLLING_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_DOUBLE_BUFFER_LOOP_UNROLLING_H_

#include <cstdint>
#include <variant>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/gpu/backend_configs.pb.h"

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
//
// If a DS/DUS in the loop body has a DynamicSliceConfig, loop unrolling must
// update that metadata together with the loop shape. A peeled iteration, or
// each copy created by full unrolling, corresponds to one concrete original
// iteration and therefore becomes loop-independent. The two halves left inside
// a double-buffered loop still depend on the loop iteration, but their offset
// stride is multiplied by the unroll factor.
class DoubleBufferLoopUnrolling : public HloModulePass {
 public:
  enum class UnrollStrategy {
    kDoubleBuffer,
    kFullUnroll,
    kAuto,
    kManual,
  };
  static constexpr absl::string_view kManualUnrollFull = "full";
  static constexpr absl::string_view kManualUnrollDoubleBuffer =
      "double-buffer";

  explicit DoubleBufferLoopUnrolling(
      UnrollStrategy unroll_strategy = UnrollStrategy::kDoubleBuffer)
      : unroll_strategy_(unroll_strategy) {};
  ~DoubleBufferLoopUnrolling() override = default;

  absl::string_view name() const override {
    return "loop-double-buffer-transformer";
  }

  // Describes which original loop iteration a duplicated instruction
  // represents. Some instructions are moved to a concrete iteration, while
  // instructions that stay in an unrolled loop still depend on that loop's new
  // induction value.
  struct StaticLoopIteration {
    int64_t iteration;
  };

  // Represents an affine mapping from a transformed loop iteration to the
  // original loop iteration executed by this instruction. For new loop
  // iteration `j`, the instruction corresponds to original iteration:
  //
  //   start_iteration + j * iteration_stride
  //
  // For double buffering, the original loop body is duplicated twice, so the
  // two mappings are `peeled_iterations + 0 + j * 2` and
  // `peeled_iterations + 1 + j * 2`.
  struct DynamicLoopIteration {
    int64_t start_iteration;
    int64_t iteration_stride;
  };

  using LoopIteration = std::variant<StaticLoopIteration, DynamicLoopIteration>;

  // Returns a config adjusted according to either static or dynamic loop
  // iteration metadata.
  static DynamicSliceConfig MakeConfigForLoopIteration(
      const DynamicSliceConfig& config, const LoopIteration& loop_iteration);

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  UnrollStrategy unroll_strategy_;
};

}  // end namespace gpu
}  // end namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_DOUBLE_BUFFER_LOOP_UNROLLING_H_
