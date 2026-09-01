/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_CONSTANT_FOLDING_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_CONSTANT_FOLDING_H_

#include <atomic>
#include <cstdint>
#include <functional>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/shape.h"

namespace xla {

// A pass which performs constant folding in order to avoid unnecessary
// computation on constants.
class HloConstantFolding : public HloModulePass {
 public:
  enum class Level {
    // The default choice that only folds in the cases where it is expected to
    // always improve runtime performance as well as limiting compile time
    // overhead.
    kDefault,
    // Aggressively folds all operations where it is possible to do so,
    // including evaluating an unbounded number of while loop iterations.
    // This has been shown to give significant performance improvements for
    // some workloads, but it can have deteremental effects on others as well as
    // having a large increase in compile time.
    // Use with caution.
    kAggressive,
  };

  struct Options {
    Level level = Level::kDefault;
    // When false, only instructions whose operands and results are all
    // integer or pred typed, plus pure data movement ops of any type (bitcast,
    // slice, reshape, copy, concatenate, transpose, reverse, pad, select,
    // dynamic-slice), are folded. Integer arithmetic evaluates identically on
    // the host evaluator and on backends, so folding cannot change model
    // numerics; float arithmetic may round differently. Intended for runs
    // late in a pipeline, where the pre-layout numerics contract is already
    // fixed.
    bool fold_float_arithmetic = true;
    // Post-layout mode. Producers with tuple shapes are folded by rewriting
    // their get-tuple-element users to leaf constants, and only when every
    // user is an array shaped get-tuple-element; a tuple shaped constant is
    // never materialized (backends do not support them, and after layout
    // assignment no algebraic simplifier run is guaranteed to re-expand
    // them).
    bool is_layout_sensitive = false;
    // Optional filter: when set, an instruction is folded only if this
    // returns true for its shape (also applied inside folded fusions and
    // called computations). Lets layout sensitive pipelines skip values
    // whose layouts a constant cannot legally carry, e.g. backend managed
    // tilings such as TPU SparseCore layouts.
    std::function<bool(const Shape&)> can_fold_shape;
  };

  explicit HloConstantFolding(Level level = Level::kDefault) {
    options_.level = level;
  }
  explicit HloConstantFolding(const Options& options) : options_(options) {}
  absl::string_view name() const override { return "constant_folding"; }

 protected:
  // Run constant folding operations on the given module. Returns whether the
  // module was changed (constant expressions folded).
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // Number of slow constant-folds we've encountered.  Used for firing
  // SlowOperationAlarms.
  static std::atomic<int64_t> slow_op_counter_;

  Options options_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_HLO_CONSTANT_FOLDING_H_
