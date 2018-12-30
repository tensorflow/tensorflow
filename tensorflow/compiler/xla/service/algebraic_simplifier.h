/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_ALGEBRAIC_SIMPLIFIER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_ALGEBRAIC_SIMPLIFIER_H_

#include <utility>

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

class AlgebraicSimplifierOptions {
 public:
  // Given shapes 'from_shape' and 'to_shape', determines if it is valid to
  // bitcast from 'from_shape' to 'to_shape' after considering platform
  // dependent effects on layout like alignment restrictions. Precondition: the
  // two shapes have layouts, the same number of elements and
  // ShapeUtil::ReshapeIsBitcast returns true.
  using ValidBitcastCallback =
      std::function<bool(const Shape& from_shape, const Shape& to_shape)>;

  explicit AlgebraicSimplifierOptions(
      ValidBitcastCallback valid_bitcast_callback)
      : valid_bitcast_callback_(std::move(valid_bitcast_callback)) {}
  // If valid_bitcast_callback returns true, then the pass will replace reshapes
  // and transposes with bitcasts.
  const ValidBitcastCallback& valid_bitcast_callback() const {
    return valid_bitcast_callback_;
  }

  // If is_layout_sensitive is true, then the simplifier preserves layout during
  // transformation. Otherwise, layout is ignored.
  void set_is_layout_sensitive(bool is_layout_sensitive) {
    is_layout_sensitive_ = is_layout_sensitive;
  }
  bool is_layout_sensitive() const { return is_layout_sensitive_; }

  // Enable dot simplification on platforms where it is profitable.
  void set_enable_dot_strength_reduction(bool enable_dot_strength_reduction) {
    enable_dot_strength_reduction_ = enable_dot_strength_reduction;
  }
  bool enable_dot_strength_reduction() const {
    return enable_dot_strength_reduction_;
  }

  // Enable convolution simplification on platforms where it is profitable.
  void set_enable_conv_simplification(bool enable_conv_simplification) {
    enable_conv_simplification_ = enable_conv_simplification;
  }
  bool enable_conv_simplification() const {
    return enable_conv_simplification_;
  }

  // If enable_permutation_sort_replacement is true, a sort op that is known to
  // sort a permutation will be replaced with a scatter op.
  void set_enable_permutation_sort_replacement(
      bool enable_permutation_sort_replacement) {
    enable_permutation_sort_replacement_ = enable_permutation_sort_replacement;
  }
  bool enable_permutation_sort_replacement() const {
    return enable_permutation_sort_replacement_;
  }

  // If enable_window_reduce_replacement is true, the kReduceWindow instruction
  // can be optimized by replacement with simpler operations.
  void set_enable_window_reduce_to_reduce_replacement(
      bool enable_window_reduce_to_reduce_replacement) {
    enable_window_reduce_to_reduce_replacement_ =
        enable_window_reduce_to_reduce_replacement;
  }
  bool enable_window_reduce_to_reduce_replacement() const {
    return enable_window_reduce_to_reduce_replacement_;
  }


 private:
  ValidBitcastCallback valid_bitcast_callback_;
  bool is_layout_sensitive_{false};
  bool enable_dot_strength_reduction_{true};
  bool enable_conv_simplification_{true};
  bool enable_permutation_sort_replacement_{false};
  bool enable_window_reduce_to_reduce_replacement_{true};
};

// A pass which performs algebraic simplifications.
class AlgebraicSimplifier : public HloModulePass {
 public:
  // If is_layout_sensitive is true, then the simplifier preserves layout during
  // transformation. Otherwise, layout is ignored.
  explicit AlgebraicSimplifier(const AlgebraicSimplifierOptions& options)
      : options_(options) {}
  ~AlgebraicSimplifier() override = default;
  absl::string_view name() const override { return "algsimp"; }

  // Run algebraic simplification on the given computation. Returns whether the
  // computation was changed.
  StatusOr<bool> Run(HloModule* module) override;

 private:
  AlgebraicSimplifierOptions options_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_ALGEBRAIC_SIMPLIFIER_H_
