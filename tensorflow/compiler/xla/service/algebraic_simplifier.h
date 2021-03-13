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
  AlgebraicSimplifierOptions() {}
  // Platform dependent callback to determine if a reshape `from_shape` to
  // `to_shape` is a bitcast.
  using ReshapeIsBitcastCallback =
      std::function<bool(const Shape& from_shape, const Shape& to_shape)>;
  explicit AlgebraicSimplifierOptions(
      ReshapeIsBitcastCallback reshape_is_bitcast_callback)
      : reshape_is_bitcast_callback_(std::move(reshape_is_bitcast_callback)) {}

  // Use the platform specific callback if set. It is not sensible to return
  // true here if the options are not layout sensitive.
  bool ReshapeIsBitcast(const Shape& from_shape, const Shape& to_shape) const {
    if (!is_layout_sensitive_) {
      return false;
    }
    if (!reshape_is_bitcast_callback_) {
      return ShapeUtil::ReshapeIsBitcast(from_shape, to_shape);
    }
    return reshape_is_bitcast_callback_(from_shape, to_shape);
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

  // Enable dot->multiple rewrite for dot as an outer-product
  void set_enable_dot_to_multiply_rewrite(bool enable_dot_to_multiply_rewrite) {
    enable_dot_to_multiply_rewrite_ = enable_dot_to_multiply_rewrite;
  }

  bool enable_dot_to_multiply_rewrite() const {
    return enable_dot_to_multiply_rewrite_;
  }

  // Enable convolution simplification on platforms where it is profitable.
  void set_enable_conv_simplification(bool enable_conv_simplification) {
    enable_conv_simplification_ = enable_conv_simplification;
  }
  bool enable_conv_simplification() const {
    return enable_conv_simplification_;
  }

  // Enable convolution operand swapping on platforms where it is supported.
  void set_enable_conv_operand_swap(bool enable_conv_operand_swap) {
    enable_conv_operand_swap_ = enable_conv_operand_swap;
  }
  bool enable_conv_operand_swap() const { return enable_conv_operand_swap_; }

  // Move constant scalar multiply to one operand or output of convolutions with
  // the smallest tensor size, to reduce the number of scalar multiply.
  void set_enable_scalar_multiply_reduction(
      bool enable_scalar_multiply_reduction) {
    enable_scalar_multiply_reduction_ = enable_scalar_multiply_reduction;
  }

  bool enable_scalar_multiply_reduction() const {
    return enable_scalar_multiply_reduction_;
  }

  // Also the algebraic simplifer to treat floating point values like real
  // numbers.
  void set_enable_floats_are_real(bool enable_floats_are_real) {
    enable_floats_are_real_ = enable_floats_are_real;
  }

  bool enable_floats_are_real() const { return enable_floats_are_real_; }

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

  // Sets the size of a gather operand that can be unrolled into many selects.
  void set_very_small_gather_size(int64 size) {
    very_small_gather_size_ = size;
  }

  int64 very_small_gather_size() const { return very_small_gather_size_; }

  void set_cudnn_batchnorm_forward_training_metadata(const string& c) {
    metadata_.cudnn_batchnorm_forward_training_metadata = c;
  }

  const string& get_cudnn_batchnorm_forward_training_metadata() const {
    return metadata_.cudnn_batchnorm_forward_training_metadata;
  }

  void set_enable_reduce_of_reshape(bool enable_reduce_of_reshape) {
    enable_reduce_of_reshape_ = enable_reduce_of_reshape;
  }

  bool enable_reduce_of_reshape() const { return enable_reduce_of_reshape_; }

  void set_enable_negative_padding_replacement(
      bool enable_negative_padding_replacement) {
    enable_negative_padding_replacement_ = enable_negative_padding_replacement;
  }

  bool enable_negative_padding_replacement() const {
    return enable_negative_padding_replacement_;
  }

  void set_replace_transpose_with_bitcast(bool replace_transpose_with_bitcast) {
    replace_transpose_with_bitcast_ = replace_transpose_with_bitcast;
  }

  bool replace_transpose_with_bitcast() const {
    return replace_transpose_with_bitcast_;
  }

 private:
  // Metadata struct can be used to store any metadata information encapsulated
  // with the AlgebraicSimplierOptions that can be later used in an
  // AlgebraicSimplifier pass. For example,
  // cudnn_batchnorm_forward_training_metadata can be used to store the name of
  // a custom call. If the custom call is
  // __cudnn$batchNormalizationForwardTraining, the output with index 2 is
  // guaranteed to be postive. This property has been used to recursively
  // determine if the operand of an instruction is always positive.
  struct Metadata {
    string cudnn_batchnorm_forward_training_metadata{""};
    Metadata() {}
  };
  ReshapeIsBitcastCallback reshape_is_bitcast_callback_;
  bool is_layout_sensitive_{false};
  bool enable_dot_strength_reduction_{true};
  bool enable_dot_to_multiply_rewrite_{true};
  bool enable_conv_simplification_{true};
  bool enable_conv_operand_swap_{true};
  bool enable_scalar_multiply_reduction_{false};
  bool enable_floats_are_real_{false};
  bool enable_window_reduce_to_reduce_replacement_{true};
  bool enable_reduce_of_reshape_{true};
  bool enable_negative_padding_replacement_{true};
  bool replace_transpose_with_bitcast_{true};
  int64 very_small_gather_size_{4};
  Metadata metadata_;
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

  // Create constant from literal with tiles and element size updated in the
  // constant's layout.
  std::unique_ptr<HloInstruction> CreateConstantWithLayoutUpdated(
      Literal literal) {
    auto constant = HloInstruction::CreateConstant(std::move(literal));
    UpdateLayout(constant->mutable_shape());
    return constant;
  }

 private:
  AlgebraicSimplifierOptions options_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_ALGEBRAIC_SIMPLIFIER_H_
