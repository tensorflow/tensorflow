/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_CONV_OPERAND_SWAPPER_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_CONV_OPERAND_SWAPPER_H_

#include <functional>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// A pass that swaps convolution operands to make a more efficient operation.
class ConvOperandSwapper : public HloModulePass {
 public:
  // Platform dependent callback to determine if a set of reverse dimensions is
  // lowerable.
  using ConvIsLowerableCallback = std::function<bool(HloInstruction*)>;

  explicit ConvOperandSwapper(
      ConvIsLowerableCallback conv_is_lowerable_callback = {})
      : conv_is_lowerable_callback_(std::move(conv_is_lowerable_callback)) {}
  ~ConvOperandSwapper() override = default;
  absl::string_view name() const override { return "conv-operand-swapper"; }

  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  ConvIsLowerableCallback conv_is_lowerable_callback_;
};

absl::StatusOr<bool> SwapConvolutionOperandsIfBeneficial(
    HloConvolutionInstruction* convolution,
    ConvOperandSwapper::ConvIsLowerableCallback conv_is_lowerable_callback);

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_CONV_OPERAND_SWAPPER_H_
