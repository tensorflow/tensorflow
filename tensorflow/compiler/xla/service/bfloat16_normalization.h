/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_BFLOAT16_NORMALIZATION_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_BFLOAT16_NORMALIZATION_H_

#include "tensorflow/compiler/xla/service/bfloat16_support.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// A pass which adds F32 <-> BF16 conversions for HLO instructions that do not
// support BF16 input/output or mixed precision, according to the passed-in
// backend-specific BF16 support rules.
class BFloat16Normalization : public HloPassInterface {
 public:
  explicit BFloat16Normalization(const BFloat16Support* bfloat16_support)
      : bfloat16_support_(bfloat16_support) {}

  ~BFloat16Normalization() override = default;
  absl::string_view name() const override { return "bf16-normalization"; }

  // Run BF16 normalization on the given computation. Returns whether the
  // computation was changed.
  StatusOr<bool> Run(HloModule* module) override;

 private:
  const BFloat16Support* bfloat16_support_;
};

// A pass that unconditionally removes the mixed F32/BF16 uses in HLO
// instructions (excluding convert) by adding F32 <-> BF16 conversions. Unlike
// BFloat16Normalization, this pass does not use a backend-specific
// BFloat16Support, and does not change HLOs that have BF16 data if they do not
// use mixed precision; it removes mixed precision even if the backend supports
// it. This pass is used to make the HLO module valid for other HLO passes which
// do not support mixed precision.
class BFloat16MixedPrecisionRemoval : public HloPassInterface {
 public:
  BFloat16MixedPrecisionRemoval() {}

  ~BFloat16MixedPrecisionRemoval() override = default;

  absl::string_view name() const override {
    return "bf16-mixed-precision-removal";
  }

  // Run mixed precision removal on the given computation. Returns whether the
  // computation was changed.
  StatusOr<bool> Run(HloModule* module) override {
    BFloat16Normalization normalization(&no_mixed_precision_support_);
    return normalization.Run(module);
  }

 private:
  class BFloat16SupportForMixedPrecisionRemoval : public BFloat16Support {
   public:
    BFloat16SupportForMixedPrecisionRemoval() {}

    ~BFloat16SupportForMixedPrecisionRemoval() override = default;

    bool SupportsBF16Operand(const HloInstruction& hlo,
                             int64 operand_index) const override {
      return true;
    }

    bool SupportsBF16Output(const HloInstruction& hlo) const override {
      return true;
    }

    bool SupportsMixedPrecisions(const HloInstruction& hlo) const override {
      return false;
    }
  } no_mixed_precision_support_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_BFLOAT16_NORMALIZATION_H_
