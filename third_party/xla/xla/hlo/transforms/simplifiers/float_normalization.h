/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_FLOAT_NORMALIZATION_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_FLOAT_NORMALIZATION_H_

#include <cstdint>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/float_support.h"
#include "xla/xla_data.pb.h"

namespace xla {

// A pass which adds type conversions (e.g. F32 <-> BF16) for HLO instructions
// that do not support low-precision input/output or mixed precision, according
// to the passed-in backend-specific FloatSupport instance.
class FloatNormalization : public HloModulePass {
 public:
  explicit FloatNormalization(const FloatSupport* float_support)
      : float_support_(float_support),
        name_("float-normalization-" +
              primitive_util::LowercasePrimitiveTypeName(
                  float_support_->LowPrecisionType())) {}

  ~FloatNormalization() override = default;
  absl::string_view name() const override { return name_; }

  // Run float normalization on the given computation. Returns whether the
  // computation was changed.
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const FloatSupport* float_support_;
  std::string name_;
};

// A pass that unconditionally removes the mixed F32/BF16 uses in HLO
// instructions (excluding convert) by adding F32 <-> BF16 conversions. Unlike
// FloatNormalization, this pass does not use a backend-specific
// FloatSupport, and does not change HLOs that have BF16 data if they do not
// use mixed precision; it removes mixed precision even if the backend supports
// it. This pass is used to make the HLO module valid for other HLO passes which
// do not support mixed precision. Currently, this pass is only used by the
// Despecializer, not by our normal compilation flow on TPU.
class BFloat16MixedPrecisionRemoval : public HloModulePass {
 public:
  BFloat16MixedPrecisionRemoval() = default;

  ~BFloat16MixedPrecisionRemoval() override = default;

  absl::string_view name() const override {
    return "bf16-mixed-precision-removal";
  }

  // Run mixed precision removal on the given computation. Returns whether the
  // computation was changed.
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(HloModule* module,
                           const absl::flat_hash_set<absl::string_view>&
                               execution_threads) override {
    FloatNormalization normalization(&no_mixed_precision_support_);
    return normalization.Run(module, execution_threads);
  }

 private:
  class BFloat16SupportForMixedPrecisionRemoval : public FloatSupport {
   public:
    BFloat16SupportForMixedPrecisionRemoval() : FloatSupport(BF16) {}

    ~BFloat16SupportForMixedPrecisionRemoval() override = default;

    bool SupportsLowPrecisionOperand(const HloInstruction& hlo,
                                     int64_t operand_index) const override {
      return true;
    }

    bool SupportsLowPrecisionOutput(const HloInstruction& hlo) const override {
      return true;
    }

    bool SupportsMixedPrecisions(const HloInstruction& hlo) const override {
      return false;
    }
  } no_mixed_precision_support_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_FLOAT_NORMALIZATION_H_
