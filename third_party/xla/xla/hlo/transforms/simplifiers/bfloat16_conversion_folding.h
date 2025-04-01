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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_BFLOAT16_CONVERSION_FOLDING_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_BFLOAT16_CONVERSION_FOLDING_H_

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/float_support.h"
#include "xla/xla_data.pb.h"

namespace xla {

// A pass which folds F32 <-> BF16 conversions to their operands or users, when
// it is supported by the backend.
//
// This pass follows the passed-in backend-specific BF16 support rules, but can
// introduce mixed precision in individual HLOs which breaks the assumption of
// some other HLO passes. So it should be used at the end of the HLO
// optimization pipeline followed by a DCE pass. If other passes are needed
// after this pass, run BFloat16MixedPrecisionRemoval first to undo some of the
// changed made by this pass.
class BFloat16ConversionFolding : public HloModulePass {
 public:
  explicit BFloat16ConversionFolding(const FloatSupport* bfloat16_support)
      : bfloat16_support_(bfloat16_support) {
    DCHECK(bfloat16_support->LowPrecisionType() == BF16);
  }

  ~BFloat16ConversionFolding() override = default;
  absl::string_view name() const override { return "bfloat16-fold"; }

  // Run BF16 conversion folding on the given computation. Returns whether the
  // computation was changed.
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const FloatSupport* bfloat16_support_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_BFLOAT16_CONVERSION_FOLDING_H_
