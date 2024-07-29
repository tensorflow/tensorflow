/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_COLLECTIVE_QUANTIZER_H_
#define XLA_SERVICE_COLLECTIVE_QUANTIZER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

// Reduces the amount of data transferred in all-gather, all-to-all,
// collective-broadcast and collective-permute ops by exchanging the collectives
// with subsequent quantizations or type conversions to a narrower type. When
// present, unary ops such as bitcasts, copies, reshapes and slices between
// collective and quantization/type conversion are shifted, i.e. transforms
//
//   collective --> unary --> quantization/type conversion
//
// into
//
//   quantization/type conversion --> collective --> unary.
class CollectiveQuantizer : public HloModulePass {
 public:
  absl::string_view name() const override { return "collective-quantizer"; }

  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_COLLECTIVE_QUANTIZER_H_
