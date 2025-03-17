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

#ifndef XLA_HLO_TRANSFORMS_LITERAL_CANONICALIZER_H_
#define XLA_HLO_TRANSFORMS_LITERAL_CANONICALIZER_H_

#include <cstddef>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/literal_pool.h"

namespace xla {

// Canonicalizes literals larger than 'min_size_bytes' in the HLO module using
// the given literal pool.
class LiteralCanonicalizer : public HloModulePass {
 public:
  LiteralCanonicalizer(LiteralPool* literal_pool, size_t min_size_bytes);

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  absl::string_view name() const override { return "literal-canonicalizer"; }

 protected:
  LiteralPool* literal_pool_;
  size_t min_size_bytes_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_LITERAL_CANONICALIZER_H_
