/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_SHAPE_CANONICALIZER_H_
#define XLA_HLO_TRANSFORMS_SHAPE_CANONICALIZER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/shape_pool.h"

namespace xla {

// Canonicalizes HLO instructions shapes in the HLO module using the given shape
// pool. Most of the instructions in the HLO module has the same shape, and by
// sharing shapes we can reduce the memory usage of the HLO module.
class ShapeCanonicalizer : public HloModulePass {
 public:
  class Visitor;

  explicit ShapeCanonicalizer(ShapePool* shape_pool);

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  absl::string_view name() const override { return "shape-canonicalizer"; }

 protected:
  ShapePool* shape_pool_;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SHAPE_CANONICALIZER_H_
