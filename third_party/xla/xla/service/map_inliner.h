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

#ifndef XLA_SERVICE_MAP_INLINER_H_
#define XLA_SERVICE_MAP_INLINER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// A pass which performs map inlining. This replaces kMap instructions with
// their equivalent sequence of array operations. For example:
//   map({X, Y}, add) -> add(X, Y)).
class MapInliner : public HloModulePass {
 public:
  ~MapInliner() override = default;
  absl::string_view name() const override { return "map-inline"; }

  // Run map inlining on the given computation. Returns whether the computation
  // was changed.
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_MAP_INLINER_H_
