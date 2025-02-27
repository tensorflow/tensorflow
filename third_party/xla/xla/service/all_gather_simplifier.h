/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_ALL_GATHER_SIMPLIFIER_H_
#define XLA_SERVICE_ALL_GATHER_SIMPLIFIER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// A pass that detects unnecessary all-gathers and replaces them with their
// operands. Examples:
// 1. a trivial all-gather where the input and output shapes are
// compatible will be replaced by its operand.
//
// 2. an all-gather with a single consumer that is a dynamic-slice such that the
// output of the dynamic-slice is the same as the input of the all-gather will
// be replaced by the operand.
class AllGatherSimplifier : public HloModulePass {
 public:
  static constexpr absl::string_view kName = "all-gather-simplifier";
  absl::string_view name() const override { return kName; }

  // Run all-gather simplification on the given computation. Returns whether the
  // computation was changed.
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_ALL_GATHER_SIMPLIFIER_H_
