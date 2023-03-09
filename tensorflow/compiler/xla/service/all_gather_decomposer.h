/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_ALL_GATHER_DECOMPOSER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_ALL_GATHER_DECOMPOSER_H_

#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

namespace xla {

// AllGatherDecomposer is a pass which converts unsupported all-gathers into
// dynamic-update-slices and all-reduces.
class AllGatherDecomposer : public HloModulePass {
 public:
  explicit AllGatherDecomposer(
      std::function<bool(const HloAllGatherInstruction&)> should_decompose)
      : should_decompose_(std::move(should_decompose)) {}
  AllGatherDecomposer()
      : should_decompose_(
            [](const HloAllGatherInstruction& ag) { return true; }) {}
  absl::string_view name() const override { return "all_gather_decomposer"; }

  // Run AllGatherDecomposer pass on computations in 'module'.
  // Returns whether the 'module' was changed.
  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  std::function<bool(const HloAllGatherInstruction&)> should_decompose_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_ALL_GATHER_DECOMPOSER_H_
