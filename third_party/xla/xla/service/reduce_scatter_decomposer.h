/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_REDUCE_SCATTER_DECOMPOSER_H_
#define XLA_SERVICE_REDUCE_SCATTER_DECOMPOSER_H_

#include <functional>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

// A pass that decomposes a reduce-scatter into an all-reduce followed by a
// dynamic-slice.
class ReduceScatterDecomposer : public HloModulePass {
 public:
  explicit ReduceScatterDecomposer(
      std::function<void(Shape&)> update_layout = nullptr,
      std::function<bool(const HloInstruction*)> should_decompose = nullptr)
      : update_layout_(update_layout), should_decompose_(should_decompose) {}
  absl::string_view name() const override {
    return "reduce-scatter-decomposer";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
  std::function<void(Shape&)> update_layout_;
  std::function<bool(const HloInstruction*)> should_decompose_;
};

}  // namespace xla

#endif  // XLA_SERVICE_REDUCE_SCATTER_DECOMPOSER_H_
