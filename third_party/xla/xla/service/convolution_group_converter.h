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

#ifndef XLA_SERVICE_CONVOLUTION_GROUP_CONVERTER_H_
#define XLA_SERVICE_CONVOLUTION_GROUP_CONVERTER_H_

#include <functional>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/status_macros.h"

namespace xla {

// A pass which rewrites convolutions with feature_group_count > 1 into
// convolutions with feature_group_count = 1.
class ConvolutionGroupConverter : public HloModulePass {
 public:
  ConvolutionGroupConverter(std::function<bool(HloInstruction*)> should_expand,
                            std::function<bool(HloInstruction*)> is_cost_viable,
                            bool convert_batch_groups_only,
                            bool filter_expansion = true)
      : should_expand_(should_expand),
        is_cost_viable_(is_cost_viable),
        convert_batch_groups_only_(convert_batch_groups_only),
        filter_expansion_(filter_expansion) {}

  absl::string_view name() const override {
    return "convolution-group-converter";
  }

  // Run convolution rewriting on the given computation. Returns whether the
  // computation was changed.
  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  // Predicate that determines whether this pass should rewrite a given
  // convolution.
  std::function<bool(HloInstruction*)> should_expand_;

  // Lambda containing cost model that decides whether to expand
  // batch_group_count.
  std::function<bool(HloInstruction*)> is_cost_viable_;

  // Decides whether to convert batch groups or feature groups.
  bool convert_batch_groups_only_;

  // Tells whether filter expansion is required.
  bool filter_expansion_;
};

}  // namespace xla

#endif  // XLA_SERVICE_CONVOLUTION_GROUP_CONVERTER_H_
