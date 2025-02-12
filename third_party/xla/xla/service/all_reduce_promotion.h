/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_ALL_REDUCE_PROMOTION_H_
#define XLA_SERVICE_ALL_REDUCE_PROMOTION_H_

#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/service/change_op_data_type.h"
#include "xla/xla_data.pb.h"

namespace xla {

class AllReducePromotion : public HloModulePass {
 public:
  explicit AllReducePromotion(
      absl::Span<std::pair<PrimitiveType, PrimitiveType> const> from_to_types);
  absl::string_view name() const override { return "all-reduce-promotion"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  ChangeOpDataType pass_;
};

}  // namespace xla
#endif  // XLA_SERVICE_ALL_REDUCE_PROMOTION_H_
