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

#ifndef XLA_SERVICE_CHANGE_OP_DATA_TYPE_H_
#define XLA_SERVICE_CHANGE_OP_DATA_TYPE_H_

#include <functional>
#include <memory>
#include <utility>

#include "xla/service/hlo_pass_interface.h"

namespace xla {

// Changes `from_ty op(from_ty a, from_ty b)` into
// `from_ty convert(op(to_ty convert(a), to_ty convert(b)))`.
//
// One place where this pass is useful is for fp16 dots/convs in XLA:CPU.
// Although XLA:CPU supports fp16 dots/convs, they are significantly slower than
// fp32 convs.   This pass lets us run the fp16 dot/conv as "convert to fp32,
// run in fp32, then convert back to fp16".  (This is of course not
// mathematically the same, but it's close enough for our purposes.)
//
// This pass only considers ops that match `op_matcher` and where all operands
// have type `from_ty`.  It will not do the correct thing for ops like
// dynamic-slice where only some of the arguments should be converted; it's up
// to you to avoid matching such ops with `op_matcher`.
//
// The pass support multiple <from_ty, to_ty> pairs and will apply the transform
// if all operands match one of the types in from_ty.
//
// It uses provided `cloner` to clone an instruction with shape and converted
// operands. If the cloner is not provided, it will uses `CloneWithNewOperands`.
class ChangeOpDataType : public HloModulePass {
 public:
  using HloCloner = std::function<std::unique_ptr<HloInstruction>(
      const HloInstruction*, const Shape&, absl::Span<HloInstruction* const>)>;
  ChangeOpDataType(
      absl::Span<std::pair<PrimitiveType, PrimitiveType> const> from_to_types,
      HloPredicate op_matcher, HloCloner cloner = nullptr)
      : op_matcher_(op_matcher), cloner_(cloner) {
    for (const std::pair<PrimitiveType, PrimitiveType>& pair : from_to_types) {
      to_type_map_[pair.first] = pair.second;
    }
  }

  ChangeOpDataType(PrimitiveType from_ty, PrimitiveType to_ty,
                   HloPredicate op_matcher, HloCloner cloner = nullptr)
      : op_matcher_(op_matcher), cloner_(cloner) {
    to_type_map_[from_ty] = to_ty;
  }

  absl::string_view name() const override { return "change-op-data-type"; }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  // map with key = from_type and value = to_type.
  absl::flat_hash_map<PrimitiveType, PrimitiveType> to_type_map_;
  HloPredicate op_matcher_;
  HloCloner cloner_;
};

}  // namespace xla

#endif  // XLA_SERVICE_CHANGE_OP_DATA_TYPE_H_
