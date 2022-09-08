/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CHANGE_OP_DATA_TYPE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CHANGE_OP_DATA_TYPE_H_

#include <functional>

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

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
class ChangeOpDataType : public HloModulePass {
 public:
  ChangeOpDataType(PrimitiveType from_ty, PrimitiveType to_ty,
                   std::function<bool(const HloInstruction*)> op_matcher)
      : from_ty_(from_ty), to_ty_(to_ty), op_matcher_(op_matcher) {}

  absl::string_view name() const override { return "change-op-data-type"; }
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  PrimitiveType from_ty_;
  PrimitiveType to_ty_;
  std::function<bool(const HloInstruction*)> op_matcher_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CHANGE_OP_DATA_TYPE_H_
