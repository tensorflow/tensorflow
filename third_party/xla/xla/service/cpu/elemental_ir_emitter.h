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

#ifndef XLA_SERVICE_CPU_ELEMENTAL_IR_EMITTER_H_
#define XLA_SERVICE_CPU_ELEMENTAL_IR_EMITTER_H_

#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/service/elemental_ir_emitter.h"

namespace xla::cpu {

class CpuElementalIrEmitter final : public ElementalIrEmitter {
 public:
  using ThreadLocalCallPrototype = absl::StatusOr<std::vector<llvm::Value*>>(
      const HloComputation& callee, absl::Span<llvm::Value* const> parameters,
      absl::string_view name, bool is_reducer);
  using ThreadLocalCallCallback = absl::AnyInvocable<ThreadLocalCallPrototype>;

  CpuElementalIrEmitter(llvm::Module* llvm_module, llvm::IRBuilderBase* builder,
                        ThreadLocalCallCallback thread_local_call_fn,
                        bool use_truncate_f32_to_bf16_conversion,
                        bool fast_min_max)
      : ElementalIrEmitter(llvm_module, builder,
                           Options{use_truncate_f32_to_bf16_conversion}),
        thread_local_call_fn_(std::move(thread_local_call_fn)),
        fast_min_max_(fast_min_max) {}

 private:
  absl::StatusOr<llvm::Value*> EmitAtan2(PrimitiveType prim_type,
                                         llvm::Value* lhs, llvm::Value* rhs,
                                         absl::string_view) override;

  absl::StatusOr<llvm::Value*> EmitTanh(PrimitiveType prim_type,
                                        llvm::Value* value) override;

  absl::StatusOr<llvm::Value*> EmitErf(PrimitiveType prim_type,
                                       llvm::Value* value) override;

  absl::StatusOr<llvm::Value*> EmitExp(PrimitiveType prim_type,
                                       llvm::Value* value,
                                       absl::string_view name) override;

  absl::StatusOr<std::vector<llvm::Value*>> EmitThreadLocalCall(
      const HloComputation& callee, absl::Span<llvm::Value* const> parameters,
      absl::string_view name, bool is_reducer) override;

  bool fast_min_max() override { return fast_min_max_; }

 private:
  ThreadLocalCallCallback thread_local_call_fn_;
  bool fast_min_max_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_ELEMENTAL_IR_EMITTER_H_
