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

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Value.h"
#include "xla/service/elemental_ir_emitter.h"

namespace xla::cpu {

class CpuElementalIrEmitter : public ElementalIrEmitter {
 public:
  CpuElementalIrEmitter(llvm::Module* llvm_module, llvm::IRBuilderBase* builder,
                        bool use_truncate_f32_to_bf16_conversion,
                        bool fast_min_max)
      : ElementalIrEmitter(llvm_module, builder,
                           Options{use_truncate_f32_to_bf16_conversion}),
        fast_min_max_(fast_min_max) {}

 private:
  absl::StatusOr<llvm::Value*> EmitAtan2(PrimitiveType prim_type,
                                         llvm::Value* lhs, llvm::Value* rhs,
                                         absl::string_view) override;

  absl::StatusOr<llvm::Value*> EmitTanh(PrimitiveType prim_type,
                                        llvm::Value* value) override;

  absl::StatusOr<llvm::Value*> EmitErf(PrimitiveType prim_type,
                                       llvm::Value* value) override;

  bool fast_min_max() override { return fast_min_max_; }

  bool fast_min_max_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_ELEMENTAL_IR_EMITTER_H_
