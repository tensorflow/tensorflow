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

#ifndef XLA_BACKENDS_CPU_CODEGEN_EMITTERS_TRANSFORMS_PASSES_H_
#define XLA_BACKENDS_CPU_CODEGEN_EMITTERS_TRANSFORMS_PASSES_H_

#include <memory>

#include "mlir/Pass/Pass.h"
#include "xla/codegen/emitters/ir/xla_dialect.h"  // IWYU pragma: keep

namespace xla::cpu {

#define GEN_PASS_DECL
#include "xla/backends/cpu/codegen/emitters/transforms/passes.h.inc"

std::unique_ptr<mlir::Pass> CreateLowerToLLVMPass();
std::unique_ptr<mlir::Pass> CreateLowerXlaSharedPass();

#define GEN_PASS_REGISTRATION
#include "xla/backends/cpu/codegen/emitters/transforms/passes.h.inc"

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_CODEGEN_EMITTERS_TRANSFORMS_PASSES_H_
