/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_CODEGEN_XTILE_IR_TRANSFORMS_PASSES_H_
#define XLA_CODEGEN_XTILE_IR_TRANSFORMS_PASSES_H_

#include <memory>  // IWYU pragma: keep

#include "mlir/IR/BuiltinOps.h"  // IWYU pragma: keep
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep

namespace xla::xtile {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "xla/codegen/xtile/ir/transforms/passes.h.inc"

}  // namespace xla::xtile

#endif  // XLA_CODEGEN_XTILE_IR_TRANSFORMS_PASSES_H_
