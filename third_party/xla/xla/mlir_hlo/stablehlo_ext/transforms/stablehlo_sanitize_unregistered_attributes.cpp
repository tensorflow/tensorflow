/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2023 The StableHLO Authors.
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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "stablehlo_ext/transforms/passes.h"  // NOLINT: Used in passes.h.inc
#include "utils/unregistered_attributes.h"

namespace mlir {
namespace stablehlo_ext {

#define GEN_PASS_DEF_STABLEHLOSANITIZEUNREGISTEREDATTRIBUTESPASS
#include "stablehlo_ext/transforms/passes.h.inc"

namespace {

struct StablehloSanitizeUnregisteredAttributesPass
    : public impl::StablehloSanitizeUnregisteredAttributesPassBase<
          StablehloSanitizeUnregisteredAttributesPass> {
  using StablehloSanitizeUnregisteredAttributesPassBase::
      StablehloSanitizeUnregisteredAttributesPassBase;

  void runOnOperation() override {
    // Remove unregistered attributes from module.
    ModuleOp module = getOperation();
    for (auto attr : module->getDiscardableAttrs()) {
      auto name = attr.getName();
      if (!xla::IsKnownUnregisteredAttribute({name.data(), name.size()})) {
        module->removeDiscardableAttr(name);
      }
    }

    // Remove unregistered attributes from functions.
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      for (auto attr : func->getDiscardableAttrs()) {
        auto name = attr.getName();
        if (!xla::IsKnownUnregisteredAttribute({name.data(), name.size()})) {
          func->removeDiscardableAttr(name);
        }
      }
    }
  }
};

}  // namespace
}  // namespace stablehlo_ext
}  // namespace mlir
