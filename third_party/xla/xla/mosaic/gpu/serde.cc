/* Copyright 2025 The JAX Authors.

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

#include "xla/mosaic/gpu/serde.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "xla/mosaic/serde.h"

namespace mosaic::gpu {

namespace {

constexpr llvm::StringRef kMangledDialect = "stable_mosaic_gpu.";
constexpr llvm::StringRef kVersionAttrName = "stable_mosaic_gpu.version";
// When this is bumped, we should file a TODO to update the forward-compatible
// version in Mosaic GPU lowering in a month!
constexpr int kVersion = 1;

using SerdeRuleType = jaxlib::mosaic::SerdeRuleType;

const llvm::StringMap<SerdeRuleType>& upgrade_rules() {
  static auto rules = new llvm::StringMap<SerdeRuleType>{};
  return *rules;
}

const llvm::StringMap<SerdeRuleType>& downgrade_rules() {
  static auto rules = new llvm::StringMap<SerdeRuleType>{};
  return *rules;
}

}  // namespace

void SerdePass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  if (!serialize.hasValue()) {
    module.emitError("serialize option must be specified");
    return signalPassFailure();
  }
  int serialize_version = -1;
  if (serialize) {
    serialize_version = target_version.hasValue() ? target_version : kVersion;
  }
  if (mlir::failed(jaxlib::mosaic::RunSerde(
          module, upgrade_rules(), downgrade_rules(), serialize,
          {.dialect_prefix = kMangledDialect,
           .highest_version = kVersion,
           .version_attr_name = kVersionAttrName,
           .serialize_version = serialize_version}))) {
    signalPassFailure();
  }
}

}  // namespace mosaic::gpu
