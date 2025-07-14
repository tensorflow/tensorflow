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

#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_SERDE_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_SERDE_H_

#include <memory>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "xla/mosaic/pass_boilerplate.h"

namespace mosaic::gpu {

struct SerdePassOptions {
  bool serialize;
  int target_version;
};

struct SerdePass : public jaxlib::mlir::Pass<SerdePass, mlir::ModuleOp> {
  using jaxlib::mlir::Pass<SerdePass, mlir::ModuleOp>::Pass;

  static constexpr llvm::StringLiteral kArgumentName = "mosaic_gpu-serde";
  static constexpr llvm::StringLiteral kPassName = "MosaicGPUSerdePass";

  SerdePass() = default;

  explicit SerdePass(SerdePassOptions options) {
    serialize = options.serialize;
    target_version = options.target_version;
  }

  SerdePass(const SerdePass &other) {
    serialize = other.serialize;
    target_version = other.target_version;
  }

  SerdePass &operator=(const SerdePass &other) {
    serialize = other.serialize;
    target_version = other.target_version;
    return *this;
  }

  void runOnOperation();

 protected:
  ::mlir::Pass::Option<bool> serialize{*this, "serialize", llvm::cl::desc("")};
  ::mlir::Pass::Option<int> target_version{*this, "target-version",
                                           llvm::cl::desc("")};
};

inline std::unique_ptr<::mlir::Pass> createSerdePass() {
  return std::make_unique<SerdePass>();
}

inline std::unique_ptr<::mlir::Pass> createSerdePass(SerdePassOptions options) {
  return std::make_unique<SerdePass>(std::move(options));
}

inline void registerSerdePass() {
  ::mlir::registerPass(
      []() -> std::unique_ptr<::mlir::Pass> { return createSerdePass(); });
}

}  // namespace mosaic::gpu

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_GPU_SERDE_H_
