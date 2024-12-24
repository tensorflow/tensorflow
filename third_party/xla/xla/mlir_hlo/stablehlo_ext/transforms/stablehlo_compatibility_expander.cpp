/* Copyright 2024 The StableHLO Authors.
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

#include <memory>
#include <string>
#include <utility>

#include "mlir/Pass/Pass.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo_ext/transforms/passes.h"

namespace mlir {
namespace stablehlo_ext {

// TODO(b/369406385): remove this method (and file) once issue is resolved.

std::unique_ptr<::mlir::Pass> createStablehloCompatibilityExpanderPass(
    std::string targetVersionOption) {
  return mlir::stablehlo::createStablehloCompatibilityExpanderPass(
      {std::move(targetVersionOption)});
}

}  // namespace stablehlo_ext
}  // namespace mlir
