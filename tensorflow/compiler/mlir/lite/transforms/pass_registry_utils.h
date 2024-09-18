/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PASS_REGISTRY_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PASS_REGISTRY_UTILS_H_

#include <memory>

#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project

namespace mlir {
namespace TFL {

////////////////////////////////////////////////////////////////////////////////
// Pass, Pipeline and Options Creation Utilities
////////////////////////////////////////////////////////////////////////////////

template <typename PassType>
std::unique_ptr<mlir::Pass> Create() {
  return std::make_unique<PassType>();
}

template <typename PassType>
std::unique_ptr<mlir::Pass> Create(const mlir::detail::PassOptions& options) {
  return std::make_unique<PassType>(options);
}

////////////////////////////////////////////////////////////////////////////////
// Registration Utilities
////////////////////////////////////////////////////////////////////////////////

// Utility to register a pass without options.
template <typename PassType>
void Register() {
  PassRegistration<PassType> pass([] { return Create<PassType>(); });
}

// Utility to register a pass with options.
template <typename PassType, typename PassOptionsType>
void Register() {
  auto pass_argument = PassType::GetArgument();
  auto pass_description = PassType::GetDescription();

  PassPipelineRegistration<PassOptionsType>(
      pass_argument, pass_description,
      [](OpPassManager& pm, const PassOptionsType& options) {
        pm.addPass(std::move(Create<PassType>(options)));
      });
}

}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PASS_REGISTRY_UTILS_H_
