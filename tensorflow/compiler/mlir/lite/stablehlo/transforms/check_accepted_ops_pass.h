/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_CHECK_DIALECTS_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_CHECK_DIALECTS_PASS_H_

#include <memory>
#include <string>
#include <vector>

#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace odml {

// Creates a pass which checks if there exists allowed dialect ops only or not.
// Based on the list of dialect and op names, it signals failure or not.
// If some ops are in the `optional_accepted_dialects`, then it warns them.
std::unique_ptr<Pass> createCheckAcceptedOpsPass(
    const std::vector<std::string> &optional_accepted_dialects = {});

}  // namespace odml
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_CHECK_DIALECTS_PASS_H_
