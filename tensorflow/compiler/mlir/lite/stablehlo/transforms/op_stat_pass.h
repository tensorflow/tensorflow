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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_OP_STAT_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_OP_STAT_PASS_H_

#include <memory>
#include <string>
#include <vector>

#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace odml {

// Creates a pass which prints out a detailed report of conversion stats with:
// success or not, % of Ops non-converted, list of non-converted Ops, etc.
std::unique_ptr<Pass> createPrintOpStatsPass(
    std::vector<std::string> accepted_dialects);

}  // namespace odml
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STABLEHLO_TRANSFORMS_OP_STAT_PASS_H_
