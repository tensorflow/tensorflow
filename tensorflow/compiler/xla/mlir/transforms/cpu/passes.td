/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_MLIR_TRANSFORMS_CPU_PASSES_TD_
#define TENSORFLOW_COMPILER_XLA_MLIR_TRANSFORMS_CPU_PASSES_TD_

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Auxiliary passes for lowering to XLA Cpu runtime.
//===----------------------------------------------------------------------===//

def TileThloForCpuPass :
    Pass<"xla-cpu-tile-thlo", "mlir::func::FuncOp"> {
  let summary = "Tiles thlo ops for running on CPU";

  let description = [{
    Tiles thlo ops with the preferred factors of the CPU backend.
  }];

  let constructor = "createTileThloForCpuPass()";
}

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_TRANSFORMS_CPU_PASSES_TD_
