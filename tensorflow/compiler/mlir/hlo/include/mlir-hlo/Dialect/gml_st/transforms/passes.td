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

include "mlir/Pass/PassBase.td"

def LegalizeMHLOToGMLPass : Pass<"legalize-mhlo-to-gml", "mlir::func::FuncOp"> {
  let summary = "TBD";
  let constructor = "::mlir::gml_st::createLegalizeMHLOToGMLPass()";
}

def FusionPass : Pass<"gml-fusion", "mlir::func::FuncOp"> {
  let summary = "Fuse producers in into `gml_st.materialize` operations";
  let constructor = "::mlir::gml_st::createFusionPass()";
}

def TilingPass : Pass<"gml-tiling", "mlir::func::FuncOp"> {
  let summary = "Tile operations to produce gml_st.parallel or gml_st.for";
  let constructor = "::mlir::gml_st::createTilingPass()";
  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t", "Tile sizes",
               "llvm::cl::ZeroOrMore">,
  ];
}

def ComposeSubsetOpsPass : Pass<"gml-compose-subset-ops",
    "mlir::func::FuncOp"> {
  let summary = "Compose subset operations.";
  let constructor = "::mlir::gml_st::createComposeSubsetOpsPass()";
}

def GmlStToScf : Pass<"gml-st-to-scf", "mlir::func::FuncOp"> {
  let summary = "Lower `gml_st.loop` to SCF loops and parallel loops";
  let constructor = "::mlir::gml_st::createGmlStToScfPass()";
  let dependentDialects = ["::mlir::scf::SCFDialect"];
}

def TiledLoopBufferizePass : Pass<"gml-tiled-loop-bufferize", "mlir::func::FuncOp"> {
  let summary = "Pass to bufferize linalg.tiled_loop with the ops inside it.";
  let constructor = "::mlir::gml_st::CreateTiledLoopBufferizePass()";
}
