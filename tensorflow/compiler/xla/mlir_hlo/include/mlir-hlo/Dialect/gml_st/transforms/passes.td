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

def DeprecatedFusionPass : Pass<"gml-deprecated-fusion", "mlir::func::FuncOp"> {
  let summary = "Fuse producers in into `gml_st.materialize` operations";
  let constructor = "::mlir::gml_st::createDeprecatedFusionPass()";
}

def DeprecatedTilingPass : Pass<"gml-deprecated-tiling", "mlir::func::FuncOp"> {
  let summary = "Tile operations to produce gml_st.parallel or gml_st.for";
  let constructor = "::mlir::gml_st::createDeprecatedTilingPass()";
  let options = [
    Option<"tileSizesOpt", "tile-sizes", "std::string", /*default=*/"",
        "Tile sizes in the format `[[d0,d1,...],[d0_,d1_,...],...]`. This "
        "allows to express nested tiling sizes.">,
  ];
}

def TileToForPass : Pass<"gml-tile-to-for", "mlir::func::FuncOp"> {
  let summary = "Tile operations using TilingInterface to produce gml_st.for";
  let constructor = "::mlir::gml_st::createTileToForPass()";
  let options = [
    Option<"tilingTarget", "tiling-target", "std::string", /*default=*/"",
           "Operation with this label is the anchor to latch on.">,
    ListOption<"tileSizes", "tile-sizes", "int64_t", "Tile sizes",
               "llvm::cl::ZeroOrMore">,
  ];
}

def FusionPass : Pass<"gml-fusion", "mlir::func::FuncOp"> {
  let summary = "Fuse producers in into `gml_st.materialize` operations";
  let constructor = "::mlir::gml_st::createFusionPass()";
  let options = [
    Option<"producer", "producer", "std::string", /*default=*/"",
           "Producer label.">,
    Option<"consumer", "consumer", "std::string", /*default=*/"",
           "Consumer label.">,
  ];
}

def ComposeSetOpsPass : Pass<"gml-compose-set-ops", "mlir::func::FuncOp"> {
  let summary = "Compose set operations.";
  let constructor = "::mlir::gml_st::createComposeSetOpsPass()";
}

def CollapseMaterializeOpsPass : Pass<"gml-collapse-materialize-ops",
    "mlir::func::FuncOp"> {
  let summary = "Collapse (or uncollapse) materialize operations.";
  let constructor = "::mlir::gml_st::createCollapseMaterializeOpsPass()";
  let options = [
    Option<"reverse_", "reverse", "bool", /*default=*/"false", "Whether to "
        "revert collapsing of materialize operations.">,
  ];
}

def GmlStToScf : Pass<"gml-st-to-scf", "mlir::func::FuncOp"> {
  let summary = "Lower `gml_st.loop` to SCF loops and parallel loops";
  let constructor = "::mlir::gml_st::createGmlStToScfPass()";
  let dependentDialects = ["::mlir::scf::SCFDialect"];
}

def TiledLoopBufferizePass :
    Pass<"gml-tiled-loop-bufferize", "mlir::func::FuncOp"> {
  let summary = "Pass to bufferize linalg.tiled_loop with the ops inside it.";
  let constructor = "::mlir::gml_st::CreateTiledLoopBufferizePass()";
}

def VectorizeGmlStLoopsPass :
    Pass<"vectorize-gml-st-loops", "mlir::func::FuncOp"> {
  let summary =
      "Pass to vectorize linalg.generic ops tiled to gml_st.parallel and " #
      "gml_st.for loops.";
  let constructor = "::mlir::gml_st::createVectorizeGmlStLoopsPass()";
  let dependentDialects = ["::mlir::vector::VectorDialect"];
}
