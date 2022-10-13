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

def TilingPass : Pass<"gml-tiling", "mlir::func::FuncOp"> {
  let summary = "Tile operations using TilingInterface to produce gml_st.for";
  let constructor = "::mlir::gml_st::createTilingPass()";
  let options = [
    Option<"opName", "op-name", "std::string", /*default=*/"",
           "Operation with this name is the anchor to latch on.">,
    Option<"opLabel", "op-label", "std::string", /*default=*/"",
           "Operation with this label is the anchor to latch on.">,
    Option<"distribute", "distribute", "bool", /*default=*/"true",
           "Generate gml_st.parallel or gml_st.for">,
    ListOption<"tileSizes", "tile-sizes", "int64_t", "Tile sizes",
               "llvm::cl::ZeroOrMore">,
  ];
}

def FusionPass : Pass<"gml-fusion", "mlir::func::FuncOp"> {
  let summary = "Fuse producers in into `gml_st.materialize` operations";
  let constructor = "::mlir::gml_st::createFusionPass()";
  let options = [
    Option<"producerLabel", "producer-label", "std::string", /*default=*/"",
           "Producer label.">,
    Option<"consumerLabel", "consumer-label", "std::string", /*default=*/"",
           "Consumer label.">,
  ];
}

def TilingCwisePass : Pass<"gml-tiling-cwise", "mlir::func::FuncOp"> {
  let summary = "Tile and fuse all cwise ops";
  let constructor = "::mlir::gml_st::createTilingCwisePass()";
  let options = [
    Option<"distribute_", "distribute", "bool", /*default=*/"true",
           "Generate gml_st.parallel or gml_st.for">,
    ListOption<"tileSizes_", "tile-sizes", "int64_t", 
               "Right-aligned tile sizes. Do not tile possible remaining "
               "dimensions", "llvm::cl::ZeroOrMore">,
    Option<"distributionLabel_", "distribution-label", "std::string",
            /*default=*/"", "Distribution label for generated gml_st.parallel">,
  ];
}

def TilingReductionPass : Pass<"gml-tiling-reduction", "mlir::func::FuncOp"> {
  let summary = "Tile linalg.generic reductions";
  let constructor = "::mlir::gml_st::createTilingReductionPass()";
  let dependentDialects = ["::mlir::gml_st::GmlStDialect",
                           "::mlir::arith::ArithDialect"];
}

def TilingSoftmaxPass : Pass<"gml-tiling-softmax", "mlir::func::FuncOp"> {
  let summary = "Match, tile, and fuse softmax implementations";
  let constructor = "::mlir::gml_st::createTilingSoftmaxPass()";
  let options = [
    Option<"distribute", "distribute", "bool", /*default=*/"true",
           "Generate gml_st.parallel or gml_st.for">,
    ListOption<"tileSizes", "tile-sizes", "int64_t",
               "Right-aligned tile sizes. Do not tile possible remaining "
               "dimensions", "llvm::cl::ZeroOrMore">,
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

def GmlStToGpuPass : Pass<"gml-st-to-gpu", "mlir::func::FuncOp"> {
  let summary = "Lower nested `gml_st.parallel` to `gpu.launch`";
  let dependentDialects = ["::mlir::AffineDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::gpu::GPUDialect", "::mlir::scf::SCFDialect",
                           "::mlir::vector::VectorDialect"];
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
  let options = [
    Option<"vectorizeGmlStOps", "vectorize-gml-st-ops", "bool", "false",
           "If true, vectorizes GmlSt ops in addition to linalg ops">,
  ];
  let dependentDialects = ["::mlir::vector::VectorDialect"];
}
