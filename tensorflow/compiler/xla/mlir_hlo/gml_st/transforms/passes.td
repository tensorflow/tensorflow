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

def TilingGPUWarpPass : Pass<"gml-tiling-gpu-warp", "mlir::func::FuncOp"> {
  let summary = "Tile warp-level ops for GPU";
  let constructor = "::mlir::gml_st::createTilingGpuWarpPass()";
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
    Option<"distributionLabel", "distribution-label", "std::string",
            /*default=*/"", "Distribution label for generated gml_st.parallel">,
  ];
}

def GreedyTilingAndFusionPass : Pass<"gml-greedy-tiling-and-fusion",
    "mlir::func::FuncOp"> {
  let summary = "Pass to tile the root operation and to greedily fuse "
      "producers into it.";
  let constructor = "::mlir::gml_st::createGreedyTilingAndFusionPass()";
  let options = [
    Option<"distribute", "distribute", "bool", /*default=*/"true",
           "Generate gml_st.parallel or gml_st.for">,
    ListOption<"tileSizes", "tile-sizes", "int64_t",
               "Tile sizes", "llvm::cl::ZeroOrMore">,
    Option<"distributionLabel", "distribution-label", "std::string",
           /*default=*/"", "Distribution label for generated gml_st.parallel">,
  ];
}

def CollapseShapePass : Pass<"gml-collapse-shape", "mlir::func::FuncOp"> {
  let summary = "Collapse dimensions of bcasts, reductions, and cwise ops";
  let description = [{
    Pass to collapse dimensions of bcasts, reductions, and cwise ops. A given
    number of trailing dimensions remains untouched while the remaining leading
    dimensions will be collapsed where possible.
  }];
  let options = [
    Option<"retainTrailingDims", "retain-trailing-dims", "int64_t",
           /*default=*/"0",
           "Number of trailing dimensions that will not be collapsed.">,
  ];
  let dependentDialects = ["::mlir::tensor::TensorDialect"];
}

def CollapseMaterializeOpsPass : Pass<"gml-collapse-materialize-ops",
    "mlir::func::FuncOp"> {
  let summary = "Collapse (or uncollapse) materialize operations.";
  let constructor = "::mlir::gml_st::createCollapseMaterializeOpsPass()";
}

def GmlStToScf : Pass<"gml-st-to-scf", "mlir::func::FuncOp"> {
  let summary = "Lower `gml_st.loop` to SCF loops and parallel loops";
  let constructor = "::mlir::gml_st::createGmlStToScfPass()";
  let dependentDialects = ["::mlir::scf::SCFDialect"];
}

def GmlStToGpuPass : Pass<"gml-st-to-gpu", "mlir::func::FuncOp"> {
  let summary = "Lower nested `gml_st.parallel` to `gpu.launch`";
  let dependentDialects = ["::mlir::gpu::GPUDialect",
                           "::mlir::vector::VectorDialect",
                           "::mlir::memref::MemRefDialect"];
  let options = [
    Option<"warpDistributionLabel", "warp-distribution-label",
           "std::string", /*default=*/"\"warp\"",
           "Direct children of `gml_st.parallel` loops with this distribution "
           "type are distributed over warps.">
  ];
}

def GmlStSimtfyPass : Pass<"gml-st-simtfy", "mlir::func::FuncOp"> {
  let summary = "Lower nested `gml_st.parallel` to `gpu.launch`";
  let dependentDialects = ["::mlir::AffineDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::gpu::GPUDialect",
                           "::mlir::scf::SCFDialect"];
  let options = [
    Option<"blockDistributionLabel", "block-distribution-label",
           "std::string", /*default=*/"\"block\"",
           "Direct children of `gml_st.parallel` loops with this distribution "
           "type are distributed over blocks.">
  ];
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
    ListOption<"distributionLabels", "included-distribution-labels",
               "std::string", "Distribution labels of gml_st.parallel ops "
               "where vectorization is allowed. Empty list signifies that "
               "vectorization is allowed within all loops.",
               "llvm::cl::ZeroOrMore">,
  ];
  let dependentDialects = ["::mlir::vector::VectorDialect"];
}

def VectorizePerfectlyTiledLoopsPass :
    Pass<"vectorize-perfectly-tiled-loops", "mlir::func::FuncOp"> {
  let summary = "Pass to vectorize gml_st.for loops that are tiled perfectly.";
  let constructor = "::mlir::gml_st::createVectorizePerfectlyTiledLoopsPass()";
  let dependentDialects = ["::mlir::vector::VectorDialect"];
}

def LowerVectorContractPass :
    Pass<"lower-vector-contract", "mlir::func::FuncOp"> {
  let summary = "Pass to lower vector.contract";
  let constructor = "::mlir::gml_st::createLowerVectorContractPass()";
  let dependentDialects = ["::mlir::vector::VectorDialect",
                           "::mlir::gml_st::GmlStDialect"];
}

def TransformScatterForCpuPass :
    Pass<"xla-cpu-transform-scatter", "mlir::func::FuncOp"> {
  let summary = "Transform scatter ops for running on CPU";

  let constructor = "createTransformScatterForCpuPass()";
}

def TransformMatmulForCpuPass :
    Pass<"xla-cpu-transform-matmul", "mlir::func::FuncOp"> {
  let summary = "Transform matmul ops for running on CPU";

  let constructor = "createTransformMatmulForCpuPass()";

  let options = [
    Option<"lowerToMmt4D", "lower-to-mmt4d", "bool", "false",
           "If true, lower linalg.matmul into linalg.mmt4d">,
    ListOption<"tileSizes", "tile-sizes", "int64_t",
               "Tile sizes for a `linalg.matmul`">,
  ];
}

def TransformMapForCpuPass :
    Pass <"gml-st-cpu-transform-map", "mlir::func::FuncOp"> {
  let summary = "Transform map ops for running on CPU";

  let constructor = "::mlir::gml_st::createTransformMapForCpuPass()";

  let options = [
    Option<"tileSize", "tile-size", "int64_t", "1",
           "Tile size for the innermost dimension of `linalg.map`">,
  ];
}

def TransformTransposeForCpuPass :
    Pass<"gml-st-cpu-transform-transpose", "mlir::func::FuncOp"> {
  let summary = "Transform transpose ops for running on CPU";

  let constructor = "createTransformTransposeForCpuPass()";

  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t",
               "Tile sizes for a `linalg.transpose`">,
  ];
}

def TransformReduceForCpuPass :
    Pass<"xla-cpu-transform-reduce", "mlir::func::FuncOp"> {
  let summary = "Transform reduce ops for running on CPU";

  let constructor = "createTransformReduceForCpuPass()";

  let options = [
    Option<"vectorSize", "vector-size", "int64_t", "8",
           "Vector size for a 1D `linalg.reduce`">,
    Option<"tileSize1D", "tile-size-1d", "int64_t", "32",
               "Tile size for a 1D `linalg.reduce`">,
    ListOption<"tileSizes2D", "tile-sizes-2d", "int64_t",
               "Tile sizes for a `linalg.reduce`. tileSizes[0] is the parallel "
               "dimension and tileSizes[1] is the reduction dimension.">,
  ];
}
