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

def TilingSoftmaxPass : Pass<"gml-tiling-softmax", "mlir::func::FuncOp"> {
  let summary = "Match, tile, and fuse softmax implementations";
  let constructor = "::mlir::gml_st::createTilingSoftmaxPass()";
  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t",
               "Right-aligned tile sizes. Do not tile possible remaining "
               "dimensions", "llvm::cl::ZeroOrMore">,
  ];
}

def TileByOnePass : Pass<"gml-tile-by-one", "mlir::func::FuncOp"> {
  let summary = "Tile all tileable ops by size 1";
  let description = [{
    Tile all tileable ops to size 1. This is meant as a fallback for those ops
    that were not previously tiled and vectorized.
  }];
  let constructor = "::mlir::gml_st::createTileByOnePass()";
}

def CollapseShapePass : Pass<"gml-collapse-shape", "mlir::func::FuncOp"> {
  let summary = "Collapse dimensions of bcasts, reductions, and cwise ops";
  let description = [{
    Pass to collapse dimensions of bcasts, reductions, and cwise ops. A given
    number of trailing dimensions remains untouched while the remaining leading
    dimensions will be collapsed where possible.
  }];
  let constructor = "::mlir::gml_st::createCollapseShapePass()";
  let options = [
    Option<"retainTrailingDims", "retain-trailing-dims", "int64_t",
           /*default=*/"0",
           "Number of trailing dimensions that will not be collapsed.">,
  ];
  let dependentDialects = ["::mlir::tensor::TensorDialect"];
}

def ComposeExtractInsertSlicePass : Pass<"gml-compose-extract-insert-slice",
    "mlir::func::FuncOp"> {
  let summary = "Compose tensor.extract_slice/insert_slice ops.";
  let constructor = "::mlir::gml_st::createComposeExtractInsertSlicePass()";
}

def VectorizeForCPUPass : Pass<"vectorize-for-cpu", "mlir::func::FuncOp"> {
  let summary = "Pass to vectorize gml_st.for loops that are tiled perfectly.";
  let constructor = "::mlir::gml_st::createVectorizeForCPUPass()";
  let dependentDialects = [
    "::mlir::vector::VectorDialect",
    "::mlir::tensor::TensorDialect"
  ];
  let options = [
    Option<"numElementsThreshold", "num-elements-threshold", "int64_t",
           /*default=*/"128",
           "Number of elements max of the tensor operands in order for the op to be vectorized.">,
  ];
}

def VectorizeCopyPass :
    Pass<"vectorize-copy", "mlir::func::FuncOp"> {
  let summary = "Pass to vectorize `memref.copy`.";
  let constructor = "::mlir::gml_st::createVectorizeCopyPass()";
  let dependentDialects = ["::mlir::vector::VectorDialect"];
}

def NaiveCopyRemovalPass : Pass<"naive-copy-removal", "mlir::func::FuncOp"> {
  let summary = "Pass to remove redundant `memref.copy` ops.";
  let constructor = "::mlir::gml_st::createNaiveCopyRemovalPass()";
  let dependentDialects = ["::mlir::memref::MemRefDialect"];
}

def LowerVectorsPass : Pass<"lower-vectors", "mlir::func::FuncOp"> {
  let summary = "Pass to lower vector operations progressively.";
  let constructor = "::mlir::gml_st::createLowerVectorsPass()";
  let dependentDialects = [
    "::mlir::LLVM::LLVMDialect",
    "::mlir::vector::VectorDialect",
    "::mlir::affine::AffineDialect",
  ];
  let options = [
    Option<"enableAVX2", "enable-avx2", "bool", /*default=*/"true",
           "Enable specialized lowerings for AVX2.">,
    Option<"flatten", "flatten", "bool", /*default=*/"false",
           "Flatten multiple small n-D vector transfers into a large 1-D transfer.">,
  ];
}

def ScalarizationPass : Pass<"scalarize", "mlir::func::FuncOp"> {
  let summary = "Converts ops on tensors with 1 element to scalar ops.";
  let dependentDialects = [
    "arith::ArithDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect"
  ];
  let constructor = "createScalarizationPass()";
  let options = [
    Option<"scalarizeAllThlo", "scalarize-all-thlo", "bool", /*default=*/"true",
           "Enable scalarization of thlo.concatenate/gather/scatter.">,
  ];
}

def PackMatmulPass : Pass<"xla-cpu-pack-matmul", "mlir::func::FuncOp"> {
  let summary = "Pack linalg.matmul as linalg.mmt4d";
  let constructor = "createPackMatmulPass()";
}

def TransformScatterForCpuPass :
    Pass<"xla-cpu-transform-scatter", "mlir::func::FuncOp"> {
  let summary = "Transform scatter ops for running on CPU";

  let constructor = "createTransformScatterForCpuPass()";
}

def TransformDotForCpuPass :
    Pass<"xla-cpu-transform-dot", "mlir::func::FuncOp"> {
  let summary = "Transform dot ops for running on CPU";
  let constructor = "createTransformDotForCpuPass()";
}

def TransformPackForCpuPass :
    Pass<"xla-cpu-transform-pack", "mlir::func::FuncOp"> {
  let summary = "Transform tensor.pack/unpack ops for running on CPU";
  let constructor = "createTransformPackForCpuPass()";
}

def TransformMmt4DForCpuPass :
    Pass<"xla-cpu-transform-mmt4d", "mlir::func::FuncOp"> {
  let summary = "Transform linalg.mmt4d ops for running on CPU";
  let constructor = "createTransformMmt4DForCpuPass()";
}

def TransformElementwiseForCpuPass :
    Pass <"xla-cpu-transform-elementwise", "mlir::func::FuncOp"> {
  let summary = "Transform elementwise ops for running on CPU";
  let description = [{
    Transforms elementwise ops, i.e. map, transpose, broadcast, concat, reverse.
  }];
  let constructor = "::mlir::gml_st::createTransformElementwiseForCpuPass()";

  let options = [
    Option<"vectorSize", "vector-size", "int64_t", "8", "Vector size.">,
    Option<"fuseDegenerateReshapes", "fuse-degenerate-reshapes", "bool",
           /*default=*/"false",
           "Fuse through degenerate tensor.expand/collapse_shape">,
  ];
}

def TransformReduceForCpuPass :
    Pass<"xla-cpu-transform-reduce", "mlir::func::FuncOp"> {
  let summary = "Transform reduce ops for running on CPU";
  let options = [
    Option<"enableHeuristic", "enable_heuristic", "bool", "false",
           "Enable heuristic for tiling sizes. Currently only for 1D.">,
    Option<"tileSize1D", "reduction-1d-tile-size", "int64_t", "32",
           "Tile size for a 1D reduction.">,
    Option<"splitRatio1D", "reduction-1d-split-ratio", "int64_t", "8",
           "Ratio used to split the reduction dimension, i.e. tiled reduce op "
           "`reduce(tensor<N>)` will be split into a composition of a "
           " column reduction `reduce(tensor<N/splitRatio1D x splitRatio1D>)` "
           "and a row 1D reductionreduce(tensor<splitRatio1D>)`." >,
    Option<"parallelDimTileSize2D",
            "reduction-2d-parallel-dim-tile-size", "int64_t", "4",
           "Tile size for the parallel dimension of a 2D reduction.">,
    Option<"reductionDimTileSize2D",
           "reduction-2d-reduction-dim-tile-size", "int64_t", "4",
           "Tile size for the reduction dimension of a 2D reduction.">,
  ];
  let constructor = "::mlir::gml_st::createTransformReduceForCpuPass()";
}

def AddDebugInfoPass :
    Pass<"add-debug-info", "mlir::ModuleOp"> {
  let summary = "Add debug info for the whole module";
  let constructor = "::mlir::gml_st::createAddDebugInfoPass()";
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
}

def CollectStatsPass :
    Pass<"collect-stats", "mlir::func::FuncOp"> {
  let summary = "Print stats about tileable ops";
  let constructor = "::mlir::gml_st::createCollectStatsPass()";
}

def RemoveLabelPass : Pass<"remove-label", "mlir::func::FuncOp"> {
  let summary = "Remove transformed labels from tiled ops";
  let constructor = "::mlir::gml_st::createRemoveLabelPass()";
}

def FusionPlanningForCpuPass :
    Pass<"gml-st-cpu-fusion-planning", "mlir::func::FuncOp"> {
  let summary = "Create fusion clusters.";
  let constructor = "createFusionPlanningForCpuPass()";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::gml_st::GmlStDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::tensor::TensorDialect"
  ];

  let options = [
    Option<"vectorSize", "vector-size", "int64_t", "8",
           "Tile size for the innermost dimension of `linalg.map`">,
  ];
}

def RewriteFromElementsOpPass
    : Pass<"gml-st-rewrite-from-elements-ops", "mlir::func::FuncOp"> {
  let summary = "Pass to rewrite tensor.from_elements into tensor.insert.";
  let constructor = "createRewriteFromElementsOpPass()";
}

def RewriteForallOpPass
    : Pass<"gml-st-rewrite-forall-ops", "mlir::func::FuncOp"> {
  let summary = "Pass to rewrite scf.forall to scf.for.";
  let constructor = "createRewriteForallOpPass()";
}

def FusionOutliningPass : Pass<"gml-fusion-outlining", "mlir::ModuleOp"> {
  let summary = "Pass to outline fusion regions into functions.";
  let constructor = "createFusionOutliningPass()";
}

def InlineFusionClustersPass :
    Pass<"gml-st-inline-fusion-clusters", "mlir::func::FuncOp"> {
  let summary = "Replaces all gml_st.fusion op with ops from the region.";
  let constructor = "createInlineFusionClustersPass()";
  let dependentDialects = [
    "::mlir::gml_st::GmlStDialect",
  ];
}

def OptimizeLinalgOpsPass
    : Pass<"gml-st-optimize-linalg-ops-pass", "mlir::func::FuncOp"> {
  let summary = "Canonicalization patterns for linalg ops.";
  let constructor = "createOptimizeLinalgOpsPass()";

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::tensor::TensorDialect"
  ];
}
