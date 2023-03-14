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
  ];
  let options = [
    Option<"enableAVX2", "enable-avx2", "bool", /*default=*/"true",
           "Enable specialized lowerings for AVX2.">,
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

def TransformMatmulForCpuPass :
    Pass<"xla-cpu-transform-matmul", "mlir::func::FuncOp"> {
  let summary = "Transform matmul ops for running on CPU";

  let constructor = "createTransformMatmulForCpuPass()";

  let options = [
    Option<"lowerToMmt4D", "lower-to-mmt4d", "bool", "false",
           "If true, lower linalg.matmul into linalg.mmt4d">,
  ];
}

def TransformConvForCpuPass :
    Pass<"xla-cpu-transform-conv", "mlir::func::FuncOp"> {
  let summary = "Transform conv ops for running on CPU";
  let constructor = "createTransformConvForCpuPass()";
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

def TransformReverseForCpuPass :
  Pass<"xla-cpu-transform-reverse", "mlir::func::FuncOp"> {
    let summary = "Transform reverse ops for running on CPU";
    let constructor = "createTransformReverseForCpuPass()";
    let options = [
      Option<"vectorSize", "vector-size", "int64_t", "8",
           "Vector size for 'thlo.reverse`">,
    ];
  }

def AddDebugInfoPass :
    Pass<"add-debug-info", "mlir::ModuleOp"> {
  let summary = "Add debug info for the whole module";
  let constructor = "::mlir::gml_st::createAddDebugInfoPass()";
  let dependentDialects = ["::mlir::LLVM::LLVMDialect"];
}

def FusionPlanningForCpuPass :
    Pass<"gml-st-cpu-fusion-planning", "mlir::func::FuncOp"> {
  let summary = "Create fusion clusters.";
  let constructor = "createFusionPlanningForCpuPass()";
  let dependentDialects = [
    "::mlir::gml_st::GmlStDialect",
    "::mlir::arith::ArithDialect",
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
