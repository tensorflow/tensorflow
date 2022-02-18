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

def TestGmlStLoopPeeling : Pass<"test-gml-st-loop-peeling", "mlir::FuncOp"> {
  let summary = "Peel `gml_st.loop`";
  let constructor = "::mlir::gml_st::createTestGmlStLoopPeelingPass()";
  let options = [
    Option<"skip_partial", "skip-partial", "bool", /*default=*/"false",
           "Skip loops inside partial iterations during peeling">,
    ListOption<"dims", "dims", "unsigned", "Dimensions to peel",
           "llvm::cl::OneOrMore, llvm::cl::MiscFlags::CommaSeparated">,
  ];
}

def TestGmlStLoopTiling : Pass<"test-gml-st-loop-tiling", "mlir::FuncOp"> {
  let summary = "Tile `gml_st.loop`.";
  let constructor = "::mlir::gml_st::createTestGmlStLoopTilingPass()";
  let dependentDialects = [
    "AffineDialect",
    "gml_st::GmlStDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect"
  ];
  let options = [
    ListOption<"tile_sizes", "tile-sizes", "int64_t", "Tile sizes",
               "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">,
    ListOption<"distribution_types", "distribution-types", "std::string",
               "Distribution types",
               "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">

  ];
}
