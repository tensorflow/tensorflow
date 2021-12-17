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

include "mlir/Pass/PassBase.td"

def LinalgTrivialBufferForwarding
    : FunctionPass<"tf-cpurt-linalg-trivial-buffer-forwarding"> {
  let summary = "Trivial input to output buffer forwarding for linalg.generic"
                " operations";
  let constructor = "tensorflow::CreateLinalgTrivialBufferForwardingPass()";
  let description = [{
    Trivial pass that reuses input buffers as outputs in linalg.generic
    operations. This pass does not try to do any buffer alias analysis, and
    is intented to work only as a part of TF -> CPURT compilation pipeline. It
    will almost certainly produce invalid IR in any other use case.

    Input buffer forwarding requirements:
      1. Input and output memrefs must have the same shape.
      2. Input and output memrefs should have the same indexing map.
      3. Input and output buffer must be contiguous in memory.
      4. All iterator types must be "parallel".
      5. Input memref deallocated after the linalg.generic operation.

    In this case it is safe to use input memref as an output memref.
  }];
}

def LinalgTrivialCopyRemoval
    : FunctionPass<"tf-cpurt-linalg-trivial-copy-removal"> {
  let summary = "Trivial removal of linalg.copy operations";
  let constructor = "tensorflow::CreateLinalgTrivialCopyRemovalPass()";
  let description = [{
    A simple pass that replaces patterns of the form

    %1 = memref.alloc() : memref<1x1xf32>
    linalg.copy(%0, %1) : memref<1x1xf32>, memref<1x1xf32>
    memref.dealloc %0 : memref<1x1xf32>

    by replacing uses of %1 with %0 and dropping the copy.
  }];
}

def CodegenCWise : FunctionPass<"tf-cpurt-codegen-cwise"> {
  let summary = "Tile elementwise ops on tensors";
  let constructor = "tensorflow::CreateCodegenStrategyForCWisePass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
  ];
  let options = [
    Option<"cwise_tile_size", "cwise-tile-size",
           "int64_t", /*default=*/"8",
           "Tile size for the innermost dimension of an elementwise op.">,
  ];
}

def PeelTiledLoops : FunctionPass<"tf-cpurt-peel-tiled-loops"> {
  let summary = "Optimize away padding in tiled loops";
  let constructor = "tensorflow::CreatePeelTiledLoopsPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
  ];
}

def CodegenReduction : FunctionPass<"tf-cpurt-codegen-reduction"> {
  let summary = "Tile and fuse linalg.generic reduction on tensors.";
  let constructor = "tensorflow::CreateCodegenStrategyForReductionPass()";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect"
  ];

  let description = [{
    Matches linalg.generic to understand whether it is a reduction or not.
    After that performs tiling for vectorization and fusion of producers.
  }];

  let options = [
    Option<"reduction_1d_tile_size", "reduction-1d-tile-size",
           "int64_t", /*default=*/"8", "Tile size for a 1D reduction.">,
    ListOption<"reduction_2d_tile_sizes", "reduction-2d-tile-sizes", "int64_t",
               "Tile sizes for a 2D reduction.",
               "llvm::cl::MiscFlags::CommaSeparated">,
  ];
}

def FuseFillIntoTiledReduction
    : FunctionPass<"tf-cpurt-fuse-fill-into-tiled-reduction"> {
  let summary = "Fuse `linalg.fill` into `linalg.tiled_loop` with a reduction.";
  let constructor = "tensorflow::CreateFuseFillIntoTiledReductionPass()";
  let dependentDialects = ["mlir::linalg::LinalgDialect"];

  let description = [{
    Fuses `linalg.fill` producers of output tensor arguments into
    `linalg.tiled_loop`.
  }];
}

def Fission : FunctionPass<"tf-cpurt-fission"> {
  let summary = "Split _Fused Tensorflow kernels into primitives";
  let constructor = "tensorflow::CreateFissionPass()";
  let dependentDialects = [
    "mlir::TF::TensorFlowDialect"
  ];
}

def Fusion : FunctionPass<"tf-cpurt-fusion"> {
  let summary = "Fuse Linalg generic operations on Tensors";
  let constructor = "tensorflow::CreateFusionPass()";
  let description = [{
    Fuse Linalg generic operations on Tensors using custom heuristics for
    producer fusion profitability.
  }];
  let dependentDialects = [
    "mlir::TF::TensorFlowDialect"
  ];
}

def CpuRtLegalizeI1Types : Pass<"tf-cpurt-legalize-i1-types", "mlir::ModuleOp"> {
  let summary = "Legalize 'i1' tensor types";
  let constructor = "tensorflow::CreateCpuRtLegalizeI1TypesPass()";
  let description = [{
    Convert 'i1' tensor types used in any operation into 'i8' tensor types.
  }];
  let dependentDialects = [
    "mlir::TF::TensorFlowDialect",
    "mlir::mhlo::MhloDialect",
    "mlir::StandardOpsDialect",
  ];
}

def SymbolicShapeOptimization
    : FunctionPass<"tf-cpurt-symbolic-shape-optimization"> {
  let summary = "Optimizes broadcasts based on the symbolic shapes";
  let constructor = "tensorflow::CreateSymbolicShapeOptimizationPass()";
  let description = [{
    A simple pass that replaces shape constraints with const witnesses and
    rewrites mhlo.broadcast_in_dim operations with linalg.generic broadcasts
    using the symbolic shape attributes defined on the entrypoint function
    arguments.
  }];
  let dependentDialects = [
    "mlir::mhlo::MhloDialect",
    "mlir::linalg::LinalgDialect"
  ];

  let options = [
   Option<"optimize_only_constraints", "optimize-only-constraints",
          "bool", /*default=*/"false",
          "Optimize only shape constraints and do not touch broadcasts.">,

  ];
}

def Clustering : FunctionPass<"tf-cpurt-clustering"> {
  let summary = "Creates `tf_device.cluster` operations according to the TF "
                "CPURT clustering policy";

  let constructor = "tensorflow::CreateTfCpurtClusteringPass()";

  let dependentDialects = ["mlir::tf_device::TensorFlowDeviceDialect"];

  let options = [
   Option<"min_cluster_size", "min-cluster-size", "int" , /*default=*/"1",
          "Do not form clusters smaller of the given size.">,
   // TODO(ezhulenev): This is a temporary workaround to control TF->CPURT
   // clustering policy at runtime.
   ListOption<"oplist", "oplist", "std::string",
               "Explicitly allow operations for clustering. Only operations in "
               "this list will be passed to the TF->CPURT clustering policy. "
               "Alternatively use 'tier1', ..., 'all' to allow clustering for "
               "all operations included in the given clustering tier.",
               "llvm::cl::MiscFlags::CommaSeparated">
  ];
}

def VectorizeTiledOps : FunctionPass<"tf-cpurt-vectorize-tiled-ops"> {
  let summary = "Vectorize tiled Linalg ops inside `tiled_loop`";
  let constructor = "tensorflow::CreateVectorizeTiledOpsPass()";
  let dependentDialects = ["mlir::linalg::LinalgDialect"];
}

def DetensorizeLinalg : FunctionPass<"tf-cpurt-detensorize-linalg"> {
  let summary = "Replace 0d tensor inputs to LinalgOp with extracted elements.";
  let constructor = "tensorflow::CreateDetensorizeLinalgPass()";
  let dependentDialects = ["mlir::linalg::LinalgDialect"];
}

def MathApproximation : FunctionPass<"tf-cpurt-math-approximation"> {
  let summary = "Approximate math operations with an implementation meant to "
                "match Eigen's results. This is a useful property to have when "
                "comparing results from compiled TF code vs TF1 and TFRT, "
                "which use Eigen. "
                "TODO: evaluate the accuracy of these math approximations vs. "
                "those in upstream MLIR, and merge these upstream if they're "
                "more accurate.";
  let constructor = "tensorflow::CreateMathApproximationPass()";
  let options = [
   ListOption<"oplist", "oplist", "std::string",
              "List of math operations to be approximated. Use 'all' to select "
              "all supported math operations.",
              "llvm::cl::MiscFlags::CommaSeparated">,
  ];
}

def SinkUnusedOutputs : FunctionPass<"tf-cpurt-sink-unused-outputs"> {
  let summary = "Sink unused output args of `tiled_loop` into the loop body.";
  let constructor = "tensorflow::CreateSinkUnusedOutputs()";
  let dependentDialects = ["mlir::linalg::LinalgDialect"];
}
