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
    : Pass<"tf-jitrt-linalg-trivial-buffer-forwarding", "mlir::func::FuncOp"> {
  let summary = "Trivial input to output buffer forwarding for linalg.generic"
                " operations";
  let constructor = "tensorflow::CreateLinalgTrivialBufferForwardingPass()";
  let description = [{
    Trivial pass that reuses input buffers as outputs in linalg.generic
    operations. This pass does not try to do any buffer alias analysis, and
    is intented to work only as a part of TF -> JitRt compilation pipeline. It
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
    : Pass<"tf-jitrt-linalg-trivial-copy-removal", "mlir::func::FuncOp"> {
  let summary = "Trivial removal of memref.copy operations";
  let constructor = "tensorflow::CreateLinalgTrivialCopyRemovalPass()";
  let description = [{
    A simple pass that replaces patterns of the form

    %1 = memref.alloc() : memref<1x1xf32>
    memref.copy(%0, %1) : memref<1x1xf32>, memref<1x1xf32>
    memref.dealloc %0 : memref<1x1xf32>

    by replacing uses of %1 with %0 and dropping the copy.
  }];
}

def Fusion : Pass<"tf-jitrt-fusion", "mlir::func::FuncOp"> {
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

def Fission : Pass<"tf-jitrt-fission", "mlir::func::FuncOp"> {
  let summary = "Split _Fused Tensorflow kernels into primitives";
  let constructor = "tensorflow::CreateFissionPass()";
  let dependentDialects = [
    "mlir::TF::TensorFlowDialect"
  ];
}

def JitRtLegalizeI1Types
    : Pass<"tf-jitrt-legalize-i1-types", "mlir::ModuleOp"> {
  let summary = "Legalize 'i1' tensor types";
  let constructor = "tensorflow::CreateJitRtLegalizeI1TypesPass()";
  let description = [{
    Convert 'i1' tensor types used in any operation into 'i8' tensor types.
  }];
  let dependentDialects = [
    "mlir::TF::TensorFlowDialect",
    "mlir::mhlo::MhloDialect",
  ];
}

def Clustering : Pass<"tf-jitrt-clustering", "mlir::func::FuncOp"> {
  let summary = "Creates `tf_device.cluster` operations according to the TF "
                "JitRt clustering policy";

  let constructor = "tensorflow::CreateTfJitRtClusteringPass()";

  let dependentDialects = ["mlir::tf_device::TensorFlowDeviceDialect"];

  let options = [
   Option<"min_cluster_size", "min-cluster-size", "int" , /*default=*/"1",
          "Do not form clusters smaller of the given size.">,
   // TODO(ezhulenev): This is a temporary workaround to control TF->JitRt
   // clustering policy at runtime.
   ListOption<"oplist", "oplist", "std::string",
               "Explicitly allow operations for clustering. Only operations in "
               "this list will be passed to the TF->JitRt clustering policy. "
               "Alternatively use 'tier1', ..., 'all' to allow clustering for "
               "all operations included in the given clustering tier.">
  ];
}

def MathApproximation : Pass<"tf-jitrt-math-approximation", "mlir::func::FuncOp"> {
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
              "all supported math operations.">,
  ];
}
