/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifdef XLA_GPU_OPS
#else
#define XLA_GPU_OPS

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"

include "tensorflow/compiler/xla/mlir/backends/gpu2/ir/xla_gpu_dialect.td"

//===----------------------------------------------------------------------===//
// Op definitions
//===----------------------------------------------------------------------===//

class XLAGPU_Op<string mnemonic, list<Trait> traits = []> :
      Op<XlaGpuDialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// XLA:GPU graph region operation
//===----------------------------------------------------------------------===//

def XLAGPU_GraphRegionOp : XLAGPU_Op<"graph.region", [
    NoTerminator,
    NoRegionArguments
  ]> {
  let summary = "marker for XLA:GPU graph regions";

  let description = [{
    This is a marker to group operations scheduled for later extraction into a
    GPU graph (CUDA graph for NVIDIA backend).

    Example: fusion operation prepared to become a GPU graph

    ```mlir
      xla_gpu.graph.region {
        lmhlo.fusion {
          ...
        }
      }
    ```
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$body attr-dict";
}

//===----------------------------------------------------------------------===//
// XLA:GPU graph dispatch operation
//===----------------------------------------------------------------------===//

def XLAGPU_GraphDispatchOp : XLAGPU_Op<"graph.dispatch", [
    NoTerminator
  ]> {
  let summary = "dispatches XLA:GPU graph";

  let description = [{
    Graph dispatch region captures GPU graph builder function in a nested
    region. This is intermediate step before lowering to XLA:GPU runtime calls
    to create and update graph executables.

    Example: dispatching a GPU graph with a single kernel node (compiled fusion)

    ```mlir
      xla_gpu.graph.dispatch graph(%graph: !xla_gpu.graph) {
        %0 = call @xla_gpu.graph.create_kernel_node(%graph, ...)
      }
    ```
  }];

  let regions = (region SizedRegion<1>:$body);

  let extraClassDeclaration = [{
    mlir::TypedValue<GraphType> getGraph() {
      return mlir::cast<mlir::TypedValue<GraphType>>(getBody().getArgument(0));
    }
  }];

  let assemblyFormat = "custom<GraphDispatchRegion>($body) attr-dict";
}

#endif // XLA_GPU_OPS