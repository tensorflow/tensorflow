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

#ifdef XLA_GPU_DIALECT
#else
#define XLA_GPU_DIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"

//===----------------------------------------------------------------------===//
// XLA GPU Dialect
//===----------------------------------------------------------------------===//

def XlaGpuDialect : Dialect {
  let name = "xla_gpu";

  let description = [{
    This dialect contains types required for lowering XLA:GPU programs to
    XLA:GPU runtime with all the library integrations. At run-time all MHLO
    attributes (e.g. `#mhlo.dot` on `mhlo.dot_general` operation) are passed
    around as reference-counted values (e.g. dot dimensions is just a struct).
  }];

  let cppNamespace = "::xla::gpu";
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// XLA GPU Types
//===----------------------------------------------------------------------===//

class XLA_GPU_Type<string name, string typeMnemonic> :
  TypeDef<XlaGpuDialect, name> {
  let mnemonic = typeMnemonic;
}

def ExecutionContextType : XLA_GPU_Type<"ExecutionContext",
                                        "execution_context"> {
  let summary = "XLA:GPU execution context";
}

def DotConfigType : XLA_GPU_Type<"DotConfig", "dot_config"> {
  let summary = "Config for dot operation";
}

def DotDimensionNumbersType : XLA_GPU_Type<"DotDimensionNumbers",
                                           "dot_dimension_numbers"> {
  let summary = "Dimension numbers for dot operation";
}

def DotPrecisionType : XLA_GPU_Type<"DotPrecision", "dot_precision"> {
  let summary = "Precision for dot operation";
}

def GraphType : XLA_GPU_Type<"Graph", "graph"> {
  let summary = "XLA:GPU graph";
}

def GraphNodeType : XLA_GPU_Type<"GraphNode", "graph.node"> {
  let summary = "XLA:GPU graph node";
}

def KernelType : XLA_GPU_Type<"Kernel", "kernel"> {
  let summary = "XLA:GPU device kernel";
}

def TraceType : XLA_GPU_Type<"Trace", "trace"> {
  let summary = "XLA:GPU trace annotation";
}

#endif // XLA_GPU_DIALECT
