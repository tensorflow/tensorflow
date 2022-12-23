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

//===- rt_dialect.td ------------------------------------------------------===//
//
// Dialect, types and attributes definitions for the RT dialect.
//
//===----------------------------------------------------------------------===//

#ifdef RT_DIALECT
#else
#define RT_DIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

include "tensorflow/compiler/xla/mlir/runtime/ir/rt_interfaces.td"

def RuntimeDialect : Dialect {
  let name = "rt";

  let description = [{
    The run-time dialect defines types and operations for the XLA executables to
    call into the XLA C++ runtime.
  }];

  let cppNamespace = "::xla::runtime";

  let useDefaultTypePrinterParser = 1;

  let useDefaultAttributePrinterParser = 1;

  let hasOperationAttrVerify = 1;
}

def RT_Ordinal : SignlessIntegerAttrBase<I32, "ordinal value">;

//===----------------------------------------------------------------------===//
// Types for integrating XLA executables with the runtime.
//===----------------------------------------------------------------------===//

class RT_Type<string name, string typeMnemonic> : TypeDef<RuntimeDialect,
                                                          name> {
  let mnemonic = typeMnemonic;
}

def ExecutionContextType : RT_Type<"ExecutionContext", "execution_context"> {
  let summary = "Execution Context type";
  let description = [{
    Handle to an instance of the runtime execution context.
  }];
}

def StatusType : RT_Type<"Status", "status"> {
  let summary = "Status type";
  let description = [{
    A status type returned from the XLA runtime API intrinsics.
  }];
}

def OpaqueValueType : RT_Type<"Opaque", "opaque"> {
  let summary = "Opaque type";
  let description = [{
    Opaque type for passing opaque values (at run time just `void*`) to and from
    compiled executables and custom calls.
  }];
}

//===----------------------------------------------------------------------===//
// Tracing Annotation Attributes
//===----------------------------------------------------------------------===//

def HloTraceAttr : AttrDef<RuntimeDialect, "HloTrace",
    [RT_TraceAnnotationAttrInterface]> {
  let mnemonic = "hlo_trace";
  let summary = "Trace execution of the HLO operation";

  let parameters = (ins
    StringRefParameter<"hlo_op">:$hlo_op
  );

  let assemblyFormat = "`<` $hlo_op `>`";
}

#endif // RT_DIALECT
