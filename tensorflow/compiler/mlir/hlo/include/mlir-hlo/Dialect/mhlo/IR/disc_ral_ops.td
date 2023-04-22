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

// This is the operation definition file for DISC RAL ops.

#ifndef DISC_RAL_OPS
#define DISC_RAL_OPS

include "mlir/IR/OpBase.td"

def RAL_Dialect : Dialect {
  let name = "disc_ral";
  let cppNamespace = "::mlir::disc_ral";
}

class RAL_Op<string mnemonic, list<OpTrait> traits> :
    Op<RAL_Dialect, mnemonic, traits> {
  let verifier = [{ return Verify(*this); }];
}

def RAL_RalExecutionContextType : DialectType<RAL_Dialect,
    CPred<"$_self.isa<::mlir::disc_ral::RalExecutionContextType>()">,
          "context">,
    BuildableType<"$_builder.getType<::mlir::disc_ral::RalExecutionContextType>()"> {
  let description = [{
    RalExecutionContextType corresponds to C++ class RalExecution defined in
    disc.
  }];
}

def RAL_RecvInputOp: RAL_Op<"recv_input", []> {
  let summary = "receive input operator";
  let description = [{
    Fetch the specified input.

    In DISC, inputs are not passed as parameter directly. Instead, users need to
    explicitly fetch the individual input using this op.
  }];
  let arguments = (ins
    RAL_RalExecutionContextType:$ctx,
    Index:$input_idx
  );
  let results = (outs MemRefOf<[AnyType]>);
}

def RAL_SendOutputOp: RAL_Op<"send_output", []> {
  let summary = "send output operator";
  let description = [{
    Send the specified output.

    In DISC, outputs are not returned to the caller directly. Instead, users need to
    explicitly send the individual output using this op.
  }];
  let arguments = (ins
    RAL_RalExecutionContextType:$ctx,
    Index:$output_idx,
    MemRefOf<[AnyType]>:$result
  );
  let results = (outs);
}

def RAL_DispatchOp: RAL_Op<"dispatch", []> {
  let summary = "Dispatch operator";
  let description = [{
    A dispatch op invokes code external to disc. The `args` are passed to the
    external code, and the external code is expected to produce a result of the
    given type. The exact mechanism is backend-specific.

    `call_target_name` and `backend_config` can be arbitrary strings, but
    `call_target_name` should be short as it may be used in labels.
    `backend_config` can encode arbitrarily large amounts of information.
  }];
  let arguments = (ins
    RAL_RalExecutionContextType:$ctx,
    Variadic<AnyType>:$args,
    StrAttr:$call_target_name,
    DefaultValuedAttr<BoolAttr, "false">:$has_side_effect,
    DefaultValuedAttr<StrAttr, "">:$backend_config
  );
  let results = (outs Variadic<AnyType>);
}

#endif // DISC_RAL_OPS
