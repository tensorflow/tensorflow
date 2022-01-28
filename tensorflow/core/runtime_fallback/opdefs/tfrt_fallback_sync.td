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

#ifdef TFRT_FALLBACK_SYNC_OPS
#else
#define TFRT_FALLBACK_SYNC_OPS

include "tfrt/tfrt_op_base.td"
include "tfrt/core_runtime/opdefs/corert_base.td"
include "tfrt/core_runtime/opdefs/corert_traits.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "tfrt_fallback.td"

def FallbackSync_Dialect : Dialect {
  let name = "tfrt_fallback_sync";

  let description = [{
    The TFRT Fallback Sync Dialect.
  }];

  let cppNamespace = "::tfrt::fallback_sync";

  let emitAccessorPrefix = kEmitAccessorPrefix_Raw;
}

class FallbackSync_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<FallbackSync_Dialect, mnemonic, !listconcat(traits, [IsolatedFromAbove])> {
  // Each registered op in the fallback namespace needs to provide all of a
  // verifier, a printer and a parser.
  let verifier = [{ return tfrt::fallback_sync::verify(*this); }];
  let printer = [{ return tfrt::fallback_sync::print(p, *this); }];
  let parser = [{ return tfrt::fallback_sync::parse$cppClass(parser, result); }];
}

def ExecuteOp : FallbackSync_Op<"executeop",
    [NoSideEffect, CoreRT_TypedAttributeTrait]> {
  let summary = "The Fallback Sync ExecuteOp";
  let description = [{
    The ExecuteOp executes an operation on the specified device.

    Example:
      %res = tfrt_fallback_sync.executeop "some.op"(%arg) : 1

    Note that the trailing number indicates the number of results.
  }];

  let arguments = (ins
    Variadic<TensorType>:$operands,
    ArrayAttr:$op_attrs,
    StrAttr:$op_name
  );

  let results = (outs
    Variadic<TensorType>:$results
  );

  let extraClassDeclaration = [{
    void getOpAttrs(SmallVectorImpl<std::pair<StringRef, Attribute>>* op_attrs);
  }];

  let builders = [
    OpBuilder<(ins "ArrayRef<Type>":$results, "ValueRange":$operands,
      "ArrayRef<std::pair<StringRef, Attribute>>":$op_attrs,
      "StringRef":$op_name)>];
}

#endif
