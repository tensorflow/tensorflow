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

#ifndef DEALLOCATION_OPS_TD_
#define DEALLOCATION_OPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def DeallocationDialect : Dialect {
  let name = "deallocation";

  let summary = "Operations for the deallocation dialect";
  let description = [{
    Ops for modelling owned/unowned memrefs using null pointers.
  }];
  let cppNamespace = "::mlir::deallocation";
  let useFoldAPI = kEmitFoldAdaptorFolder;
}

class DeallocationOp<string mnemonic, list<Trait> traits = []>
    : Op<DeallocationDialect, mnemonic, traits>;

def GetBufferOp : DeallocationOp<"get_buffer", [Pure]> {
  let summary = "extracts the data pointer as an index";

  let arguments = (ins AnyRankedOrUnrankedMemRef:$memref);
  let results = (outs Index:$result);

  let assemblyFormat = "attr-dict $memref `:` type($memref)";
}

def NullOp : DeallocationOp<"null", [Pure]> {
  let summary = "null pointer";

  let results = (outs AnyRankedOrUnrankedMemRef:$result);

  let assemblyFormat = "attr-dict `:` type($result)";
}

def RetainOp : DeallocationOp<"retain", [AttrSizedOperandSegments]> {
  let summary = "null-safe dealloc";

  let description = [{
    For each memref in `retained`, finds the alloc in `allocs` that it is
    derived from and returns it. If not found, returns `null`.

    Any allocs that are not in the result are deallocated.

    `allocs` may contain `null`s. Otherwise, all allocs must be distinct.
  }];

  let arguments = (ins Variadic<AnyRankedOrUnrankedMemRef>:$retained,
                       Variadic<AnyRankedOrUnrankedMemRef>:$allocs);
  let results = (outs Variadic<AnyRankedOrUnrankedMemRef>:$result_allocs);

  let assemblyFormat = [{
    `(` $retained `)` `of` `(` $allocs `)` attr-dict `:`
      functional-type(operands, results)
  }];

  let hasCanonicalizer = 1;
}

#endif  // DEALLOCATION_TD_
