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

#ifndef MEMREF_DISC_OPS
#define MEMREF_DISC_OPS

include "mlir-hlo/Dialect/disc/IR/memref_disc_base.td"
include "mlir/IR/OpBase.td"

class MemRefDisc_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<MemRefDisc_Dialect, mnemonic, traits> {
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

//===----------------------------------------------------------------------===//
// LoadLinIdxOp 
//===----------------------------------------------------------------------===//
def LoadLinIdxOp : MemRefDisc_Op<"load_lin_idx",
     [AttrSizedOperandSegments,
      TypesMatchWith<"result type matches element type of 'memref'",
                     "memref", "result",
                     "$_self.cast<MemRefType>().getElementType()">]> {

  let summary = "load_lin_idx operation";
  let description = [{
    An variant of the "load" op with optional linear index or multidim
    index. Either linear or multidim index can be empty but at least one
    of them must be provided, unless for load from scalar tensor.

    examples:
      %3 = load %0[][%1, %1] : memref<4x4xi32>
      %3 = load %0[%2][%1, %1] : memref<4x4xi32>
      %3 = load %0[%2][] : memref<4x4xi32>
  }];

  let arguments = (ins AnyMemRef:$memref,
                       Variadic<Index>:$linear_index,
                       Variadic<Index>:$indices);
  let results = (outs AnyType:$result);

  let builders = [
    OpBuilder<(ins "Value":$memref,
      "ValueRange":$linear_index,
      "ValueRange":$indices), [{
       auto memrefType = memref.getType().cast<MemRefType>();
       $_state.addOperands(memref);
       $_state.addOperands(linear_index);
       $_state.addOperands(indices);
       $_state.types.push_back(memrefType.getElementType());
    }]>];

  let extraClassDeclaration = [{
    Value getMemRef() { return getOperand(0); }
    void setMemRef(Value value) { setOperand(0, value); }
    MemRefType getMemRefType() {
      return getMemRef().getType().cast<MemRefType>();
    }

    bool hasLinearIndex() { return llvm::size(linear_index()) > 0; }
    int64_t getNumIndices() { return llvm::size(indices()); }
    Value getLinearIndex() { return linear_index()[0]; }
    operand_range getIndices() { return indices(); }
  }];

  let assemblyFormat = "$memref `[` $linear_index `]``[` $indices `]` attr-dict `:` type($memref)";
}

//===----------------------------------------------------------------------===//
// StoreLinIdxOp 
//===----------------------------------------------------------------------===//
def StoreLinIdxOp : MemRefDisc_Op<"store_lin_idx",
     [AttrSizedOperandSegments,
      TypesMatchWith<"type of 'value' matches element type of 'memref'",
                     "memref", "value",
                     "$_self.cast<MemRefType>().getElementType()">]> {
  let summary = "store operation with potential linear index";
  let description = [{
    An variant of the "store" op with optional linear index or multidim
    index. Either linear or multidim index can be empty but at least one
    of them must be provided, unless for store to scalar tensor.

    examples:
      store %v, %A[][%i, %j] : memref<4x128xf32>
      store %v, %A[%m][%i, %j] : memref<4x128xf32>
      store %v, %A[%m][] : memref<4x128xf32>
  }];

  let arguments = (ins AnyType:$value,
                       AnyMemRef:$memref,
                       Variadic<Index>:$linear_index,
                       Variadic<Index>:$indices);

  let builders = [
    OpBuilder<(ins "Value":$valueToStore, "Value":$memref), [{
      $_state.addOperands(valueToStore);
      $_state.addOperands(memref);
  }]>];

  let extraClassDeclaration = [{
    Value getValueToStore() { return getOperand(0); }
    Value getMemRef() { return getOperand(1); }
    void setMemRef(Value value) { setOperand(1, value); }
    MemRefType getMemRefType() {
      return getMemRef().getType().cast<MemRefType>();
    }

    bool hasLinearIndex() { return llvm::size(linear_index()) > 0; }
    int64_t getNumIndices() { return llvm::size(indices()); }
    Value getLinearIndex() { return linear_index()[0]; }
    operand_range getIndices() { return indices(); }
  }];

  let verifier = [{ return success(); }];
  let assemblyFormat = [{
    $value `,` $memref `[` $linear_index `]``[` $indices `]` attr-dict `:` type($memref)
  }];
}

#endif // MEMREF_DISC_OPS
