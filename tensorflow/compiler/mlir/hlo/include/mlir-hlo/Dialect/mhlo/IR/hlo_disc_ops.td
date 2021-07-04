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

// This is the operation definition file for MHLO DISC ops.

#ifndef MHLO_DISC_OPS
#define MHLO_DISC_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir-hlo/Dialect/mhlo/IR/hlo_ops_base.td"

def HLODISC_Dialect : Dialect {
  let name = "mhlo_disc";
  let cppNamespace = "::mlir::mhlo_disc";
}

class HLODISC_Op<string mnemonic, list<OpTrait> traits> :
    Op<HLODISC_Dialect, mnemonic, traits> {
  let verifier = [{ return Verify(*this); }];
}

def HLODISC_H2DOp: HLODISC_Op<"h2d", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "H2D operator";
  let description = [{
    Copy `operand` from host to device.
  }];
  let arguments = (ins HLO_Tensor);
  let results = (outs HLO_Tensor);
}

def HLODISC_D2HOp: HLODISC_Op<"d2h", [NoSideEffect, SameOperandsAndResultType]> {
  let summary = "D2H operator";
  let description = [{
    Copy `operand` from device to host.
  }];
  let arguments = (ins HLO_Tensor);
  let results = (outs HLO_Tensor);
}

#endif // MHLO_DISC_OPS
