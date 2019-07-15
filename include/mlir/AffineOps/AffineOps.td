//===- Ops.td - Affine operation definitions ---------------*- tablegen -*-===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// Defines MLIR affine operations.
//
//===----------------------------------------------------------------------===//

#ifdef AFFINE_OPS
#else
#define AFFINE_OPS

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def Affine_Dialect : Dialect {
  let name = "affine";
  let cppNamespace = "";
}

// Base class for Affine dialect ops.
class Affine_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Affine_Dialect, mnemonic, traits> {
  // For every affine op, there needs to be a:
  //   * void print(OpAsmPrinter *p, ${C++ class of Op} op)
  //   * LogicalResult verify(${C++ class of Op} op)
  //   * ParseResult parse${C++ class of Op}(OpAsmParser *parser,
  //                                         OperationState *result)
  // functions.
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}

def AffineTerminatorOp :
    Affine_Op<"terminator", [Terminator]> {
  let summary = "affine terminator operation";
  let description = [{
    Affine terminator is a special terminator operation for blocks inside affine
    loops and branches. It unconditionally transmits the control flow to the
    successor of the operation enclosing the region.

    This operation does _not_ have a custom syntax. However, affine control
    operations omit the terminator in their custom syntax for brevity.
  }];

  // No custom parsing/printing form.
  let parser = ?;
  let printer = ?;

  // Fully specified by traits.
  let verifier = ?;
}

#endif // AFFINE_OPS
