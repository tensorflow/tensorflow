//===- LinalgLibraryOps.td - Linalg dialect library ops -*- tablegen ----*-===//
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
// This is the operation definition file for linear algebra operations that 
// correspond to underlying library calls (e.g. BLAS).
//
//===----------------------------------------------------------------------===//

include "mlir/Linalg/IR/LinalgBase.td"

#ifdef LINALG_LIBRARY_OPS
#else
#define LINALG_LIBRARY_OPS

class LinalgParametricNativeOpTrait<string prop, string parameters> :
  NativeOpTrait<"linalg::" # prop # parameters>
{}

class LinalgParametricIntNativeOpTrait<string prop, list<int> parameters> :
  LinalgParametricNativeOpTrait<
    prop,
    !strconcat("<",
               !cast<string>(!head(parameters)),
               !foldl("",
                      !tail(parameters),
                      sum,
                      param,
                      sum # "," # !cast<string>(param)),
               ">::Impl")>
{}

// The Linalg `NInputsAndOutputs` trait provides the API for ops that are known
// to have a specified number of inputs and outputs, all passed as operands.
// See Linalg/LinalgTraits.h for implementation details an usage.
class NInputsAndOutputs<int n_ins, int n_outs> :
  LinalgParametricIntNativeOpTrait<"NInputsAndOutputs", [n_ins, n_outs]>
{}

// The linalg `NLoopTypes` trait provides the API for ops that are known to have
// a specified number of parallel (n_par), reduction (n_red) and window (n_win)
// loops.
// See Linalg/LinalgTraits.h for implementation details an usage.
class NLoopTypes<int n_par, int n_red, int n_win> :
LinalgParametricIntNativeOpTrait<"NLoopTypes", [n_par, n_red, n_win]>
{}

// The linalg `ViewRanks` trait the API for ops that are known to have a
// specified list of view ranks.
// See Linalg/LinalgTraits.h for implementation details an usage.
class ViewRanks<list<int> ranks> :
LinalgParametricIntNativeOpTrait<"ViewRanks", ranks>
{}

// Base Tablegen class for Linalg ops.
class LinalgLibrary_Op<string mnemonic, list<OpTrait> props>
  : Op<Linalg_Dialect, mnemonic, props> {
  let parser = [{ return parseLinalgLibraryOp(parser, result); }];
  let printer = [{ printLinalgLibraryOp(p, *this); }];

  let extraClassDeclaration = [{
    static StringRef getLibraryCallName() {
      return "linalg_}] # mnemonic # [{";
    }
  }];
}

////////////////////////////////////////////////////////////////////////////////
// Concrete Linalg ops.
////////////////////////////////////////////////////////////////////////////////
def FillOp : LinalgLibrary_Op<"fill", [NInputsAndOutputs<0, 1>]> {
  let arguments = (ins View, AnyTypeOf<[AnyFloat, AnyInteger, AnyVector]>);
  let extraClassDeclaration = [{
    unsigned getNumParallelLoops() {
      auto *view = *(getOperands().begin());
      return view->getType().cast<ViewType>().getRank();
    }
    unsigned getNumReductionLoops() { return 0; }
    unsigned getNumWindowLoops() { return 0; }
    unsigned getNumLoops() { return getNumParallelLoops(); }
    Value *getValue() {
      return *(getOperands().begin() + getNumInputsAndOutputs());
    }
  }];
  let verifier = [{ return ::verify(*this); }];
}
def DotOp : LinalgLibrary_Op<"dot",
                            [NInputsAndOutputs<2, 1>,
                             NLoopTypes<0, 1, 0>,
                             ViewRanks<[1, 1, 0]>]> {
  let arguments = (ins View, View, View);
}
def MatvecOp : LinalgLibrary_Op<"matvec",
                                  [NInputsAndOutputs<2, 1>,
                                   NLoopTypes<1, 1, 0>,
                                   ViewRanks<[2, 1, 1]>]> {
  let arguments = (ins View, View, View);
}
def MatmulOp : LinalgLibrary_Op<"matmul",
                                  [NInputsAndOutputs<2, 1>,
                                   NLoopTypes<2, 1, 0>,
                                   ViewRanks<[2, 2, 2]>]> {
  let arguments = (ins View, View, View);
}

#endif // LINALG_LIBRARY_OPS
