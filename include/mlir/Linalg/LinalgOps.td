//===- LinalgOps.td - Linear algebra dialect ops -----------*- tablegen -*-===//
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
// This is the operation definition file for linear algebra operations.
//
//===----------------------------------------------------------------------===//

#ifdef LINALG_OPS
#else

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def Linalg_Dialect : Dialect {
  let name = "linalg";
}

// Whether a type is a ViewType.
def LinalgIsViewTypePred : CPred<"$_self.isa<ViewType>()">;
def View : Type<LinalgIsViewTypePred, "view">;

class ParametricNativeOpTrait<string prop, string parameters> :
  NativeOpTrait<prop # parameters>;

class ParametricIntNativeOpTrait<string prop, list<int> parameters> :
  ParametricNativeOpTrait<
    prop,
    !strconcat("<",
               !cast<string>(!head(parameters)),
               !foldl("",
                      !tail(parameters),
                      sum,
                      param,
                      sum # "," # !cast<string>(param)),
               ">::Impl")>;

// The Linalg `NInputsAndOutputs` trait provides the API for ops that are known
// to have a specified number of inputs and outputs, all passed as operands.
// See Linalg/LinalgTraits.h for implementation details an usage.
class NInputsAndOutputs<int n_ins, int n_outs> :
  ParametricIntNativeOpTrait<"NInputsAndOutputs", [n_ins, n_outs]>;

// The linalg `NLoopTypes` trait provides the API for ops that are known to have
// a specified number of parallel (n_par), reduction (n_red) and window (n_win)
// loops.
// See Linalg/LinalgTraits.h for implementation details an usage.
class NLoopTypes<int n_par, int n_red, int n_win> :
  ParametricIntNativeOpTrait<"NLoopTypes", [n_par, n_red, n_win]>;

// The linalg `ViewRanks` trait the API for ops that are known to have a
// specified list of view ranks.
// See Linalg/LinalgTraits.h for implementation details an usage.
class ViewRanks<list<int> ranks> :
  ParametricIntNativeOpTrait<"ViewRanks", ranks>;

// Base Tablegen class for Linalg ops.
class LinalgOp<string mnemonic, list<OpTrait> props> :
    Op<Linalg_Dialect, mnemonic, props> {
  // The default variadic builder.
  let arguments = (ins Variadic<View>);

  let parser = [{ return impl::parseLinalgLibraryOp(parser, result); }];

  let printer = [{ impl::printLinalgLibraryOp(p, *this); }];
}

////////////////////////////////////////////////////////////////////////////////
// Concrete Linalg ops.
////////////////////////////////////////////////////////////////////////////////
def DotOp : LinalgOp<"dot", [NInputsAndOutputs<2, 1>,
                             NLoopTypes<0, 1, 0>,
                             ViewRanks<[1, 1, 0]>]>;
def MatvecOp : LinalgOp<"matvec", [NInputsAndOutputs<2, 1>,
                                   NLoopTypes<1, 1, 0>,
                                   ViewRanks<[2, 1, 1]>]>;
def MatmulOp : LinalgOp<"matmul", [NInputsAndOutputs<2, 1>,
                                   NLoopTypes<2, 1, 0>,
                                   ViewRanks<[2, 2, 2]>]>;

#endif // LINALG_OPS
