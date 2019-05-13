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

// Whether a type is a BufferType.
def LinalgIsBufferTypePred : CPred<"$_self.isa<BufferType>()">;
def Buffer : Type<LinalgIsBufferTypePred, "buffer">;

// Whether a type is a ViewType.
def LinalgIsViewTypePred : CPred<"$_self.isa<ViewType>()">;
def View : Type<LinalgIsViewTypePred, "view">;

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
class LinalgOp<string mnemonic, list<OpTrait> props> :
Op<Linalg_Dialect, mnemonic, props> {
  let arguments = (ins Variadic<View>); // default variadic builder
  let parser = [{ return parseLinalgLibraryOp(parser, result); }];
  let printer = [{ printLinalgLibraryOp(p, *this); }];
}

def BufferSizeOp :
    Op<Linalg_Dialect, "buffer_size", [NoSideEffect]>,
    Arguments<(ins Buffer)>,
    Results<(outs Index)>
{
  let parser = [{ return parseBufferSizeOp(parser, result); }];
  let printer = [{ return printBufferSizeOp(p, *this); }];
}

def DimOp : Op<Linalg_Dialect, "dim", [NoSideEffect]>,
    Arguments<(ins View:$view, APIntAttr:$index)>,
    Results<(outs Index)> {
  let summary = "dimension index operation";
  let description = [{
    The "linalg.dim" operation takes a linalg.view and returns an
    "index". It requires a single integer attribute named "index". It
     returns the size of the specified dimension. For example:

      %1 = linalg.dim %0, 2 : view<?x?x?xf32>
  }];

  let parser = [{ return parseDimOp(parser, result); }];
  let printer = [{ return printDimOp(p, *this); }];
  let verifier = [{ return ::verify(*this); }];

  let builders = [OpBuilder<
    "Builder *builder, OperationState *result, Value *view," "unsigned index",
    [{
      result->addOperands(view);
      result->addAttribute(
        "index", builder->getIntegerAttr(builder->getIndexType(), index));
      result->types.push_back(builder->getIndexType());
    }]>];

  let extraClassDeclaration = [{
    unsigned getIndex() {
      return getAttrOfType<IntegerAttr>("index").getValue().getZExtValue();
    }
  }];
}

////////////////////////////////////////////////////////////////////////////////
// Concrete Linalg ops.
////////////////////////////////////////////////////////////////////////////////
def DotOp : LinalgOp<"dot", [NInputsAndOutputs<2, 1>,
                             NLoopTypes<0, 1, 0>,
                             ViewRanks<[1, 1, 0]>]> {}
def MatvecOp : LinalgOp<"matvec", [NInputsAndOutputs<2, 1>,
                                   NLoopTypes<1, 1, 0>,
                                   ViewRanks<[2, 1, 1]>]> {}
def MatmulOp : LinalgOp<"matmul", [NInputsAndOutputs<2, 1>,
                                   NLoopTypes<2, 1, 0>,
                                   ViewRanks<[2, 2, 2]>]> {}

#endif // LINALG_OPS