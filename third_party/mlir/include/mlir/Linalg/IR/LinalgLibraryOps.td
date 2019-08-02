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
// Linalg ops that correspond to library calls operate on linalg::View as their
// first operands. These may be optionally followed by non-view operands
// depending on the specific Linalg op.
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

def AffineMapAttr : Attr<
    CPred<"$_self.isa<AffineMapAttr>()">, "AffineMap attribute"> {
  let storageType = [{ AffineMapAttr }];
  let returnType = [{ AffineMap }];
  let constBuilderCall = "$_builder.getAffineMapAttr($0)";
}

////////////////////////////////////////////////////////////////////////////////
// Concrete Linalg ops.
////////////////////////////////////////////////////////////////////////////////
def CopyOp : LinalgLibrary_Op<"copy", [NInputsAndOutputs<1, 1>]> {
  let description = [{
    Copies the data in the input view into the output view.

    Usage:
      linalg.copy(%arg0, %arg1) : !linalg.view<?xf32>, !linalg.view<?xf32>

    One possible lowering to loop form is:
      %0 = linalg.dim %arg0, 0 : index
      loop.for %i0 = %c0 to %0 step %c1 {
        %1 = linalg.load %arg0[%i0] : !linalg.view<?xf32>
        linalg.store %1, %arg1[%i0] : !linalg.view<?xf32>
      }

    Optionally, can take `input_permutation` and `output_permutation` attributes
    to reorder the dimensions of the input and output views.

    Usage:
      linalg.copy(%arg0, %arg1) {inputPermutation : (i, j, k) -> (i, k, j),
                                 outputPermutation : (i, j, k) -> (k, j, i)} :
        !linalg.view<?x?x?xf32>, !linalg.view<?x?x?xf32>

    One possible lowering to loop form is:
      %0 = linalg.dim %arg0, 0
      %1 = linalg.dim %arg0, 1
      %2 = linalg.dim %arg0, 2
      loop.for %i0 = %c0 to %{{.*}} step %c1 {
        loop.for %i1 = %c0 to %{{.*}} step %c1 {
          loop.for %i2 = %c0 to %{{.*}} step %c1 {
            %3 = linalg.load %arg0[%i0, %i2, %i1] : !linalg.view<?x?x?xf32>
            linalg.store %3, %arg1[%i2, %i1, %i0] : !linalg.view<?x?x?xf32>

    The views are expected to be compatible for correctness but this is not
    enforced at the moment.
  }];
  let arguments = (ins
    View,
    View,
    OptionalAttr<AffineMapAttr>:$inputPermutation,
    OptionalAttr<AffineMapAttr>:$outputPermutation);
  // TODO(ntv) this should go away once the usage of OptionalAttr triggers
  // emission of builders with default arguments left unspecified.
  let builders = [OpBuilder<
    "Builder *builder, OperationState *result, Value *input, Value *output", [{
    return build(
      builder, result, input, output, AffineMapAttr(), AffineMapAttr());
  }]>];
  let extraClassDeclaration = [{
    unsigned getNumParallelLoops() {
      auto *view = *(getOperands().begin());
      return view->getType().cast<ViewType>().getRank();
    }
    unsigned getNumReductionLoops() { return 0; }
    unsigned getNumWindowLoops() { return 0; }
  }];
  let verifier = [{ return ::verify(*this); }];
}

def FillOp : LinalgLibrary_Op<"fill", [NInputsAndOutputs<0, 1>]> {
  let arguments = (ins View, AnyTypeOf<[AnyFloat, AnyInteger, AnyVector]>);
  let extraClassDeclaration = [{
    unsigned getNumParallelLoops() {
      auto *view = *(getOperands().begin());
      return view->getType().cast<ViewType>().getRank();
    }
    unsigned getNumReductionLoops() { return 0; }
    unsigned getNumWindowLoops() { return 0; }
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

def ConvOp : LinalgLibrary_Op<"conv", [NInputsAndOutputs<2, 1>]> {
  let description = [{
    Generic n-D convolution as described in the TF documentation:
    https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/nn/convolution

    ```
      output[b, x[0], ..., x[N-1], k] =
      sum_{z[0], ..., z[N-1], q}
          filter[z[0], ..., z[N-1], q, k] *
          padded_input[b,
                       x[0] * strides[0] + dilation_rate[0] * z[0],
                       ...,
                       x[N-1] * strides[N-1] + dilation_rate[N-1] * z[N-1],
                       q]
    ```
  }];
  // TODO(ntv) padding.
  // Following the TF source of truth above, strides and dilations are integer
  // attributes of the same rank as the number of window dimensions.
  let arguments = (ins View:$filter, View:$input, View:$output,
                   OptionalAttr<I64ArrayAttr>:$strides,
                   OptionalAttr<I64ArrayAttr>:$dilations);
  let extraClassDeclaration = [{
    // TODO(ntv) extend to support more than 1 dimensions and potentially
    // grouping too.
    unsigned getNumBatchDimensions() { return 1; }
    unsigned getNumInputFeatureDimensions() { return 1; }
    unsigned getNumOutputFeatureDimensions() { return 1; }

    // Outer parallel loops are always the number of output dimensions; i.e.
    // [ b, xs, q] in the TF notation above.
    unsigned getNumParallelLoops() { return getOutputViewType(0).getRank(); }

    // Window loops are a special kind of reduction that is neither tiled or
    // parallelized across; i.e. [zs] in the TF notation above whose number
    // match `xs` (i.e. 1 window loop per "image" dimension).
    unsigned getNumWindowLoops() {
      return getNumParallelLoops() - getNumBatchDimensions() -
             getNumInputFeatureDimensions(); }

    // Reduction loops are exactly the non-parallel, non-window loops (i.e. `q`)
    // We distinguish between reduction loops and convolution window loops for
    // now. That distinction may disappear in the future.
    unsigned getNumReductionLoops() { return getNumInputFeatureDimensions(); }

    int64_t getStride(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!strides().hasValue()) return 1;
      return strides()->getValue()[i]
        .cast<IntegerAttr>().getValue().getSExtValue();
    }

    int64_t getDilation(unsigned i) {
      assert(i < getNumWindowLoops());
      if (!dilations().hasValue()) return 1;
      return dilations()->getValue()[i]
        .cast<IntegerAttr>().getValue().getSExtValue();
    }
  }];
  let verifier = [{ return ::verify(*this); }];
}

#endif // LINALG_LIBRARY_OPS
