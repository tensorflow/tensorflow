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

#ifdef LINALG_LIBRARY_OPS
#else
#define LINALG_LIBRARY_OPS

include "mlir/AffineOps/AffineOpsBase.td"
include "mlir/Linalg/IR/LinalgBase.td"

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

def ViewTraits : NativeOpTrait<"linalg::ViewTraits">;

// Base Tablegen class for Linalg ops.
// Linalg ops that correspond to library calls operate on linalg::View as their
// first operands. These may be optionally followed by non-view operands
// depending on the specific Linalg op.
class LinalgLibraryBase_Op<string mnemonic, list<OpTrait> props>
  : Op<Linalg_Dialect, mnemonic, !listconcat(props, [ViewTraits])> {
  let parser = [{ return parseLinalgLibraryOp(parser, result); }];
  let printer = [{ printLinalgLibraryOp(p, *this); }];
}

class LinalgLibrary_Op<string mnemonic, list<OpTrait> props>
  : LinalgLibraryBase_Op<mnemonic, props> {
  code libraryCallName = [{
    std::string getLibraryCallName() {
      return generateLibraryCallName(getOperation());
    }
  }];
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
  let extraClassDeclaration = libraryCallName # [{
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
  let extraClassDeclaration = libraryCallName # [{
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
  let extraClassDeclaration = libraryCallName;
}

def MatvecOp : LinalgLibrary_Op<"matvec",
                                  [NInputsAndOutputs<2, 1>,
                                   NLoopTypes<1, 1, 0>,
                                   ViewRanks<[2, 1, 1]>]> {
  let arguments = (ins View, View, View);
  let extraClassDeclaration = libraryCallName;
}

def MatmulOp : LinalgLibrary_Op<"matmul",
                                  [NInputsAndOutputs<2, 1>,
                                   NLoopTypes<2, 1, 0>,
                                   ViewRanks<[2, 2, 2]>]> {
  let arguments = (ins View, View, View);
  let extraClassDeclaration = libraryCallName;
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
  let extraClassDeclaration = libraryCallName # [{
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

def GenericOp : LinalgLibraryBase_Op<"generic", []> {
  let description = [{
    Generic Linalg op form where the key properties of the computation are
    specified as attributes. In pretty form, a linalg.generic op is written as:

      ```
        linalg.generic #trait_attribute %A, %B, %C {other-attributes} :
          !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
      ```

    Where #trait_attributes is an alias of a dictionary attribute containing:
      - doc [optional]: a documentation string
      - fun: a SymbolRefAttr that must resolve to an existing function symbol.
        To support inplace updates in a generic fashion, the signature of the
        function must be:
        ```
          fun([input views element types], [output views element types])
            -> ([output views element types])
        ```
      - indexing_maps: a list of AffineMapAttr, one AffineMapAttr per each input
        and output view. Such AffineMapAttr specifies the mapping between the
        loops and the indexing within each view.
      - library_call [optional]: a StringAttr containing the name of an
        external library function that the linalg.generic operation maps to.
        The external library is assumed to be dynamically linked and no strong
        compile-time guarantees are provided. In the absence of such a library
        call, linalg.generic will always lower to loops.
      - n_loops: a triple of I64Attr representing the number of enclosing
        [parallel, reduction, window] loops respectively.
      - n_views: a pair of I64Attr representing the number of input (readonly)
        and output (readwrite) views.

    Example:
    Defining a #matmul_trait attribute in MLIR can be done as follows:
      ```
        func @fma(%a: f32, %b: f32, %c: f32) -> f32 {
          %d = mulf %a, %b: f32
          %e = addf %c, %d: f32
          return %e: f32
        }
        #matmul_accesses = [
          (m, n, k) -> (m, k),
          (m, n, k) -> (k, n),
          (m, n, k) -> (m, n)
        ]
        #matmul_trait = {
          doc = "C(m, n) += A(m, k) * B(k, n)",
          fun = @fma,
          indexing_maps = #matmul_accesses,
          library_call = "linalg_matmul",
          n_views = [2, 1],
          n_loop_types = [2, 1, 0]
        }
      ```

    And can be reused in multiple places as:
      ```
        linalg.generic #matmul_trait %A, %B, %C [other-attributes] :
          !linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>
      ```

    This may lower to either:
      ```
        call @linalg_matmul(%A, %B, %C) :
          (!linalg.view<?x?xf32>, !linalg.view<?x?xf32>, !linalg.view<?x?xf32>)
          -> ()
      ```

    or IR resembling:
    ```
    loop.for %m = %c0 to %M step %c1 {
      loop.for %n = %c0 to %N step %c1 {
        loop.for %k = %c0 to %K step %c1 {
          %a = linalg.load %A[%m, %k] : !linalg.view<?x?xf32>
          %b = linalg.load %B[%k, %n] : !linalg.view<?x?xf32>
          %c = linalg.load %C[%m, %n] : !linalg.view<?x?xf32>
          %d = call @mac(%a, %b, %c) : (f32, f32, f32) -> (f32)
          linalg.store %d, %C[%m, %n] : !linalg.view<?x?x?xf32>
        }
      }
    }
    ```
  }];
  let arguments = (ins Variadic<View>:$views,
                   AffineMapArrayAttr:$indexing_maps,
                   I64ArrayAttr:$n_loop_types,
                   I64ArrayAttr:$n_views,
                   OptionalAttr<StrAttr>:$doc,
                   OptionalAttr<SymbolRefAttr>:$fun,
                   OptionalAttr<StrAttr>:$library_call);
  let regions = (region AnyRegion:$region);
  let extraClassDeclaration = [{
    SmallVector<StringRef, 8> linalgTraitAttrNames() {
      return SmallVector<StringRef, 8>{
        "doc", "fun", "indexing_maps", "library_call", "n_loop_types", "n_views"
      };
    }
    unsigned getNumInputs() {
      if (!getAttr("n_views") || n_views().getValue().size() != 2)
        return 0;
      auto val = n_views().getValue()[0].cast<IntegerAttr>().getValue();
      assert(val.getSExtValue() >= 0);
      return val.getZExtValue();
    }
    unsigned getNumOutputs() {
      if (!getAttr("n_views") || n_views().getValue().size() != 2)
        return 0;
      auto val = n_views().getValue()[1].cast<IntegerAttr>().getValue();
      assert(val.getSExtValue() >= 0);
      return val.getZExtValue();
    }
    unsigned getNumParallelLoops() {
      if (!getAttr("n_loop_types") || n_loop_types().getValue().size() != 3)
        return 0;
      auto val = n_loop_types().getValue()[0].cast<IntegerAttr>().getValue();
      assert(val.getSExtValue() >= 0);
      return val.getZExtValue();
    }
    unsigned getNumReductionLoops() {
      if (!getAttr("n_loop_types") || n_loop_types().getValue().size() != 3)
        return 0;
      auto val = n_loop_types().getValue()[1].cast<IntegerAttr>().getValue();
      assert(val.getSExtValue() >= 0);
      return val.getZExtValue();
    }
    unsigned getNumWindowLoops() {
      if (!getAttr("n_loop_types") || n_loop_types().getValue().size() != 3)
        return 0;
      auto val = n_loop_types().getValue()[2].cast<IntegerAttr>().getValue();
      assert(val.getSExtValue() >= 0);
      return val.getZExtValue();
    }
    unsigned getNumLoops() {
      return getNumParallelLoops() + getNumReductionLoops() +
        getNumWindowLoops();
    }
    FuncOp getFunction() {
      auto moduleOp = getParentOfType<ModuleOp>();
      return fun().hasValue() ?
        moduleOp.lookupSymbol<FuncOp>(fun().getValue()) : FuncOp();
    }
    StringRef getLibraryCallName() {
      return library_call().hasValue() ? library_call().getValue() : "";
    }
    AffineMap getIndexingMap(unsigned i) {
      assert(i < getNumInputsAndOutputs());
      return indexing_maps().getValue()[i].cast<AffineMapAttr>().getValue();
    }
    AffineMap getInputIndexingMap(unsigned i) {
      assert(i < getNumInputs());
      return indexing_maps().getValue()[i].cast<AffineMapAttr>().getValue();
    }
    AffineMap getOutputIndexingMap(unsigned i) {
      assert(i < getNumOutputs());
      return indexing_maps().getValue()[i + getNumInputs()]
          .cast<AffineMapAttr>().getValue();
    }
  }];
  let printer = [{ return ::print(p, *this); }];
  let verifier = [{ return ::verify(*this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
}
#endif // LINALG_LIBRARY_OPS
