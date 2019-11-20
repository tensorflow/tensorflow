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

#ifndef LINALG_LIBRARY_OPS
#define LINALG_LIBRARY_OPS

#ifndef AFFINE_OPS_BASE
include "mlir/Dialect/AffineOps/AffineOpsBase.td"
#endif // AFFINE_OPS_BASE

#ifndef LINALG_BASE
include "mlir/Dialect/Linalg/IR/LinalgBase.td"
#endif // LINALG_BASE

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

def ViewTraits : NativeOpTrait<"linalg::ViewTraits">;

// The linalg 'LinalgLibraryInterface' provides access to the 'LinalgOp'
// interface.
def LinalgLibraryInterface : OpInterface<"LinalgOp"> {
  let methods = [
    InterfaceMethod<
      "Query the number of inputs from the current operation.",
      "unsigned", "getNumInputs"
    >,
    InterfaceMethod<
      "Query the number of outputs from the current operation.",
      "unsigned", "getNumOutputs"
    >,
    InterfaceMethod<
      "Query the number of inputs and outputs from the current operation.",
      "unsigned", "getNumInputsAndOutputs"
    >,
    InterfaceMethod<
      "Query the input operands from the current operation.",
      "Operation::operand_range", "getInputs"
    >,
    InterfaceMethod<
      "Query the output operands from the current operation.",
      "Operation::operand_range", "getOutputs"
    >,
    InterfaceMethod<
      "Query the input and output operands from the current operation.",
      "Operation::operand_range", "getInputsAndOutputs"
    >,
    InterfaceMethod<
      "Query the number of parallel loops within the current operation.",
      "unsigned", "getNumParallelLoops"
    >,
    InterfaceMethod<
      "Query the number of reduction loops within the current operation.",
      "unsigned", "getNumReductionLoops"
    >,
    InterfaceMethod<
      "Query the number of window loops within the current operation.",
      "unsigned", "getNumWindowLoops"
    >,
    InterfaceMethod<
      "Query the number of loops within the current operation.",
      "unsigned", "getNumLoops", (ins), [{
      return op.getNumParallelLoops() + op.getNumReductionLoops() +
             op.getNumWindowLoops();
    }]>,
    InterfaceMethod<"Query the input view at the given index.",
      "Value *", "getInput", (ins "unsigned":$i)
    >,
    InterfaceMethod<"Query the output view at the given index.",
      "Value *", "getOutput", (ins "unsigned":$i)
    >,
    InterfaceMethod<[{
        Query the index of the given input value, or `None` if the value is not
        an input.
      }],
      "llvm::Optional<unsigned>", "getIndexOfInput", (ins "Value *":$view)
    >,
    InterfaceMethod<[{
        Query the index of the given view value, or `None` if the value is not
        an view.
      }],
      "llvm::Optional<unsigned>", "getIndexOfOutput", (ins "Value *":$view)
    >,
    InterfaceMethod<[{
        Query the type of the input view at the given index.
      }], "MemRefType", "getInputViewType", (ins "unsigned":$i)>,
    InterfaceMethod<[{
        Query the type of the output view at the given index.
      }], "MemRefType", "getOutputViewType", (ins "unsigned":$i)>,

    StaticInterfaceMethod<[{
        Create an operation of the current type with the given location,
        operands, and attributes.
      }],
      "Operation *", "create",
      (ins "OpBuilder &":$builder, "Location":$loc,
           "ArrayRef<Value *>":$operands,
           "ArrayRef<NamedAttribute>":$attributes), [{
        return builder.create<ConcreteOp>(loc, ArrayRef<Type>{}, operands,
                                          attributes);
      }]
    >,

    /// Clone an operation with the given location and operands. This is used to
    /// abstract away the optional underlying region creation.
    InterfaceMethod<[{
        Clone the current operation with the given location and operands. This
        is used to abstract away the optional underlying region creation.
      }],
      "Operation *", "clone",
      (ins "OpBuilder &":$b, "Location":$loc, "ArrayRef<Value *>":$operands), [{
        BlockAndValueMapping map;
        unsigned numRegions = op.getOperation()->getNumRegions();
        Operation *res = create(b, loc, operands, op.getAttrs());
        assert(res->getNumRegions() == numRegions && "inconsistent # regions");
        for (unsigned ridx = 0; ridx < numRegions; ++ridx)
          op.getOperation()->getRegion(ridx).cloneInto(
            &res->getRegion(ridx), map);
        return res;
      }]
    >
  ];
}

// Base Tablegen class for Linalg ops.
// Linalg ops that correspond to library calls operate on linalg::View as their
// first operands. These may be optionally followed by non-view operands
// depending on the specific Linalg op.
class LinalgLibraryBase_Op<string mnemonic, list<OpTrait> props>
  : Op<Linalg_Dialect, mnemonic,
       !listconcat(props, [ViewTraits, LinalgLibraryInterface])> {
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
      ```mlir
      linalg.copy(%arg0, %arg1) : memref<?xf32, stride_specification>,
                                  memref<?xf32, stride_specification>
      ```

    One possible lowering to loop form is:
      ```mlir
      %0 = linalg.dim %arg0, 0 : index
      loop.for %i0 = %c0 to %0 step %c1 {
        %1 = linalg.load %arg0[%i0] : memref<?xf32, stride_specification>
        linalg.store %1, %arg1[%i0] : memref<?xf32, stride_specification>
      }
      ```

    Optionally, can take `input_permutation` and `output_permutation` attributes
    to reorder the dimensions of the input and output views.

    Usage:
      ```mlir
      linalg.copy(%arg0, %arg1) {inputPermutation : (i, j, k) -> (i, k, j),
                                 outputPermutation : (i, j, k) -> (k, j, i)} :
        memref<?x?x?xf32, stride_specification>,
        memref<?x?x?xf32, stride_specification>
     ```

    One possible lowering to loop form is:
      ```mlir
      %0 = linalg.dim %arg0, 0
      %1 = linalg.dim %arg0, 1
      %2 = linalg.dim %arg0, 2
      loop.for %i0 = %c0 to %{{.*}} step %c1 {
        loop.for %i1 = %c0 to %{{.*}} step %c1 {
          loop.for %i2 = %c0 to %{{.*}} step %c1 {
            %3 = linalg.load %arg0[%i0, %i2, %i1] :
                    memref<?x?x?xf32, stride_specification>
            linalg.store %3, %arg1[%i2, %i1, %i0] :
                    memref<?x?x?xf32, stride_specification>
      ```

    The views are expected to be compatible for correctness but this is not
    enforced at the moment.
  }];
  let arguments = (ins
    AnyStridedMemRef:$input,
    AnyStridedMemRef:$output,
    OptionalAttr<AffineMapAttr>:$inputPermutation,
    OptionalAttr<AffineMapAttr>:$outputPermutation);
  // TODO(ntv) this should go away once the usage of OptionalAttr triggers
  // emission of builders with default arguments left unspecified.
  let builders = [OpBuilder<
    "Builder *builder, OperationState &result, Value *input, Value *output", [{
    return build(
      builder, result, input, output, AffineMapAttr(), AffineMapAttr());
  }]>];
  let extraClassDeclaration = libraryCallName # [{
    unsigned getNumParallelLoops() {
      auto *view = *(getOperands().begin());
      return view->getType().cast<MemRefType>().getRank();
    }
    unsigned getNumReductionLoops() { return 0; }
    unsigned getNumWindowLoops() { return 0; }
  }];
  let verifier = [{ return ::verify(*this); }];
}

def FillOp : LinalgLibrary_Op<"fill", [NInputsAndOutputs<0, 1>]> {
  let arguments = (ins AnyStridedMemRef,
                   AnyTypeOf<[AnyFloat, AnyInteger, AnyVector]>);
  let extraClassDeclaration = libraryCallName # [{
    unsigned getNumParallelLoops() {
      auto *view = *(getOperands().begin());
      return view->getType().cast<MemRefType>().getRank();
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
                             NLoopTypes<0, 1, 0>]> {
  let arguments = (ins AnyStridedMemRefOfRank<1>,
                       AnyStridedMemRefOfRank<1>,
                       AnyStridedMemRefOfRank<0>);
  let extraClassDeclaration = libraryCallName;
}

def MatvecOp : LinalgLibrary_Op<"matvec",
                                  [NInputsAndOutputs<2, 1>,
                                   NLoopTypes<1, 1, 0>]> {
  let arguments = (ins AnyStridedMemRefOfRank<2>,
                       AnyStridedMemRefOfRank<1>,
                       AnyStridedMemRefOfRank<1>);
  let extraClassDeclaration = libraryCallName;
}

def MatmulOp : LinalgLibrary_Op<"matmul",
                                  [NInputsAndOutputs<2, 1>,
                                   NLoopTypes<2, 1, 0>]> {
  let arguments = (ins AnyStridedMemRefOfRank<2>,
                       AnyStridedMemRefOfRank<2>,
                       AnyStridedMemRefOfRank<2>);
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
  let arguments = (ins AnyStridedMemRef:$filter, AnyStridedMemRef:$input,
                   AnyStridedMemRef:$output,
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

class GenericOpBase<string mnemonic> : LinalgLibraryBase_Op<mnemonic, []> {
  let arguments = (ins Variadic<AnyStridedMemRef>:$views,
                   AffineMapArrayAttr:$indexing_maps,
                   I64ArrayAttr:$n_loop_types,
                   I64ArrayAttr:$n_views,
                   OptionalAttr<StrAttr>:$doc,
                   OptionalAttr<FlatSymbolRefAttr>:$fun,
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
  let parser = [{ return ::parseGenericOp(parser, result); }];
}

def GenericOp : GenericOpBase<"generic"> {
  let description = [{
    Generic Linalg op form where the key properties of the computation are
    specified as attributes. In pretty form, a linalg.generic op is written as:

      ```mlir
        linalg.generic #trait_attribute %A, %B, %C {other-attributes} :
          memref<?x?xf32, stride_specification>,
          memref<?x?xf32, stride_specification>,
          memref<?x?xf32, stride_specification>
      ```

    Where #trait_attributes is an alias of a dictionary attribute containing:
      - doc [optional]: a documentation string
      - fun: a FlatSymbolRefAttr that must resolve to an existing function symbol.
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
      ```mlir
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
      ```mlir
        linalg.generic #matmul_trait %A, %B, %C [other-attributes] :
          memref<?x?xf32, stride_specification>,
          memref<?x?xf32, stride_specification>,
          memref<?x?xf32, stride_specification>
      ```

    This may lower to either:
      ```mlir
        call @linalg_matmul(%A, %B, %C) :
          (memref<?x?xf32, stride_specification>,
           memref<?x?xf32, stride_specification>,
           memref<?x?xf32, stride_specification>)
          -> ()
      ```

    or IR resembling:
    ```mlir
    loop.for %m = %c0 to %M step %c1 {
      loop.for %n = %c0 to %N step %c1 {
        loop.for %k = %c0 to %K step %c1 {
          %a = linalg.load %A[%m, %k] : memref<?x?xf32, stride_specification>
          %b = linalg.load %B[%k, %n] : memref<?x?xf32, stride_specification>
          %c = linalg.load %C[%m, %n] : memref<?x?xf32, stride_specification>
          %d = call @func_of_elements(%a, %b, %c)
                 : (f32, f32, f32) -> (f32)
          linalg.store %d, %C[%m, %n] : memref<?x?x?xf32, stride_specification>
        }
      }
    }
    ```
  }];
  let verifier = [{ return ::verify(*this); }];
}

def IndexedGenericOp : GenericOpBase<"indexed_generic"> {
  let description = [{
    Indexed Generic Linalg op form where the key properties of the computation
    are specified as attributes. In pretty form, a linalg.indexed_generic op is
    written as:

      ```mlir
        linalg.indexed_generic #trait_attribute %A, %B, %C {other-attributes} :
          memref<?x?xf32, stride_specification>,
          memref<?x?xf32, stride_specification>,
          memref<?x?xf32, stride_specification>
      ```

    Where #trait_attributes is an alias of a dictionary attribute containing:
      - doc [optional]: a documentation string
      - fun: a FlatSymbolRefAttr that must resolve to an existing function symbol.
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
        external library function that the linalg.indexed_generic operation
        maps to.  The external library is assumed to be dynamically linked and
        no strong compile-time guarantees are provided. In the absence of such
        a library call, linalg.indexed_generic will always lower to loops.
      - n_loops: a triple of I64Attr representing the number of enclosing
        [parallel, reduction, window] loops respectively.
      - n_views: a pair of I64Attr representing the number of input (readonly)
        and output (readwrite) views.

    Example:
    Defining a #matmul_trait attribute in MLIR can be done as follows:
      ```mlir
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
      ```mlir
        linalg.indexed_generic #matmul_trait %A, %B, %C [other-attributes] :
          memref<?x?xf32, stride_specification>,
          memref<?x?xf32, stride_specification>,
          memref<?x?xf32, stride_specification>
      ```

    This may lower to either:
      ```mlir
        call @linalg_matmul(%A, %B, %C) :
          (memref<?x?xf32, stride_specification>,
           memref<?x?xf32, stride_specification>,
           memref<?x?xf32, stride_specification>)
          -> ()
      ```

    or IR resembling:
    ```mlir
    loop.for %m = %c0 to %M step %c1 {
      loop.for %n = %c0 to %N step %c1 {
        loop.for %k = %c0 to %K step %c1 {
          %a = linalg.load %A[%m, %k] : memref<?x?xf32, stride_specification>
          %b = linalg.load %B[%k, %n] : memref<?x?xf32, stride_specification>
          %c = linalg.load %C[%m, %n] : memref<?x?xf32, stride_specification>
          %d = call @func_of_elements_and_indices(%m, %n, %k, %a, %b, %c)
                 : (index, index, index, f32, f32, f32) -> (f32)
          linalg.store %d, %C[%m, %n] : memref<?x?x?xf32, stride_specification>
        }
      }
    }
    ```
  }];
  let verifier = [{ return ::verify(*this); }];
}

#endif // LINALG_LIBRARY_OPS
