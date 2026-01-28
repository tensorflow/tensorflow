/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_CODEGEN_XTILE_IR_XTILE_OPS
#define XLA_CODEGEN_XTILE_IR_XTILE_OPS
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/Interfaces.td"
include "xla/codegen/xtile/ir/xtile_dialect.td"
include "xla/codegen/xtile/ir/xtile_attrs.td"

class XTile_Op<string mnemonic, list<Trait> traits = []> :
      Op<XTileDialect, mnemonic, traits> {
}

def TiledBufferInterface : OpInterface<"TiledBufferInterface",
    [DeclareOpInterfaceMethods<BufferizableOpInterface,
      ["bufferizesToMemoryRead",
       "bufferizesToMemoryWrite",
       "bufferizesToAllocation",
       "getAliasingValues",
       "getAliasingOpOperands",
       "isWritable",
       "bufferize"]>
    ]> {
  let description = [{

  }];

  let cppNamespace = "::xla::xtile";

  let extraClassDeclaration = [{
    // Returns the set of unit dimensions that are removed from the tile
    // dimensions.
    llvm::SmallDenseSet<unsigned> getReducedDimensions();
  }];

  let extraTraitClassDeclaration = [{
    llvm::SmallDenseSet<unsigned> getReducedDimensions() {
      return static_cast<TiledBufferInterface>($_op).getReducedDimensions();
    }
  }];

  let methods = [
    InterfaceMethod<"Get the memref value of the buffer.",
                    "mlir::TypedValue<mlir::MemRefType>", "getBuffer">,
    InterfaceMethod<"Get the tensor value of the tile.",
                    "mlir::TypedValue<mlir::RankedTensorType>", "getTile">,
    InterfaceMethod<"Get the offsets into the buffer.",
                    "mlir::ValueRange", "getOffsets">,
    InterfaceMethod<"Get the full unreduced tile shape.",
                    "llvm::ArrayRef<int64_t>", "getFullTileShape">,
    InterfaceMethod<"Get the stride lengths of the buffer.",
                    "llvm::ArrayRef<int64_t>", "getStrides">,
  ];
}


// Define your custom entry operation
def EntryFuncOp : XTile_Op<"entry_func", [
    Symbol,
    IsolatedFromAbove,
    AutomaticAllocationScope,
    FunctionOpInterface]>
{
  let summary = "My custom entry function operation";

  let description = [{
    This operation defines a custom entry function that is the starting
    point for execution. It has a single-block region, takes a
    variadic list of memrefs and exactly one tile id index arguments,
    it has no return values.
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<XTile_TilingInfoAttr>:$tile_info,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       // The number of arguments that are opaque to tiling
                       // infrastructure and hence do not correspond to a memref
                       // argument.
                       DefaultValuedAttr<I32Attr, "0">:$num_opaque_args
  );

  // The entry function has no return values.
  let results = (outs);

  let regions = (region SizedRegion<1>:$body);

  let builders = [OpBuilder<(ins
    "mlir::StringRef":$name, "mlir::ArrayRef<mlir::Type>":$memref_arg_types,
    CArg<"mlir::ArrayRef<mlir::NamedAttribute>", "{}">:$attrs,
    CArg<"mlir::ArrayRef<mlir::DictionaryAttr>", "{}">:$memref_arg_attrs)
  >];

  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // Helper Methods
    //===------------------------------------------------------------------===//
    mlir::ValueRange getBufferArgs() {
      // +1 for the tile id argument.
      return getBody().getArguments().drop_back(getNumOpaqueArgs() + 1);
    }

    mlir::ValueRange getOpaqueArgs() {
      return getBody().getArguments().take_back(getNumOpaqueArgs() + 1)
          .drop_back();
    }

    mlir::Value getTileId() {
      return getBody().getArguments().back();
    }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable.
    mlir::Region *getCallableRegion() { return &getBody(); }

    /// Returns the argument types of this function.
    mlir::ArrayRef<mlir::Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    mlir::ArrayRef<mlir::Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def EntryFuncReturnOp : XTile_Op<"return", [Pure, HasParent<"EntryFuncOp">,
                                ReturnLike, Terminator]> {
  let summary = "Terminates the entry function";

  let description = [{
    This operation terminates the entry function block. It has no operands
    and produces no results.
  }];

  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = "attr-dict";
}

def ExtractTileOp : XTile_Op<"extract", [TiledBufferInterface]> {
  let summary = "Extract a tile from a memref.";
  let description = [{
  }];

  let arguments = (ins
    AnyMemRef:$source,
    Variadic<Index>:$offsets,
    DenseI64ArrayAttr:$full_tile_shape,
    DenseI64ArrayAttr:$strides
  );


  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $source `[` $offsets `]` $full_tile_shape $strides
    `:` type($source) `->` type($result) attr-dict
  }];

  let extraClassDeclaration = [{
    mlir::TypedValue<mlir::MemRefType> getBuffer();
    mlir::TypedValue<mlir::RankedTensorType> getTile();
  }];

  let hasVerifier = 1;
}

def InsertTileOp : XTile_Op<"insert", [TiledBufferInterface]> {
  let summary = "Insert a tile into a memref.";
  let description = [{
  }];

  let arguments = (ins
    AnyRankedTensor:$source,
    AnyMemRef:$destination,
    Variadic<Index>:$offsets,
    DenseI64ArrayAttr:$full_tile_shape,
    DenseI64ArrayAttr:$strides
  );

  let results = (outs);

  let assemblyFormat = [{
    $source `into` $destination `[` $offsets `]` $full_tile_shape $strides
    `:` type($source) `->` type($destination) attr-dict
  }];

  let extraClassDeclaration = [{
    mlir::TypedValue<mlir::MemRefType> getBuffer();
    mlir::TypedValue<mlir::RankedTensorType> getTile();
  }];

  let hasVerifier = 1;
}

def MaskOp : XTile_Op<"mask",
                      [Pure,
                       AllTypesMatch<["source", "result"]>,
                       TypesMatchWith<"mask type matches result element type",
                          "result", "value", ElementType<"_self">.result>
                      ]> {
  let summary = "Mask the values of a tensor.";

  let description = [{
    Masks out the values of the input tensor that are outside the range of the
    given mask upper bound. Masked values are set to the provided value.
  }];


  let arguments = (ins
    AnyRankedTensor:$source,
    DenseI64ArrayAttr:$bounds,
    AnyType:$value
  );

  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $source `bounds` $bounds `,` $value `:` type($result) attr-dict
  }];

  let extraClassDeclaration = [{
    // Get the dimensions where the mask bound is smaller than the dimension
    // size. The returned array is sorted in increasing order.
    llvm::SmallVector<int64_t> getMaskedDimensions();
  }];

  let hasVerifier = 1;
  let hasFolder = 1;
}


//
// DotScaled Op
//
// TODO(basioli): This op was copied from the triton dialect. If we want to use 
// it more we should probably consider documenting it properly, and including 
// more checks (e.x. similar to the Triton DotOpInterface).
def DotScaledOp : XTile_Op<"dot_scaled", [Pure, AttrSizedOperandSegments]> {
    let summary = "dot_scaled";

    let description = [{
        $result = matrix_multiply(scale($lhs, $lhs_scale), scale($rhs, $rhs_scale)).
        Where scale(x, s) is a function that applies the scale per block following microscaling spec.
    }];

    let arguments = (
      ins
      // inputs are floats if we have a type for them, otherwise (fp4),
      // they are packed in pairs in an I8Tensor
      RankedTensorOf<[AnyFloat,I8]>:$lhs,
      RankedTensorOf<[AnyFloat,I8]>:$rhs,
      Optional<RankedTensorOf<[AnyFloat,I8]>>:$lhs_scale,
      Optional<RankedTensorOf<[AnyFloat,I8]>>:$rhs_scale,
      BoolAttr:$fastMath,
      DefaultValuedAttr<BoolAttr, "true">:$lhs_k_pack,
      DefaultValuedAttr<BoolAttr, "true">:$rhs_k_pack
    );

    let results = (outs RankedTensorOf<[AnyFloat]>:$result);

    let assemblyFormat = [{
      $lhs (`scale` $lhs_scale^)? `,` $rhs (`scale` $rhs_scale^)? attr-dict
      `:` type($lhs) (`,` type($lhs_scale)^)? `*` type($rhs) (`,` type($rhs_scale)^)? `->` type($result)
    }];
}

#endif // XLA_CODEGEN_XTILE_IR_XTILE_OPS

