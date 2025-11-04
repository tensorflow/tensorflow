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

def TiledBufferInterface : OpInterface<"TiledBufferInterface"> {
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

// TODO(willfroom): Revisit if/where these are needed after the migration of the
// triton emitter to support 0D tensors is complete.
def ToScalarOp : XTile_Op<"to_scalar", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Converts a 0D tensor to a scalar";
  let arguments = (ins 0DTensorOf<[AnyType]>:$input);
  let results = (outs AnyType:$output);

  let assemblyFormat = "$input `:` type($input) attr-dict";

  let hasFolder = 1;
}

def ToTensorOp : XTile_Op<"to_tensor", [Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Converts a scalar to a 0D tensor";
  let arguments = (ins AnyType:$input);
  let results = (outs 0DTensorOf<[AnyType]>:$output);

  let assemblyFormat = "$input `:` type($input) attr-dict";

  let hasFolder = 1;
}

#endif // XLA_CODEGEN_XTILE_IR_XTILE_OPS

