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

#ifndef XLA_CODEGEN_XTILE_IR_XTILE_ATTRS
#define XLA_CODEGEN_XTILE_IR_XTILE_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "xla/codegen/xtile/ir/xtile_dialect.td"

class XTile_Attr<string name, list<Trait> traits = []> :
      AttrDef<XTileDialect, name, traits> {
}

def XTile_TilingInfoAttr : XTile_Attr<"TilingInfo"> {
  let summary = "Contains auxillary information about how a program is tiled.";
  let mnemonic = "tiling_info";
  let parameters = (ins "int32_t":$tile_count, "int32_t":$tiles_per_workgroup);

  let assemblyFormat = [{
    `<` `tile_count` `:` $tile_count `,`
    `tiles_per_workgroup` `:` $tiles_per_workgroup `>`
  }];
}

def XTile_LayoutAttr : XTile_Attr<"Layout", [
    DeclareAttrInterfaceMethods<MemRefLayoutAttrInterface>]> {
  let mnemonic = "layout";
  let parameters = (ins "::mlir::DenseI64ArrayAttr":$minor_to_major);
  let assemblyFormat = "`<` $minor_to_major `>`";
  let extraClassDeclaration = [{
    mlir::LogicalResult verifyLayout(
        mlir::ArrayRef<int64_t> shape, mlir::function_ref<mlir::InFlightDiagnostic()> emit_error) const;
    mlir::LogicalResult getStridesAndOffset(
        mlir::ArrayRef<int64_t> shape, mlir::SmallVectorImpl<int64_t>& strides, int64_t& offset) const;
  }];
}

#endif // XLA_CODEGEN_XTILE_IR_XTILE_ATTRS
