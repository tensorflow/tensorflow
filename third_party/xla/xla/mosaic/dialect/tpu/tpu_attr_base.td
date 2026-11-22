/* Copyright 2026 The OpenXLA Authors.

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

#ifndef TPU_ATTR_BASE
#define TPU_ATTR_BASE

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

class TPU_TiledLayoutAttrBase<Dialect dialect, code extraDecl = "">
  : AttrDef<dialect, "TiledLayout",
             [DeclareAttrInterfaceMethods<MemRefLayoutAttrInterface, ["verifyLayout"]>]> {
  let mnemonic = "tiled";
  let description = [{
    This attribute represents tiled layouts in memrefs.

    Multiple levels of tiling are supported with the following restriction:
    - Additional levels of tiling may not add any padding.
    - Additional levels of tiling may not tile previously untiled dimensions,
      that is, they cannot tile across first-level tiles.

    Tile strides encode the stride when moving along a given dimension. They
    must have the same rank as the shape and must be decreasing with increasing
    dimension number. For tiled dimensions, the stride applies only when moving
    across first-level tiles. The strides are in units of the size of the first
    tile, or 1 if there are no tiles.
  }];
  let parameters = (ins
    ArrayRefParameter<"::xla::Tile", "">:$tiles,
    ArrayRefParameter<"int64_t", "">:$tile_strides
  );
  let extraClassDeclaration = [{
    static ::llvm::SmallVector<int64_t> getContiguousTileStrides(
        ::llvm::ArrayRef<::xla::Tile> tiles, ::llvm::ArrayRef<int64_t> shape);
    static TiledLayoutAttr getContiguous(::mlir::MLIRContext* context, ::llvm::ArrayRef<::xla::Tile> tiles, ::llvm::ArrayRef<int64_t> shape);

    // Returns the number of trailing dimensions where tiles can be statically
    // determined to be contiguous (there are no gaps or overlap between tiles).
    // NOTE: Tiles being contiguous does not imply the _data_ is contiguous
    //   since the tiles may contain padding. This means this function is
    //   different from MemRefType::getNumContiguousTrailingDims.
    int64_t getNumTrailingDimsWithContiguousTiles(
        ::llvm::ArrayRef<int64_t> shape) const;
    bool tilesAreKnownContiguous(const ::llvm::ArrayRef<int64_t> shape) const {
      return getNumTrailingDimsWithContiguousTiles(shape) == getRank();
    }

    int64_t getRank() const {
      return getTileStrides().size();
    }

    ::llvm::SmallVector<int64_t> getExpandedStrides() const;

    // Returns true if any of the tile strides are dynamic.
    bool hasDynamicStrides() const;
    int64_t getNumDynamicStrides() const;
  }] # extraDecl;
  let extraClassDefinition = [{
    void $cppClass::print(::mlir::AsmPrinter& printer) const {
      ::mlir::tpu::impl::printTiledLayout(getTiles(), getTileStrides(),
                                          printer);
    }

    ::mlir::Attribute $cppClass::parse(::mlir::AsmParser& parser,
                                       ::mlir::Type type) {
      ::llvm::SmallVector<::xla::Tile, 2> tiles;
      ::llvm::SmallVector<int64_t, 2> tile_strides;
      if (::mlir::failed(::mlir::tpu::impl::parseTiledLayout(
              parser, tiles, tile_strides))) {
        return {};
      }
      return $cppClass::get(parser.getContext(), tiles, tile_strides);
    }

    ::mlir::AffineMap $cppClass::getAffineMap() const {
      return ::mlir::tpu::impl::getTiledLayoutAffineMap(
          getContext(), getTiles(), getTileStrides());
    }

    ::llvm::SmallVector<int64_t> $cppClass::getContiguousTileStrides(
        ::llvm::ArrayRef<::xla::Tile> tiles, ::llvm::ArrayRef<int64_t> shape) {
      return ::mlir::tpu::impl::getContiguousTileStrides(tiles, shape);
    }

    $cppClass $cppClass::getContiguous(::mlir::MLIRContext* context,
                                       ::llvm::ArrayRef<::xla::Tile> tiles,
                                       ::llvm::ArrayRef<int64_t> shape) {
      return ::mlir::tpu::impl::getContiguousTiledLayout<$cppClass>(
          context, tiles, shape);
    }

    int64_t $cppClass::getNumTrailingDimsWithContiguousTiles(
        ::llvm::ArrayRef<int64_t> shape) const {
      return ::mlir::tpu::impl::getNumTrailingDimsWithContiguousTiles(
          getTiles(), getTileStrides(), shape);
    }

    ::llvm::SmallVector<int64_t> $cppClass::getExpandedStrides() const {
      auto strides =
          ::mlir::tpu::impl::getExpandedStrides(getTiles(), getTileStrides());
      CHECK(::mlir::succeeded(strides));
      return *strides;
    }

    bool $cppClass::hasDynamicStrides() const {
      return ::llvm::any_of(getTileStrides(), ::mlir::ShapedType::isDynamic);
    }

    int64_t $cppClass::getNumDynamicStrides() const {
      return ::llvm::count_if(getTileStrides(), ::mlir::ShapedType::isDynamic);
    }

    ::llvm::LogicalResult $cppClass::verify(
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
        ::llvm::ArrayRef<::xla::Tile> tiles,
        ::llvm::ArrayRef<int64_t> tile_strides) {
      return ::mlir::tpu::impl::verifyTiledLayout(emitError, tiles,
                                                  tile_strides);
    }

    ::llvm::LogicalResult $cppClass::verifyLayout(
        ::llvm::ArrayRef<int64_t> shape,
        ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError) const {
      return ::mlir::tpu::impl::verifyTiledLayoutRank(getRank(), shape,
                                                      emitError);
    }
  }];

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;
}

#endif  // TPU_ATTR_BASE
