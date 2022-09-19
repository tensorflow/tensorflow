/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef GML_ST_TILING_INTERFACE
#define GML_ST_TILING_INTERFACE

include "mlir/IR/OpBase.td"

def TilingInterface : OpInterface<"TilingInterface"> {
  let description = [{
    Interface for operations to expose information needed to tile them.
  }];
  let cppNamespace = "::mlir::gml_st";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          Returns a list of operands into which the result of the
          tiled implementation is written into. With `tensor`
          operands, this will be used as the initial tensor into which
          the tiled results are inserted into. With `memref` operands,
          this will be the operand into which the result of the tiled
          operation is written into.
        }],
        /*retType=*/"SmallVector<Value>",
        /*methodName=*/"getDestinationOperands",
        /*args=*/(ins "OpBuilder &":$b)
      >,
      InterfaceMethod<
        /*desc=*/[{
          Returns a list of `StringRef`s that describe the number of
          loops and the iterator types of the operation. The list is
          expected to use
          `getParallelIteratorTypeName()`/`getReductionIteratorTypeName()`
          from MLIR Structured Op Utils.
        }],
        /*retType=*/"SmallVector<StringRef>",
        /*methodName=*/"getLoopIteratorTypes",
        /*args=*/(ins)
      >,
      InterfaceMethod<
        /*desc=*/[{
          Returns a list of ranges that describe the loop bounds and
          step for the loops of the operation.
        }],
        /*retTy=*/"SmallVector<Range>",
        /*methodName=*/"getIterationDomain",
        /*args=*/(ins "OpBuilder &":$b)
      >,
      InterfaceMethod<
        /*desc=*/[{
          Method to generate the tiled implementation of an operation.

          The iteration space of the operation is returned by
          `getIterationDomain`. The caller provides the information of the
          tile within this iteration space whose implementation the
          caller needs.
          - `offsets` provides the offset of the tile in the coordinate system
            of the original iteration space, i.e., if an iteration space
            dimension had non-zero offset, it must be included in the offset
            provided here (as opposed to zero-based offset "relative" to the
            iteration space).
          - `sizes` provides the size of the tile.

          The method returns the operation that is the tiled
          implementation.
        }],
        /*retType=*/"mlir::gml_st::TilingInterface",
        /*methodName=*/"getTiledImplementation",
        /*args=*/(ins
            "OpBuilder &":$b,
            "ArrayRef<OpFoldResult>":$offsets,
            "ArrayRef<OpFoldResult>":$sizes)
      >,
      InterfaceMethod<
        /*desc=*/[{
          Generates the IR that computes the tile of a result of the
          operation.  The `offsets` and `sizes` describe the tile of
          the output required. This is different from
          `getTiledImplementation` which generates the tiled
          implementation of the operation given a tile of the
          iteration space. This method generates a tiled
          implementation of the operation based on the tile of the
          result required. This method enables fusion by using tile
          and fuse. The method returns failure if the operation can't be
          tiled to generate the result tile. In practical terms this
          implies it cannot be tiled and fused with its consumers.

          - `offsets` provides the offset of the tile in the coordinate system
            of the original iteration space, i.e., if an iteration space
            dimension had non-zero offset, it must be included in the offset
            provided here (as opposed to zero-based offset "relative" to the
            iteration space).
          - `sizes` provides the size of the tile.
        }],
        /*retType=*/"FailureOr<Value>",
        /*methodName=*/"generateResultTileValue",
        /*args=*/(ins
          "OpBuilder &":$b,
          "unsigned":$resultNumber,
          "ArrayRef<OpFoldResult>":$offsets,
          "ArrayRef<OpFoldResult>":$sizes)
      >
  ];
}
#endif // GML_ST_TILING_INTERFACE
