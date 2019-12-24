//===- LinalgBase.td - Linalg dialect base support ---------*- tablegen -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the definition file for base linear algebra support.
//
//===----------------------------------------------------------------------===//

#ifndef LINALG_BASE
#define LINALG_BASE

include "mlir/IR/OpBase.td"

def Linalg_Dialect : Dialect {
  let name = "linalg";
  let description = [{
    The `linalg` dialect groups together a set of types, operations and
    transformations that are useful to implement a structured abstraction where
    ops can lower to scalar load/store and operations or to more general library
    calls.

    The `linalg` dialect manipulates the following types and operations:

    ### Core data types and special ops.

    The following abstractions are used by the `linalg` dialect:

    #### Views
    The current implementation uses the strided memref abstraction. In the
    future other abstractions than strided memref will be used.

    #### `!linalg.range`
    This data type is currently just a triple (`min`,`max`, `step`) that does
    not pass function boundaries.

    #### `linalg.yield`
    This op is used as a terminator within the appropriate `linalg` regions.

    In the future, richer `view` and `range` representations are expected, in
    particular to represent sparse traversals.

    ### Metadata Ops
    A set of ops that manipulate metadata but do not move memory. These ops take
    `view` operands + extra attributes and return new `view`s. The returned
    `view`s generally alias the operand `view`. At the moment the existing ops
    are:

        * `std.view`,
        * `std.subview`,
        * `linalg.range`,
        * `linalg.slice`,
        * `linalg.transpose`.

    Future ops are added on a per-need basis but should include:

        * `linalg.reshape`,
        * `linalg.tile`,
        * `linalg.intersection`,
        * `linalg.convex_union`,
        * `linalg.difference` (would need to work on a list of views).

    ### Payload Ops
    A set of payload carrying operations that implement the [structured ops](
    https://docs.google.com/presentation/d/1P-j1GrH6Q5gLBjao0afQ-GfvcAeF-QU4GXXeSy0eJ9I/edit#slide=id.p
    )
    abstraction on buffers. `linalg` has `2` generic operations `linalg.generic`
    and `linalg.indexed_generic` for expressing custom operations. This is
    subject to further evolution as transformations and analyses continue to be
    developed.

    Additionally, `linalg` provides some common named operations:

        * `linalg.copy`,
        * `linalg.fill`,
        * `linalg.dot`,
        * `linalg.matmul`,
        * `linalg.conv`.

    Future ops are added on a per-need basis but should include:

        * `linalg.pad`.

    In an ideal world, all the named ops would be automatically generated from
    a description in terms of only the `2` generic ops. Unfortunately we do not
    have such support yet (contributions are most welcome).

    ### Convention for external library interop
    The `linalg` dialect adopts a convention that is similar to `BLAS` when
    offloading operations to fast library implementations: pass a non-owning
    pointer to input and output data with additional metadata. This convention
    is also found in libraries such as `MKL`, `OpenBLAS`, `BLIS`, `cuBLAS`,
    `cuDNN`, etc.. and more generally at interface points across language
    boundaries (e.g. C++ / Python).

    Generally, `linalg` passes non-owning pointers to strided memref data
    structures to precompiled library calls linked externally. The name `view`
    is used interchangeably in `linalg` to signify strided memref discussed at
    length in the [strided memref RFC](
    https://groups.google.com/a/tensorflow.org/g/mlir/c/MaL8m2nXuio/m/a_v07o9yBwAJ).
  }];
}

// Whether a type is a RangeType.
def LinalgIsRangeTypePred : CPred<"$_self.isa<RangeType>()">;
def Range : Type<LinalgIsRangeTypePred, "range">;

#endif // LINALG_BASE
