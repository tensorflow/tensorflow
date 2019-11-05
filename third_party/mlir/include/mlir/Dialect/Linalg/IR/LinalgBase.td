//===- LinalgBase.td - Linalg dialect base support ---------*- tablegen -*-===//
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
// This is the definition file for base linear algebra support.
//
//===----------------------------------------------------------------------===//

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE

#ifndef LINALG_BASE
#define LINALG_BASE

def Linalg_Dialect : Dialect {
  let name = "linalg";
  let description = [{
    The Linalg dialect groups together a set of types and operations that are
    useful to implement a "linear algebra"-like abstraction where ops can lower
    to scalar load/store and operations or to more general library calls.

    The Linalg dialect adopts a convention that is similar to BLAS when
    offloading operations to fast library implementations: pass a non-owning
    pointer to input and output data with additional metadata. This convention
    is also found in libraries such as MKL, OpenBLAS, cuBLAS, cuDNN, etc.. and
    more generally at interface points across language boundaries (e.g. C++ /
    Python).

    Generally, Linalg passes non-owning pointers to strided memref data
    structures to precompiled library calls linked externally. The name `view`
    is used interchangeably in Linalg to signify strided memref.
  }];
}

// Whether a type is a BufferType.
def LinalgIsBufferTypePred : CPred<"$_self.isa<BufferType>()">;
def Buffer : Type<LinalgIsBufferTypePred, "buffer">;

// Whether a type is a RangeType.
def LinalgIsRangeTypePred : CPred<"$_self.isa<RangeType>()">;
def Range : Type<LinalgIsRangeTypePred, "range">;

#endif // LINALG_BASE
