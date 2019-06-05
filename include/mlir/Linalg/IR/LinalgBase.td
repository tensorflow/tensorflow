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

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

#ifdef LINALG_BASE
#else
#define LINALG_BASE

def Linalg_Dialect : Dialect {
  let name = "linalg";
}

// Whether a type is a BufferType.
def LinalgIsBufferTypePred : CPred<"$_self.isa<BufferType>()">;
def Buffer : Type<LinalgIsBufferTypePred, "buffer">;

// Whether a type is a RangeType.
def LinalgIsRangeTypePred : CPred<"$_self.isa<RangeType>()">;
def Range : Type<LinalgIsRangeTypePred, "range">;

// Whether a type is a ViewType.
def LinalgIsViewTypePred : CPred<"$_self.isa<ViewType>()">;
def View : Type<LinalgIsViewTypePred, "view">;

#endif // LINALG_BASE