//===- TypeDetail.h - MLIR Affine Expr storage details ----------*- C++ -*-===//
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
// This holds implementation details of Type.
//
//===----------------------------------------------------------------------===//
#ifndef TYPEDETAIL_H_
#define TYPEDETAIL_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"

namespace mlir {

class AffineMap;
class MLIRContext;

namespace detail {

/// Base storage class appearing in a Type.
struct alignas(8) TypeStorage {
  TypeStorage(Type::Kind kind, MLIRContext *context)
      : context(context), kind(kind), subclassData(0) {}
  TypeStorage(Type::Kind kind, MLIRContext *context, unsigned subclassData)
      : context(context), kind(kind), subclassData(subclassData) {}

  unsigned getSubclassData() const { return subclassData; }

  void setSubclassData(unsigned val) {
    subclassData = val;
    // Ensure we don't have any accidental truncation.
    assert(getSubclassData() == val && "Subclass data too large for field");
  }

  /// This refers to the MLIRContext in which this type was uniqued.
  MLIRContext *const context;

  /// Classification of the subclass, used for type checking.
  Type::Kind kind : 8;

  /// Space for subclasses to store data.
  unsigned subclassData : 24;
};

struct IntegerTypeStorage : public TypeStorage {
  unsigned width;
};

struct FloatTypeStorage : public TypeStorage {};

struct OtherTypeStorage : public TypeStorage {};

struct FunctionTypeStorage : public TypeStorage {
  ArrayRef<Type> getInputs() const {
    return ArrayRef<Type>(inputsAndResults, subclassData);
  }
  ArrayRef<Type> getResults() const {
    return ArrayRef<Type>(inputsAndResults + subclassData, numResults);
  }

  unsigned numResults;
  Type const *inputsAndResults;
};

struct VectorOrTensorTypeStorage : public TypeStorage {
  Type elementType;
};

struct VectorTypeStorage : public VectorOrTensorTypeStorage {
  ArrayRef<int> getShape() const {
    return ArrayRef<int>(shapeElements, getSubclassData());
  }

  const int *shapeElements;
};

struct TensorTypeStorage : public VectorOrTensorTypeStorage {};

struct RankedTensorTypeStorage : public TensorTypeStorage {
  ArrayRef<int> getShape() const {
    return ArrayRef<int>(shapeElements, getSubclassData());
  }

  const int *shapeElements;
};

struct UnrankedTensorTypeStorage : public TensorTypeStorage {};

struct MemRefTypeStorage : public TypeStorage {
  ArrayRef<int> getShape() const {
    return ArrayRef<int>(shapeElements, getSubclassData());
  }

  ArrayRef<AffineMap> getAffineMaps() const {
    return ArrayRef<AffineMap>(affineMapList, numAffineMaps);
  }

  /// The type of each scalar element of the memref.
  Type elementType;
  /// An array of integers which stores the shape dimension sizes.
  const int *shapeElements;
  /// The number of affine maps in the 'affineMapList' array.
  const unsigned numAffineMaps;
  /// List of affine maps in the memref's layout/index map composition.
  AffineMap const *affineMapList;
  /// Memory space in which data referenced by memref resides.
  const unsigned memorySpace;
};

} // namespace detail
} // namespace mlir
#endif // TYPEDETAIL_H_
