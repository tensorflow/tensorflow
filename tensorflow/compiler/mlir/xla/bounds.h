/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_XLA_BOUNDS_H_
#define TENSORFLOW_COMPILER_MLIR_XLA_BOUNDS_H_

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/IntegerSet.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace mlir {

// Given the "encoding" attribute (stored in a RankedTensorType), add (or
// modify) a given dimension to have the given upper bound.
Attribute addOrModifyUpperBound(MLIRContext *context, Attribute encoding,
                                int dimension, int64_t limit);

// Returns the upper bound for a given dimension. If there is no upper bound for
// this dimension, then -1 is returned.
int64_t getUpperBoundFromAttr(::mlir::Attribute attr, int dimension);

// Returns a vector with an integer for each dimension of ty. Each element will
// either be equal to the dimension size (for non-dynamic dimension), or will
// contain the bound of that dimension size (for dynamic dimensions). If the
// dimension is dynamic but there is no bound, the entry will be -1.
llvm::SmallVector<int64_t, 4> getUpperBoundsForTensor(Attribute attr,
                                                      RankedTensorType ty);

// Attribute interface

namespace detail {
struct BoundsAttrInterfaceInterfaceTraits {
  struct Concept {};

  template <typename ConcreteAttr>
  class Model : public Concept {};

  template <typename ConcreteAttr>
  class FallbackModel : public Model<ConcreteAttr> {};

  template <typename ConcreteModel, typename ConcreteAttr>
  class ExternalModel : public Model<ConcreteModel> {};
};
}  // end namespace detail

class BoundsAttrInterface
    : public ::mlir::AttributeInterface<
          BoundsAttrInterface, detail::BoundsAttrInterfaceInterfaceTraits> {
 public:
  using ::mlir::AttributeInterface<
      BoundsAttrInterface,
      detail::BoundsAttrInterfaceInterfaceTraits>::AttributeInterface;

  int64_t getBound(int dimension) const {
    ::mlir::Attribute attr = *this;
    return getUpperBoundFromAttr(attr, dimension);
  }
  llvm::SmallVector<int64_t, 4> getBoundsForTensor(RankedTensorType ty) const {
    ::mlir::Attribute attr = *this;
    return getUpperBoundsForTensor(attr, ty);
  }
};

// Type interface

namespace detail {
struct BoundedRankedTensorTypeTraits {
  struct Concept {};

  template <typename ConcreteType>
  class Model : public Concept {};

  template <typename ConcreteType>
  class FallbackModel : public Model<ConcreteType> {};

  template <typename ConcreteModel, typename ConcreteType>
  class ExternalModel : public Model<ConcreteModel> {};
};
}  // end namespace detail

class BoundedRankedTensorType
    : public ::mlir::TypeInterface<BoundedRankedTensorType,
                                   detail::BoundedRankedTensorTypeTraits> {
 public:
  using ::mlir::TypeInterface<
      BoundedRankedTensorType,
      detail::BoundedRankedTensorTypeTraits>::TypeInterface;

  int64_t getBound(int dimension) const {
    ::mlir::Type t = *this;
    return getUpperBoundFromAttr(
        t.cast<::mlir::RankedTensorType>().getEncoding(), dimension);
  }
  llvm::SmallVector<int64_t, 4> getBounds() const {
    ::mlir::Type t = *this;
    return getUpperBoundsForTensor(
        t.cast<::mlir::RankedTensorType>().getEncoding(),
        t.cast<::mlir::RankedTensorType>());
  }
};

}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_XLA_BOUNDS_H_
