//===- LinalgTypes.h - Linalg Types ---------------------------------------===//
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

#ifndef MLIR_LINALG_LINALGTYPES_H_
#define MLIR_LINALG_LINALGTYPES_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Types.h"

namespace mlir {
class MLIRContext;

namespace linalg {
enum LinalgTypes {
  Buffer = Type::FIRST_LINALG_TYPE,
  Range,
  View,
  LAST_USED_LINALG_TYPE = View,
};

class LinalgDialect : public Dialect {
public:
  explicit LinalgDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "linalg"; }

  /// Parse a type registered to this dialect.
  Type parseType(llvm::StringRef spec, Location loc) const override;

  /// Print a type registered to this dialect.
  void printType(Type type, llvm::raw_ostream &os) const override;
};

/// A BufferType represents a contiguous block of memory that can be allocated
/// and deallocated. A buffer cannot be indexed directly, a view must be
/// laid out on a buffer to give it indexing semantics.
struct BufferTypeStorage;
class BufferType : public Type::TypeBase<BufferType, Type, BufferTypeStorage> {
public:
  // Used for generic hooks in TypeBase.
  using Base::Base;
  /// Construction hook.
  static BufferType get(MLIRContext *context, Type elementType,
                        int64_t bufferSize = -1);
  /// Used to implement llvm-style cast.
  static bool kindof(unsigned kind) { return kind == LinalgTypes::Buffer; }

  // Type-specific functionality.
  Type getElementType();
  bool hasConstantSize();
  Optional<int64_t> getBufferSize();
};

/// A RangeType represents a minimal range abstraction (min, max, step).
/// It is constructed by calling the linalg.range op with three values index of
/// index type:
///
/// ```{.mlir}
///    func @foo(%arg0 : index, %arg1 : index, %arg2 : index) {
///      %0 = linalg.range %arg0:%arg1:%arg2 : !linalg.range
///    }
/// ```
class RangeType : public Type::TypeBase<RangeType, Type> {
public:
  // Used for generic hooks in TypeBase.
  using Base::Base;
  /// Construction hook.
  static RangeType get(MLIRContext *context) {
    /// Custom, uniq'ed construction in the MLIRContext.
    return Base::get(context, LinalgTypes::Range);
  }
  /// Used to implement llvm-style cast.
  static bool kindof(unsigned kind) { return kind == LinalgTypes::Range; }
};

/// A ViewType represents a multi-dimensional range abstraction on top of an
/// underlying storage type. It is parameterizable by the underlying element
/// type and the rank of the view.
/// A new value of ViewType is constructed from a buffer with a view op and
/// passing it ranges:
///
/// ```{.mlir}
///    %1 = linalg.buffer_alloc %0 : !linalg.buffer<f32>
///    %2 = linalg.range %arg2:%arg3:%arg4 : !linalg.range
///    %3 = linalg.view %1[%2, %2] : !linalg.view<?x?xf32>
/// ```
struct ViewTypeStorage;
class ViewType : public Type::TypeBase<ViewType, Type, ViewTypeStorage> {
public:
  // Used for generic hooks in TypeBase.
  using Base::Base;
  /// Construction hook.
  static ViewType get(MLIRContext *context, Type elementType, unsigned rank);
  // Used to implement llvm-style cast.
  static bool kindof(unsigned kind) { return kind == LinalgTypes::View; }

  // Type-specific functionality.
  /// Return the underlying elemental type.
  Type getElementType();
  /// Return the rank of the view.
  /// This is the number of indexings needed to reach an underlying element.
  unsigned getRank();
};

} // namespace linalg
} // namespace mlir

#endif // MLIR_LINALG_LINALGTYPES_H_
