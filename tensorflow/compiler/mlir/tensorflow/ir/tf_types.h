/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file defines the types used in the standard MLIR TensorFlow dialect.

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_TYPES_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_TYPES_H_

#include "mlir/IR/Types.h"  // TF:local_config_mlir

namespace mlir {
namespace TF {

namespace TensorFlowTypes {
// List of supported TensorFlowType kinds, necessary for isa/dyn_cast.
enum Kind {
  FIRST_USED_TENSORFLOW_TYPE = Type::FIRST_TENSORFLOW_TYPE,
#define HANDLE_TF_TYPE(tftype, enumerant, name) enumerant,
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"
  LAST_USED_TENSORFLOW_TYPE,
};
}  // namespace TensorFlowTypes

// The base class in the tensor flow type hierarchy.
class TensorFlowType : public Type {
 public:
  using Type::Type;

  // Support method to enable LLVM-style type casting.
  static bool classof(Type type) {
    return type.getKind() >= Type::FIRST_TENSORFLOW_TYPE &&
           type.getKind() <= TensorFlowTypes::LAST_USED_TENSORFLOW_TYPE;
  }
};

namespace detail {
// Common implementation of TensorFlow types.  The template argument indicates
// the concrete derived class per CRTP.  Concrete classes must implement the
// following:
//   - `static unsigned getTypeKind()` that returns the (fixed) kind of the
//     type.
template <typename Derived>
class TensorFlowTypeImpl : public Type::TypeBase<Derived, TensorFlowType> {
 public:
  using Base = typename Type::TypeBase<Derived, TensorFlowType>;
  using TFBase = TensorFlowTypeImpl<Derived>;
  using Base::Base;

  // Gets the unique'ed type in the given context.
  static Derived get(MLIRContext *context) {
    return Base::get(context, Derived::getTypeKind());
  }

  /// Supports method to enable LLVM-style type casting.
  static bool kindof(unsigned kind) { return kind == Derived::getTypeKind(); }
};
}  // namespace detail

#define HANDLE_TF_TYPE(tftype, enumerant, name)                          \
  class tftype##Type : public detail::TensorFlowTypeImpl<tftype##Type> { \
   public:                                                               \
    using TFBase::TFBase;                                                \
    static unsigned getTypeKind() { return TensorFlowTypes::enumerant; } \
  };

// NOLINTNEXTLINE
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"

}  // namespace TF
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_TYPES_H_
