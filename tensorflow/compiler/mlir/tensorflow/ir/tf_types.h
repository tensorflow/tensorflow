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

#include "tensorflow/core/ir/types/dialect.h"

namespace mlir {
namespace TF {

// This all moved under tensorflow/core/ir/types and these using declaration are
// to help with the transition.

using ::mlir::tf_type::AreCastCompatible;          // NOLINT
using ::mlir::tf_type::ArraysAreCastCompatible;    // NOLINT
using ::mlir::tf_type::BroadcastCompatible;        // NOLINT
using ::mlir::tf_type::DropRefType;                // NOLINT
using ::mlir::tf_type::filter_resources;           // NOLINT
using ::mlir::tf_type::GetCastCompatibleType;      // NOLINT
using ::mlir::tf_type::HasCompatibleElementTypes;  // NOLINT
using ::mlir::tf_type::IsValidTFTensorType;        // NOLINT
using ::mlir::tf_type::OperandShapeIterator;       // NOLINT
using ::mlir::tf_type::ResourceType;               // NOLINT
using ::mlir::tf_type::ResultShapeIterator;        // NOLINT
using ::mlir::tf_type::ResultShapeRange;           // NOLINT
using ::mlir::tf_type::StringType;                 // NOLINT
using ::mlir::tf_type::TensorFlowRefType;          // NOLINT
using ::mlir::tf_type::TensorFlowType;             // NOLINT
using ::mlir::tf_type::TensorFlowTypeWithSubtype;  // NOLINT
using ::mlir::tf_type::VariantType;                // NOLINT

#define HANDLE_TF_TYPE(tftype, enumerant, name) \
  using tftype##Type = mlir::tf_type::tftype##Type;
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.def"


}  // end namespace TF
}  // end namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_TYPES_H_
