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

#ifndef TENSORFLOW_CORE_IR_DIALECT_H_
#define TENSORFLOW_CORE_IR_DIALECT_H_

#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "tensorflow/core/ir/types/dialect.h"

namespace mlir {
namespace tfg {
// Include the relevant TensorFlow attrs/types directly in the TFG namespace.
using mlir::tf_type::Bfloat16RefType;      // NOLINT
using mlir::tf_type::BoolRefType;          // NOLINT
using mlir::tf_type::Complex128RefType;    // NOLINT
using mlir::tf_type::Complex64RefType;     // NOLINT
using mlir::tf_type::ControlType;          // NOLINT
using mlir::tf_type::DoubleRefType;        // NOLINT
using mlir::tf_type::Float8E4M3FNRefType;  // NOLINT
using mlir::tf_type::Float8E5M2RefType;    // NOLINT
using mlir::tf_type::FloatRefType;         // NOLINT
using mlir::tf_type::FuncAttr;             // NOLINT
using mlir::tf_type::HalfRefType;          // NOLINT
using mlir::tf_type::Int16RefType;         // NOLINT
using mlir::tf_type::Int32RefType;         // NOLINT
using mlir::tf_type::Int4RefType;          // NOLINT
using mlir::tf_type::Int64RefType;         // NOLINT
using mlir::tf_type::Int8RefType;          // NOLINT
using mlir::tf_type::OpaqueTensorType;     // NOLINT
using mlir::tf_type::PlaceholderAttr;      // NOLINT
using mlir::tf_type::Qint16RefType;        // NOLINT
using mlir::tf_type::Qint16Type;           // NOLINT
using mlir::tf_type::Qint32RefType;        // NOLINT
using mlir::tf_type::Qint32Type;           // NOLINT
using mlir::tf_type::Qint8RefType;         // NOLINT
using mlir::tf_type::Qint8Type;            // NOLINT
using mlir::tf_type::Quint16RefType;       // NOLINT
using mlir::tf_type::Quint16Type;          // NOLINT
using mlir::tf_type::Quint8RefType;        // NOLINT
using mlir::tf_type::Quint8Type;           // NOLINT
using mlir::tf_type::ResourceRefType;      // NOLINT
using mlir::tf_type::ResourceType;         // NOLINT
using mlir::tf_type::ShapeAttr;            // NOLINT
using mlir::tf_type::StringRefType;        // NOLINT
using mlir::tf_type::StringType;           // NOLINT
using mlir::tf_type::Uint16RefType;        // NOLINT
using mlir::tf_type::Uint32RefType;        // NOLINT
using mlir::tf_type::Uint4RefType;         // NOLINT
using mlir::tf_type::Uint64RefType;        // NOLINT
using mlir::tf_type::Uint8RefType;         // NOLINT
using mlir::tf_type::VariantRefType;       // NOLINT
using mlir::tf_type::VariantType;          // NOLINT
using mlir::tf_type::VersionAttr;          // NOLINT

class TFGraphOpAsmInterface;
class TFOp;
}  // namespace tfg
}  // namespace mlir

// Dialect main class is defined in ODS, we include it here.
#include "tensorflow/core/ir/dialect.h.inc"  // IWYU pragma: export
// ODS-generated attribute classes.
#define GET_ATTRDEF_CLASSES
#include "tensorflow/core/ir/attributes.h.inc"

#endif  // TENSORFLOW_CORE_IR_DIALECT_H_
