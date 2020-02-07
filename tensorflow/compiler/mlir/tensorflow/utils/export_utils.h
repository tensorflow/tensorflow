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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_EXPORT_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_EXPORT_UTILS_H_

#include <functional>
#include <memory>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Location.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/Types.h"  // TF:llvm-project
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace tensorflow {

using stream_executor::port::StatusOr;

// Maps an MLIR op name in the TensorFlow dialect or the TensorFlow control
// dialect back into a TensorFlow valid op name.
StatusOr<llvm::StringRef> GetTensorFlowOpName(llvm::StringRef);

// Converts an MLIR operation to TensorFlow NodeDef with given node name. This
// name should be unique to the graph it is being inserted into. `op_name_func`
// is to map the op name of `inst` to its op name in TensorFlow. "name" and
// "device" attributes are ignored by default. Use attrs_to_ignore to specify
// any other attributes that should be ignored.
StatusOr<std::unique_ptr<NodeDef>> GetOperationNodeDef(
    const absl::flat_hash_set<absl::string_view>& attrs_to_ignore,
    mlir::Operation* inst, llvm::StringRef name);

// Converts MLIR attributes with values to their tensorflow equivalent.
// "name" and "device" attributes are ignored by default. Use attrs_to_ignore to
// specify any other attributes that should be ignored.
Status ConvertAttributes(
    const llvm::ArrayRef<mlir::NamedAttribute> attrs,
    const absl::flat_hash_set<absl::string_view>& attrs_to_ignore,
    AttrValueMap* values);

// Sets type attribute with the given name. If the attribute already exists with
// a different value, returns an error.
Status SetTypeAttribute(absl::string_view name, mlir::Type type,
                        AttrValueMap* values);

// Sets shape attribute with the given name. If the attribute already exists
// with a different value, returns an error.
Status SetShapeAttribute(absl::string_view name, mlir::ShapedType shape,
                         AttrValueMap* values);

// Sets the given size_t value as an integer attribute with the given name.
// If the attribute already exists with a different value, returns an error.
Status SetSizeAttribute(absl::string_view name, size_t size,
                        AttrValueMap* values);

// Returns true if the given instruction is an mlir::TF::LegacyCallOp or the
// result of such an operation transformed by the
// ExecutorToControlDialectConversion pass.
//
// TODO(b/145706023): When the ExecutorToControlDialectConversion pass runs
// before the exporter, it mutates an mlir::TF::LegacyCallOp instruction to
// an instruction with a different operation name. As such, this routine checks
// both forms of a LegacyCall instruction. We only need to check for
// mlir::TF::LegacyCallOp when the ticket is resolved.
bool IsLegacyCallInstruction(mlir::Operation* inst);
}  // namespace tensorflow
#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_UTILS_EXPORTER_UTILS_H_
