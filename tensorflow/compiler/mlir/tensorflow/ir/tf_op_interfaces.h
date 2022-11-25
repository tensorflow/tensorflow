/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_OP_INTERFACES_H_
#define TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_OP_INTERFACES_H_

#include <string>

#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_verifiers.h"
#include "tensorflow/core/framework/resource_mgr.h"

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// TensorFlow Contraction Fusion.
//===----------------------------------------------------------------------===//

struct ContractionFusion {
  explicit ContractionFusion(
      StringRef output_kernel, ArrayRef<int> additional_arguments = {},
      ArrayRef<NamedAttribute> additional_attributes = {})
      : output_kernel(output_kernel.str()),
        additional_arguments(additional_arguments.begin(),
                             additional_arguments.end()),
        additional_attributes(additional_attributes.begin(),
                              additional_attributes.end()) {}

  // Name of the output kernel implementing the contraction fusion.
  std::string output_kernel;

  // Indices of additional arguments that will be forwarded to the fused
  // operation (e.g. forward bias vector if fusing BiasAdd operation).
  SmallVector<int, 4> additional_arguments;

  // Add additional attributes to the fused node.
  SmallVector<NamedAttribute, 4> additional_attributes;
};

//===----------------------------------------------------------------------===//
// TensorFlow Resource Handles.
//===----------------------------------------------------------------------===//

inline bool IsResourceHandleAnonymous(StringRef name) {
  return name == ::tensorflow::ResourceHandle::ANONYMOUS_NAME;
}

// Helper struct representing an identifier for a resource handle. For resource
// handles created explicitly and shared across resource allocator ops,
// `container`, `name`, and `device` can be set. If an resource handle is tied
// to an instance of an operation (e.g. TensorFlow runtime operation caching),
// `op` can be set instead.
struct ResourceHandle {
  ResourceHandle(StringRef container, StringRef name, StringRef device,
                 Operation* op)
      : container(container), name(name), device(device), op(op) {}

  bool operator==(const ResourceHandle& rhs) const {
    return container == rhs.container && name == rhs.name &&
           device == rhs.device && op == rhs.op;
  }

  // Make ResourceHandle hashable.
  friend ::llvm::hash_code hash_value(const ResourceHandle& resource_handle);

  StringRef container;
  StringRef name;
  StringRef device;
  Operation* op = nullptr;
};

// Make ResourceHandle hashable.
inline ::llvm::hash_code hash_value(const ResourceHandle& resource_handle) {
  return ::llvm::hash_combine(resource_handle.container, resource_handle.name,
                              resource_handle.device, resource_handle.op);
}

// Helper struct holding a resource handle value and unique id associated to the
// resource handle.
struct ResourceHandleValueAndId {
  ResourceHandleValueAndId(Value value, int64_t id) : value(value), id(id) {}

  Value value;
  int64_t id = -1;
};

//===----------------------------------------------------------------------===//
// TF op helper functions for handling resource handles and ids.
//===----------------------------------------------------------------------===//

// Returns device of op if present. If op has no device set, an empty string ref
// is returned instead.
llvm::StringRef GetDeviceOrEmpty(Operation* op);

// Returns resource handle value and id for resource op based on attributes. If
// a resource handle is anonymous, a new id is always returned.
ResourceHandleValueAndId GetResourceHandleValueAndIdBase(
    llvm::StringRef container, llvm::StringRef shared_name,
    llvm::StringRef device, Value resource,
    llvm::SmallDenseMap<ResourceHandle, int64_t>& resource_handle_id_map,
    int64_t& next_id);

// Shape functions for ops that are using TF_SameOperandsAndResultTypeResolveRef
// and have at least one operand, result type can be inferred using the first
// operand's type.

#define INFER_RETURN_TYPE_COMPONENTS_FROM_OPERANDS(Op)                        \
  LogicalResult Op::inferReturnTypeComponents(                                \
      MLIRContext* context, Optional<Location> location,                      \
      ValueShapeRange operands, DictionaryAttr attributes,                    \
      RegionRange regions,                                                    \
      SmallVectorImpl<ShapedTypeComponents>& inferredReturnShapes) {          \
    return inferReturnTypeComponentsFromOperands(context, location, operands, \
                                                 attributes, regions,         \
                                                 inferredReturnShapes);       \
  }

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_op_interfaces.h.inc"
}  // namespace TF
}  // namespace mlir

namespace llvm {
template <>
struct DenseMapInfo<mlir::TF::ResourceHandle> {
  static mlir::TF::ResourceHandle getEmptyKey() {
    return {/*container=*/"", /*name=*/"", /*device=*/"",
            /*op=*/DenseMapInfo<mlir::Operation*>::getEmptyKey()};
  }

  static mlir::TF::ResourceHandle getTombstoneKey() {
    return {/*container=*/"", /*name=*/"", /*device=*/"",
            /*op=*/DenseMapInfo<mlir::Operation*>::getTombstoneKey()};
  }

  static unsigned getHashValue(
      const mlir::TF::ResourceHandle& resource_handle) {
    return mlir::TF::hash_value(resource_handle);
  }

  static bool isEqual(const mlir::TF::ResourceHandle& lhs,
                      const mlir::TF::ResourceHandle& rhs) {
    return lhs == rhs;
  }
};
}  // namespace llvm

#endif  // TENSORFLOW_COMPILER_MLIR_TENSORFLOW_IR_TF_OP_INTERFACES_H_
