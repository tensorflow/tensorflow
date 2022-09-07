/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2022 The StableHLO Authors.

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

#ifndef STABLEHLO_DIALECT_STABLEHLO_OPS_H
#define STABLEHLO_DIALECT_STABLEHLO_OPS_H

#include <algorithm>

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TensorEncoding.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/Base.h"

// Include order matters.
#include "stablehlo/dialect/StablehloEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "stablehlo/dialect/StablehloAttrs.h.inc"

namespace mlir {
namespace stablehlo {

class StablehloDialect : public Dialect {
 public:
  explicit StablehloDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "stablehlo"; }

  // Registered hook to materialize a constant operation from a given attribute
  // value with the desired resultant type.
  Operation *materializeConstant(OpBuilder &builder, Attribute value, Type type,
                                 Location loc) override;

  // Registered hook to verify region arg attributes on operations.
  LogicalResult verifyRegionArgAttribute(mlir::Operation *op,
                                         unsigned regionIndex,
                                         unsigned argIndex,
                                         mlir::NamedAttribute attr) override;

  // Registered hook to verify an attribute from this dialect on operations.
  LogicalResult verifyOperationAttribute(mlir::Operation *op,
                                         mlir::NamedAttribute attr) override;

  // Parses a type registered to this dialect.
  Type parseType(DialectAsmParser &parser) const override;

  // Prints a type registered to this dialect.
  void printType(Type type, DialectAsmPrinter &os) const override;

  // Parses an attribute registered to this dialect.
  Attribute parseAttribute(DialectAsmParser &parser, Type type) const override;

  // Prints an attribute registered to this dialect.
  void printAttribute(Attribute attr, DialectAsmPrinter &os) const override;
};

class TokenType : public Type::TypeBase<TokenType, Type, TypeStorage> {
 public:
  using Base::Base;
};

// TODO(b/236017415): remove when we migrate to prefix accessor.
namespace accessor_dispatch {
template <typename OpT>
auto getReplicaGroups(OpT op, int)
    -> decltype(op.getReplicaGroups(), DenseIntElementsAttr{}) {
  return op.getReplicaGroups();
}
template <typename OpT>
auto getReplicaGroups(OpT op, char)
    -> decltype(op.replica_groups(), DenseIntElementsAttr{}) {
  return op.replica_groups();
}
}  // namespace accessor_dispatch

// Verifies replica groups attached to collective communication operations.
// If the attribute is not empty, it must be a rank 2 tensor, and each replica
// should appear exactly once. If `is_uniform_sized` is true, then we also check
// that each group is of the same size. If the operation has
// `use_global_device_ids` set, then replica group cannot be empty.
template <typename OpT>
LogicalResult verifyReplicaGroups(OpT op, bool isUniformSized) {
  DenseIntElementsAttr attr = accessor_dispatch::getReplicaGroups(op, 0);
  auto replicaGroupType = attr.getType().dyn_cast<RankedTensorType>();
  if (!replicaGroupType || replicaGroupType.getRank() != 2 ||
      !replicaGroupType.getElementType().isInteger(/*width=*/64))
    return op.emitOpError(
        "replica groups should be a rank 2 tensor of 64 bit integers");

  if (replicaGroupType.getShape().equals(ArrayRef<int64_t>{0, 0})) {
    if (op->hasAttr("use_global_device_ids") &&
        op->getAttr("use_global_device_ids")
            .template cast<BoolAttr>()
            .getValue()) {
      return op.emitOpError(
          "if `use_global_device_ids` is set, the replica groups cannot be "
          "empty");
    }
    return success();
  }

  int64_t maxReplicaIdSeen = 0;
  llvm::SmallSet<int64_t, 8> replicaSeen;
  for (int64_t id : attr.getValues<int64_t>()) {
    // Replica groups are stored in a 2D tensor. If the op supports non-uniform
    // groups, null replica IDs are stored as -1.
    if (id == -1) {
      if (isUniformSized) {
        return op.emitOpError("Invalid replica id -1");
      }
      continue;
    }

    if (!replicaSeen.insert(id).second) {
      return op.emitOpError("replica id #") << id << " seen more than once";
    }
    maxReplicaIdSeen = std::max(maxReplicaIdSeen, id);
  }

  for (int64_t id = 0; id <= maxReplicaIdSeen; id++) {
    if (!replicaSeen.contains(id)) {
      return op.emitOpError("replica id #")
             << id << " not seen in replica groups";
    }
  }
  return success();
}

// Verifies the source target pairs attached to collective permute.
LogicalResult verifyCollectivePermuteSourceTargetPairs(
    Operation *op, DenseIntElementsAttr attr);

LogicalResult verifyReduceScatter(Operation *op, TypeRange operandTypes,
                                  TypeRange resultTypes,
                                  uint64_t scatterDimension);

void printConvolutionDimensions(AsmPrinter &p, ConvDimensionNumbersAttr dnums);
void printConvolutionDimensions(AsmPrinter &p, Operation *,
                                ConvDimensionNumbersAttr dnums);
ParseResult parseConvolutionDimensions(AsmParser &parser,
                                       ConvDimensionNumbersAttr &dnums);

// Custom formatting for convolution window attributes.
void printWindowAttributes(OpAsmPrinter &p, Operation *op,
                           llvm::Optional<DenseIntElementsAttr> windowStrides,
                           llvm::Optional<DenseIntElementsAttr> padding,
                           llvm::Optional<DenseIntElementsAttr> lhsDilation,
                           llvm::Optional<DenseIntElementsAttr> rhsDilation,
                           llvm::Optional<DenseElementsAttr> windowReversal);

ParseResult parseWindowAttributes(OpAsmParser &parser,
                                  DenseIntElementsAttr &windowStrides,
                                  DenseIntElementsAttr &padding,
                                  DenseIntElementsAttr &lhsDilation,
                                  DenseIntElementsAttr &rhsDilation,
                                  DenseElementsAttr &windowReversal);

}  // end namespace stablehlo
}  // end namespace mlir

#define GET_OP_CLASSES
#include "stablehlo/dialect/StablehloOps.h.inc"

namespace mlir {
namespace stablehlo {

SortOp createSortOp(PatternRewriter *rewriter, const Location &loc,
                    const llvm::ArrayRef<Value> &operands,
                    const llvm::ArrayRef<Type> &elementTypes, int64_t dimension,
                    bool isStable, ComparisonDirection direction);

}  // end namespace stablehlo
}  // end namespace mlir

#endif  // STABLEHLO_DIALECT_STABLEHLO_OPS_H
