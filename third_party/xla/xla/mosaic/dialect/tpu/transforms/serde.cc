/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/mosaic/dialect/tpu/transforms/serde.h"

#include <cstdint>
#include <vector>

#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/mosaic/dialect/tpu/tpu_dialect.h"
#include "xla/mosaic/serde.h"

namespace mlir::tpu {

namespace {

constexpr StringRef kMangledDialect = "stable_mosaic.";
constexpr StringRef kVersionAttrName = "stable_mosaic.version";

using SerdeRuleType = jaxlib::mosaic::SerdeRuleType;

LogicalResult store_upgrade(Operation* op, int version, bool&) {
  if (version < 11) {
    op->setAttr("add", mlir::BoolAttr::get(op->getContext(), false));
  }
  return success();
}

LogicalResult store_downgrade(Operation* op, int version, bool&) {
  if (version < 11) {
    auto add_attr = op->getAttrOfType<BoolAttr>("add");
    if (!add_attr) {
      return op->emitError("Missing or invalid add attribute");
    }
    if (add_attr.getValue()) {
      return op->emitError(
          "Can only downgrade below version 11 when add is not set");
    }
    op->removeAttr("add");
  }
  return success();
}

LogicalResult dynamic_gather_upgrade(Operation* op, int version, bool&) {
  if (version < 5) {
    auto dimension_attr = op->getAttrOfType<IntegerAttr>("dimension");
    if (!dimension_attr || dimension_attr.getValue().getBitWidth() != 32) {
      return op->emitError("Missing or invalid dimension attribute");
    }
    const int32_t dimension = dimension_attr.getInt();
    op->removeAttr("dimension");
    op->setAttr("dimensions",
                DenseI32ArrayAttr::get(op->getContext(), {dimension}));
  }
  return success();
}

LogicalResult dynamic_gather_downgrade(Operation* op, int version, bool&) {
  if (version < 5) {
    auto dimensions_attr = op->getAttrOfType<DenseI32ArrayAttr>("dimensions");
    if (!dimensions_attr) {
      return op->emitError("Missing or invalid dimensions attribute");
    }
    const ArrayRef<int32_t> dimensions = dimensions_attr.asArrayRef();
    if (dimensions.size() != 1) {
      return op->emitError(
          "Can only downgrade below version 5 when a single dimension is "
          "specified.");
    }
    const int32_t dimension = dimensions.front();
    op->removeAttr("dimensions");
    op->setAttr("dimension",
                mlir::IntegerAttr::get(
                    mlir::IntegerType::get(op->getContext(), 32), dimension));
  }
  return success();
}

// Upgrades an operation from a version without subcore_id (v13) to HEAD (v14+)
// by delinearizing a single core_id into separate core_id and subcore_id.
// This runs during deserialization (`serialize = false`), where operations have
// already been demangled from `stable_mosaic.<op>` back to regular MLIR ops.
// Therefore, we emit standard unmangled constant and arithmetic operations.
LogicalResult delinearize_subcore(Operation* op, int core_id_seg,
                                  int subcore_id_seg) {
  CoreType core_type = GetCoreTypeOfParentOp(*op);
  if (core_type != CoreType::kScVectorSubcore) {
    return success();
  }
  auto segment_attr =
      op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
  if (!segment_attr) {
    return op->emitError("Missing operandSegmentSizes attribute");
  }
  SmallVector<int32_t> sizes(segment_attr.asArrayRef());
  if (sizes.size() <= core_id_seg || sizes.size() <= subcore_id_seg) {
    return op->emitError("Unexpected size of operandSegmentSizes");
  }
  if (sizes[core_id_seg] == 0) {
    return success();
  }
  int core_id_op_idx = 0;
  for (int i = 0; i < core_id_seg; ++i) {
    core_id_op_idx += sizes[i];
  }
  int subcore_id_op_idx = 0;
  for (int i = 0; i < subcore_id_seg; ++i) {
    subcore_id_op_idx += sizes[i];
  }
  Value core_id_val = op->getOperand(core_id_op_idx);

  // For version < 14, the only supported TPUs with SC vector subcores
  // have 16 subcores per core.
  const int64_t num_subcores = 16;

  OpBuilder builder(op);
  auto loc = op->getLoc();

  auto create_constant_op = [&](Location loc, int64_t val) -> Value {
    TypedAttr val_attr = builder.getI32IntegerAttr(val);
    return builder.create<arith::ConstantOp>(loc, val_attr);
  };

  Value c16 = create_constant_op(loc, num_subcores);
  Value new_core_id = builder.create<arith::DivUIOp>(loc, core_id_val, c16);
  Value new_subcore_id = builder.create<arith::RemUIOp>(loc, core_id_val, c16);

  op->setOperand(core_id_op_idx, new_core_id);
  op->insertOperands(subcore_id_op_idx, {new_subcore_id});
  sizes[subcore_id_seg] = 1;
  op->setAttr("operandSegmentSizes",
              DenseI32ArrayAttr::get(op->getContext(), sizes));
  return success();
}

// Downgrades an operation from HEAD (v14+) to a version without subcore_id
// (v13) by combining separate core_id and subcore_id into a single linearized
// core_id. This runs during serialization (`serialize = true`), where every
// operation in the module must be mangled with `stable_mosaic.`. Because MLIR's
// post-order walk has already advanced past the insertion point of new
// constants before downgrade rules execute, `RunSerde` will not visit or
// auto-mangle them. We therefore explicitly create pre-mangled
// `stable_mosaic.arith.*` operations.
LogicalResult linearize_subcore(Operation* op, int core_id_seg,
                                int subcore_id_seg) {
  CoreType core_type = GetCoreTypeOfParentOp(*op);
  if (core_type != CoreType::kScVectorSubcore) {
    return op->emitError(
        "subcore_id linearization requested for non-SC vector subcore type");
  }
  auto segment_attr =
      op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
  if (!segment_attr) {
    return op->emitError("Missing operandSegmentSizes attribute");
  }
  SmallVector<int32_t> sizes(segment_attr.asArrayRef());
  if (sizes.size() <= core_id_seg || sizes.size() <= subcore_id_seg) {
    return op->emitError("Unexpected size of operandSegmentSizes");
  }
  // This function should only be called if subcore_id is specified, in which
  // case core_id must be specified too.
  if (sizes[core_id_seg] == 0 || sizes[subcore_id_seg] == 0) {
    return op->emitError(
        "Expected both core_id and subcore_id to be present for linearization");
  }
  int core_id_op_idx = 0;
  for (int i = 0; i < core_id_seg; ++i) {
    core_id_op_idx += sizes[i];
  }
  int subcore_id_op_idx = 0;
  for (int i = 0; i < subcore_id_seg; ++i) {
    subcore_id_op_idx += sizes[i];
  }
  Value core_id_val = op->getOperand(core_id_op_idx);
  Value subcore_id_val = op->getOperand(subcore_id_op_idx);

  OpBuilder builder(op);
  auto loc = op->getLoc();
  const int64_t num_subcores = 16;

  auto create_constant_op = [&](Location loc, int64_t val) -> Value {
    TypedAttr val_attr = builder.getI32IntegerAttr(val);
    OperationState state(loc, "stable_mosaic.arith.constant");
    state.addTypes(builder.getI32Type());
    state.addAttribute("value", val_attr);
    Operation* const_op = builder.create(state);
    return const_op->getResult(0);
  };

  Value c16 = create_constant_op(loc, num_subcores);
  OperationState mul_state(loc, "stable_mosaic.arith.muli");
  mul_state.addTypes(builder.getI32Type());
  mul_state.addOperands({core_id_val, c16});
  Operation* mul_op = builder.create(mul_state);
  OperationState add_state(loc, "stable_mosaic.arith.addi");
  add_state.addTypes(builder.getI32Type());
  add_state.addOperands({mul_op->getResult(0), subcore_id_val});
  Operation* add_op = builder.create(add_state);
  Value new_core_id = add_op->getResult(0);

  op->setOperand(core_id_op_idx, new_core_id);
  op->eraseOperand(subcore_id_op_idx);
  // The caller will update operandSegmentSizes to sizes.drop_back().
  return success();
}

LogicalResult enqueue_dma_upgrade(Operation* op, int version, bool&) {
  // Added AttrSizedOperandSegments and core_id in version 2.
  if (version < 2) {
    if (op->getNumOperands() == 3) {  // Local DMA.
      op->setAttr(
          OpTrait::AttrSizedOperandSegments<
              EnqueueDMAOp>::getOperandSegmentSizeAttr(),
          mlir::DenseI32ArrayAttr::get(op->getContext(), {1, 0, 1, 1, 0, 0}));
    } else if (op->getNumOperands() == 5) {  // Remote DMA.
      op->setAttr(
          OpTrait::AttrSizedOperandSegments<
              EnqueueDMAOp>::getOperandSegmentSizeAttr(),
          mlir::DenseI32ArrayAttr::get(op->getContext(), {1, 1, 1, 1, 1, 0}));
    } else {
      return op->emitError("Unexpected operand count in tpu.enqueue_dma: ")
             << op->getNumOperands();
    }
  }
  if (version < 4) {
    op->setAttr("priority",
                mlir::IntegerAttr::get(
                    mlir::IntegerType::get(op->getContext(), 32), 0));
  }
  if (version < 14) {
    auto segment_attr = op->getAttrOfType<DenseI32ArrayAttr>(
        OpTrait::AttrSizedOperandSegments<
            EnqueueDMAOp>::getOperandSegmentSizeAttr());
    if (!segment_attr) {
      return op->emitError("Missing or invalid AttrSizedOperandSegments");
    }
    SmallVector<int32_t> new_sizes(segment_attr.asArrayRef());
    if (new_sizes.size() != 6) {
      return op->emitError(
          "Unexpected size of AttrSizedOperandSegments for enqueue_dma");
    }
    new_sizes.push_back(0);
    op->setAttr(OpTrait::AttrSizedOperandSegments<
                    EnqueueDMAOp>::getOperandSegmentSizeAttr(),
                DenseI32ArrayAttr::get(op->getContext(), new_sizes));
    return delinearize_subcore(op, 5, 6);
  }
  return success();
}

LogicalResult enqueue_dma_downgrade(Operation* op, int version, bool&) {
  if (version < 14) {
    auto segment_attr = op->getAttrOfType<DenseI32ArrayAttr>(
        OpTrait::AttrSizedOperandSegments<
            EnqueueDMAOp>::getOperandSegmentSizeAttr());
    if (!segment_attr) {
      return op->emitError("Missing or invalid AttrSizedOperandSegments");
    }
    const ArrayRef<int32_t> sizes = segment_attr.asArrayRef();
    if (sizes.size() != 7) {
      return op->emitError(
          "Unexpected size of AttrSizedOperandSegments for enqueue_dma");
    }
    if (sizes[6] != 0) {
      LogicalResult linearize = linearize_subcore(op, 5, 6);
      if (linearize.failed()) {
        return linearize;
      }
    }
    op->setAttr(OpTrait::AttrSizedOperandSegments<
                    EnqueueDMAOp>::getOperandSegmentSizeAttr(),
                DenseI32ArrayAttr::get(op->getContext(), sizes.drop_back()));
  }
  if (version < 12) {
    auto segment_attr = op->getAttrOfType<DenseI32ArrayAttr>(
        OpTrait::AttrSizedOperandSegments<
            EnqueueDMAOp>::getOperandSegmentSizeAttr());
    if (!segment_attr) {
      return op->emitError("Missing or invalid AttrSizedOperandSegments");
    }
    const ArrayRef<int32_t> sizes = segment_attr.asArrayRef();
    if (sizes.size() != 6) {
      return op->emitError(
          "Unexpected size of AttrSizedOperandSegments for enqueue_dma");
    }
    if (sizes[3] == 0) {
      return op->emitError(
          "Cannot downgrade enqueue_dma without target_semaphore to version "
          "< 12");
    }
  }
  if (version < 8) {
    auto ordering_attr = op->getAttrOfType<BoolAttr>("strict_ordering");
    if (ordering_attr != nullptr) {
      if (ordering_attr.getValue()) {
        return op->emitError(
            "Can only downgrade below version 8 when strict ordering is not "
            "set to True");
      }
      op->removeAttr("strict_ordering");
    }
  }
  if (version < 4) {
    op->removeAttr("priority");
  }
  if (version < 2) {
    return op->emitError("Downgrade to version ") << version << " unsupported";
  }
  return success();
}

LogicalResult iota_upgrade(Operation* op, int version, bool&) {
  if (version < 6) {
    auto dimension_attr = op->getAttrOfType<IntegerAttr>("dimension");
    if (!dimension_attr || dimension_attr.getValue().getBitWidth() != 32) {
      return op->emitError("Missing or invalid dimension attribute");
    }
    const int32_t dimension = dimension_attr.getInt();
    op->removeAttr("dimension");
    op->setAttr("dimensions",
                DenseI32ArrayAttr::get(op->getContext(), {dimension}));
  }
  return success();
}

LogicalResult iota_downgrade(Operation* op, int version, bool&) {
  if (version < 6) {
    auto dimensions_attr = op->getAttrOfType<DenseI32ArrayAttr>("dimensions");
    if (!dimensions_attr) {
      return op->emitError("Missing or invalid dimensions attribute");
    }
    const ArrayRef<int32_t> dimensions = dimensions_attr.asArrayRef();
    if (dimensions.size() != 1) {
      return op->emitError(
          "Can only downgrade below version 5 when a single dimension is "
          "specified.");
    }
    const int32_t dimension = dimensions.front();
    op->removeAttr("dimensions");
    op->setAttr("dimension",
                mlir::IntegerAttr::get(
                    mlir::IntegerType::get(op->getContext(), 32), dimension));
  }
  return success();
}

LogicalResult wait_dma2_upgrade(Operation* op, int version, bool&) {
  if (version < 7) {
    if (op->getNumOperands() != 3) {
      return op->emitError("Unexpected operand count in tpu.wait_dma2: ")
             << op->getNumOperands();
    }
    op->setAttr(
        OpTrait::AttrSizedOperandSegments<
            EnqueueDMAOp>::getOperandSegmentSizeAttr(),
        mlir::DenseI32ArrayAttr::get(op->getContext(), {1, 1, 1, 0, 0}));
  }
  return success();
}

LogicalResult wait_dma2_downgrade(Operation* op, int version, bool&) {
  if (version < 7) {
    auto operands = op->getAttrOfType<mlir::DenseI32ArrayAttr>(
        OpTrait::AttrSizedOperandSegments<
            EnqueueDMAOp>::getOperandSegmentSizeAttr());
    if (!operands || operands.size() != 5) {
      return op->emitError("Missing or invalid AttrSizedOperandSegments");
    }
    if (operands[3] || operands[4]) {
      return op->emitError("Downgrade to version ")
             << version << " impossible: device_id and/or core_id is set";
    }
    op->removeAttr(OpTrait::AttrSizedOperandSegments<
                   EnqueueDMAOp>::getOperandSegmentSizeAttr());
  }
  if (version < 3) {
    return op->emitError("Downgrade to version ") << version << " unsupported";
  }
  return success();
}

LogicalResult wait_dma_upgrade(Operation* op, int version, bool&) {
  if (version < 14) {
    auto segment_attr = op->getAttrOfType<DenseI32ArrayAttr>(
        OpTrait::AttrSizedOperandSegments<
            EnqueueDMAOp>::getOperandSegmentSizeAttr());
    if (!segment_attr) {
      return op->emitError("Missing or invalid AttrSizedOperandSegments");
    }
    SmallVector<int32_t> new_sizes(segment_attr.asArrayRef());
    if (new_sizes.size() != 6) {
      return op->emitError(
          "Unexpected size of AttrSizedOperandSegments for wait_dma");
    }
    new_sizes.push_back(0);
    op->setAttr(OpTrait::AttrSizedOperandSegments<
                    EnqueueDMAOp>::getOperandSegmentSizeAttr(),
                DenseI32ArrayAttr::get(op->getContext(), new_sizes));
    return delinearize_subcore(op, 5, 6);
  }
  return success();
}

LogicalResult wait_dma_downgrade(Operation* op, int version, bool&) {
  if (version < 14) {
    auto segment_attr = op->getAttrOfType<DenseI32ArrayAttr>(
        OpTrait::AttrSizedOperandSegments<
            EnqueueDMAOp>::getOperandSegmentSizeAttr());
    if (!segment_attr) {
      return op->emitError("Missing or invalid AttrSizedOperandSegments");
    }
    const ArrayRef<int32_t> sizes = segment_attr.asArrayRef();
    if (sizes.size() != 7) {
      return op->emitError(
          "Unexpected size of AttrSizedOperandSegments for wait_dma");
    }
    if (sizes[6] != 0) {
      LogicalResult linearize = linearize_subcore(op, 5, 6);
      if (linearize.failed()) {
        return linearize;
      }
    }
    op->setAttr(OpTrait::AttrSizedOperandSegments<
                    EnqueueDMAOp>::getOperandSegmentSizeAttr(),
                DenseI32ArrayAttr::get(op->getContext(), sizes.drop_back()));
  }
  return success();
}

LogicalResult semaphore_signal_upgrade(Operation* op, int version, bool&) {
  // Added AttrSizedOperandSegments and core_id in version 2.
  if (version < 2) {
    if (op->getNumOperands() == 2) {  // Local signal.
      op->setAttr(OpTrait::AttrSizedOperandSegments<
                      EnqueueDMAOp>::getOperandSegmentSizeAttr(),
                  mlir::DenseI32ArrayAttr::get(op->getContext(), {1, 1, 0, 0}));
    } else if (op->getNumOperands() == 3) {  // Remote signal.
      op->setAttr(OpTrait::AttrSizedOperandSegments<
                      EnqueueDMAOp>::getOperandSegmentSizeAttr(),
                  mlir::DenseI32ArrayAttr::get(op->getContext(), {1, 1, 1, 0}));
    } else {
      return op->emitError("Unexpected operand count in tpu.semaphore_signal");
    }
  }
  if (version < 14) {
    auto segment_attr = op->getAttrOfType<DenseI32ArrayAttr>(
        OpTrait::AttrSizedOperandSegments<
            EnqueueDMAOp>::getOperandSegmentSizeAttr());
    if (!segment_attr) {
      return op->emitError("Missing or invalid AttrSizedOperandSegments");
    }
    SmallVector<int32_t> new_sizes(segment_attr.asArrayRef());
    if (new_sizes.size() != 4) {
      return op->emitError(
          "Unexpected size of AttrSizedOperandSegments for semaphore_signal.");
    }
    new_sizes.push_back(0);
    op->setAttr(OpTrait::AttrSizedOperandSegments<
                    EnqueueDMAOp>::getOperandSegmentSizeAttr(),
                DenseI32ArrayAttr::get(op->getContext(), new_sizes));
    return delinearize_subcore(op, 3, 4);
  }
  return success();
}

LogicalResult semaphore_signal_downgrade(Operation* op, int version, bool&) {
  if (version < 14) {
    auto segment_attr = op->getAttrOfType<DenseI32ArrayAttr>(
        OpTrait::AttrSizedOperandSegments<
            EnqueueDMAOp>::getOperandSegmentSizeAttr());
    if (!segment_attr) {
      return op->emitError("Missing or invalid AttrSizedOperandSegments");
    }
    const ArrayRef<int32_t> sizes = segment_attr.asArrayRef();
    if (sizes.size() != 5) {
      return op->emitError(
          "Unexpected size of AttrSizedOperandSegments for semaphore_signal");
    }
    if (sizes[4] != 0) {
      LogicalResult linearize = linearize_subcore(op, 3, 4);
      if (linearize.failed()) {
        return linearize;
      }
    }
    op->setAttr(OpTrait::AttrSizedOperandSegments<
                    EnqueueDMAOp>::getOperandSegmentSizeAttr(),
                DenseI32ArrayAttr::get(op->getContext(), sizes.drop_back()));
  }
  if (version < 2) {
    auto operands = op->getAttrOfType<mlir::DenseI32ArrayAttr>(
        OpTrait::AttrSizedOperandSegments<
            EnqueueDMAOp>::getOperandSegmentSizeAttr());
    if (!operands || operands.size() != 4) {
      return op->emitError("Missing or invalid AttrSizedOperandSegments");
    }
    if (operands[3]) {
      return op->emitError("Downgrade to version ")
             << version << " impossible: core_id is set";
    }
    op->removeAttr(OpTrait::AttrSizedOperandSegments<
                   EnqueueDMAOp>::getOperandSegmentSizeAttr());
  }
  return success();
}

LogicalResult vector_multi_dim_reduce_upgrade(Operation* op, int version,
                                              bool&) {
  // Changed reductions_dims from ArrayAttr of IntegerAttrs to DenseI64ArrayAttr
  // in version 3.
  if (version < 3) {
    Attribute reduction_dims_attr = op->getAttr("reduction_dims");
    if (!reduction_dims_attr) {
      return op->emitError("Missing reduction_dims attribute");
    }
    ArrayAttr reduction_dims_array = dyn_cast<ArrayAttr>(reduction_dims_attr);
    if (!reduction_dims_array) {
      return op->emitOpError("reduction_dims attribute is not an ArrayAttr");
    }
    std::vector<int64_t> reduction_dims;
    reduction_dims.reserve(reduction_dims_array.size());
    for (Attribute reduction_dim : reduction_dims_array) {
      IntegerAttr reduction_dim_attr = dyn_cast<IntegerAttr>(reduction_dim);
      if (!reduction_dim_attr) {
        return op->emitOpError(
            "reduction_dims attribute contains a non-IntegerAttr");
      }
      reduction_dims.push_back(reduction_dim_attr.getInt());
    }
    op->setAttr("reduction_dims",
                DenseI64ArrayAttr::get(op->getContext(), reduction_dims));
  }
  return success();
}

LogicalResult vector_multi_dim_reduce_downgrade(Operation* op, int version,
                                                bool&) {
  if (version < 3) {
    return op->emitOpError("Downgrade to version ")
           << version << " unsupported";
  }
  return success();
}

LogicalResult arith_constant_upgrade(Operation* op, int version, bool&) {
  return success();
}

LogicalResult arith_constant_downgrade(Operation* op, int version, bool&) {
  // The encoding of the boolean dense elements attr changed in version 10.
  // Before, it used to be a bitpacked value, but a splat true was represented
  // by a single byte with all bits set to 1. After the change, a splat true is
  // represented by a byte with the LSB set to 1. The change happens to be
  // backwards-compatible, but not forwards-compatible, which is why we need to
  // have the downgrade rule only.
  if (version < 10) {
    auto value_attr = op->getAttrOfType<DenseElementsAttr>("value");
    // Only i1 dense elements attrs are affected.
    if (!value_attr || value_attr.getElementType() !=
                           mlir::IntegerType::get(op->getContext(), 1)) {
      return success();
    }
    if (!value_attr.isSplat()) {
      return op->emitOpError("Downgrade to version ")
             << version << " not implemented: value is not a splat";
    }
    char new_splat = value_attr.getSplatValue<bool>() ? 0xff : 0x00;
    op->setAttr("value", mlir::SplatElementsAttr::getFromRawBuffer(
                             value_attr.getType(), {new_splat}));
  }
  return success();
}

LogicalResult reinterpret_cast_upgrade(Operation* op, int version, bool&) {
  if (version < 15) {
    bool dynamic_offset = op->getNumOperands() > 1;
    op->setAttr("operandSegmentSizes",
                mlir::DenseI32ArrayAttr::get(op->getContext(),
                                             {1, dynamic_offset ? 1 : 0, 0}));
  }
  return success();
}

LogicalResult reinterpret_cast_downgrade(Operation* op, int version, bool&) {
  if (version < 15) {
    auto operand_segment_sizes =
        op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
    if (operand_segment_sizes && operand_segment_sizes.asArrayRef()[2] > 0) {
      return op->emitOpError(
          "Can only downgrade below version 15 when dynamic_sizes is empty");
    }
    op->removeAttr("operandSegmentSizes");
  }
  if (version < 12) {
    if (op->getNumOperands() == 2) {
      return op->emitOpError(
          "Can only downgrade below version 12 when dynamic_offset is not set");
    }
  }
  return success();
}

LogicalResult matmul_upgrade(Operation* op, int version, bool&) {
  if (version < 13) {
    op->setAttr("transpose_lhs_hint",
                mlir::BoolAttr::get(op->getContext(), false));
  }
  return success();
}

LogicalResult matmul_downgrade(Operation* op, int version, bool&) {
  if (version < 13) {
    auto transpose_lhs_hint_attr =
        op->getAttrOfType<BoolAttr>("transpose_lhs_hint");
    if (!transpose_lhs_hint_attr) {
      return op->emitOpError("Missing transpose_lhs_hint attribute");
    }
    if (transpose_lhs_hint_attr.getValue()) {
      return op->emitOpError(
          "Can only downgrade below version 12 when transpose_lhs_hint is "
          "False ");
    }
    op->removeAttr("transpose_lhs_hint");
  }
  return success();
}

const llvm::StringMap<SerdeRuleType>& upgrade_rules() {
  static auto rules = new llvm::StringMap<SerdeRuleType>{
      {EnqueueDMAOp::getOperationName(), enqueue_dma_upgrade},
      {WaitDMA2Op::getOperationName(), wait_dma2_upgrade},
      {WaitDMAOp::getOperationName(), wait_dma_upgrade},
      {DynamicGatherOp::getOperationName(), dynamic_gather_upgrade},
      {IotaOp::getOperationName(), iota_upgrade},
      {SemaphoreSignalOp::getOperationName(), semaphore_signal_upgrade},
      {vector::MultiDimReductionOp::getOperationName(),
       vector_multi_dim_reduce_upgrade},
      {StoreOp::getOperationName(), store_upgrade},
      {arith::ConstantOp::getOperationName(), arith_constant_upgrade},
      {ReinterpretCastOp::getOperationName(), reinterpret_cast_upgrade},
      {MatmulOp::getOperationName(), matmul_upgrade},
  };
  return *rules;
}

const llvm::StringMap<SerdeRuleType>& downgrade_rules() {
  static auto rules = new llvm::StringMap<SerdeRuleType>{
      {EnqueueDMAOp::getOperationName(), enqueue_dma_downgrade},
      {WaitDMA2Op::getOperationName(), wait_dma2_downgrade},
      {WaitDMAOp::getOperationName(), wait_dma_downgrade},
      {DynamicGatherOp::getOperationName(), dynamic_gather_downgrade},
      {IotaOp::getOperationName(), iota_downgrade},
      {SemaphoreSignalOp::getOperationName(), semaphore_signal_downgrade},
      {StoreOp::getOperationName(), store_downgrade},
      {vector::MultiDimReductionOp::getOperationName(),
       vector_multi_dim_reduce_downgrade},
      {arith::ConstantOp::getOperationName(), arith_constant_downgrade},
      {ReinterpretCastOp::getOperationName(), reinterpret_cast_downgrade},
      {MatmulOp::getOperationName(), matmul_downgrade},
  };
  return *rules;
}

}  // namespace

void MosaicSerdePass::runOnOperation() {
  ModuleOp module = getOperation();
  if (!serialize.hasValue()) {
    module.emitError("serialize option must be specified");
    return signalPassFailure();
  }
  int serialize_version = -1;
  if (serialize) {
    serialize_version = target_version.hasValue() ? target_version : kVersion;
  }
  if (failed(jaxlib::mosaic::RunSerde(
          module, upgrade_rules(), downgrade_rules(), serialize,
          {.dialect_prefix = kMangledDialect,
           .highest_version = kVersion,
           .version_attr_name = kVersionAttrName,
           .serialize_version = serialize_version},
          /*keep_version_attr=*/keep_version_attr))) {
    signalPassFailure();
  }
}

}  // namespace mlir::tpu
