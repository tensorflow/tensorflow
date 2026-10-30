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

// This file defines the operations used in the MLIR TensorFlow Lite dialect.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_IR_TFL_OPS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_IR_TFL_OPS_H_

#include <cstdint>
#include <tuple>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/TypeSupport.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"  // from @llvm-project
#include "mlir/Interfaces/InferTypeOpInterface.h"  // from @llvm-project
#include "mlir/Interfaces/LoopLikeInterface.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops_dialect.h.inc"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops_enums.h.inc"
#include "tensorflow/compiler/mlir/lite/quantization/common/quantization_lib/quantization_utils.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/quantization/ir/QuantOps.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/compiler/mlir/lite/utils/utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_traits.h"
#define GET_ATTRDEF_CLASSES
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops_attrdefs.h.inc"

namespace mlir {
namespace TFL {

typedef TFLDialect TensorFlowLiteDialect;

// The Control type is a token-like value that models control dependencies
class ControlType : public Type::TypeBase<ControlType, Type, TypeStorage> {
 public:
  using Base::Base;
  static constexpr StringLiteral name = "tfl.control";
};

namespace detail {
struct BlockwiseQuantTypeStorage : public TypeStorage {
  using KeyTy =
      std::tuple<Type, Type, ArrayRef<int64_t>, ElementsAttr, ElementsAttr>;

  BlockwiseQuantTypeStorage(Type storage_type, Type expressed_type,
                            ArrayRef<int64_t> block_shape, ElementsAttr scales,
                            ElementsAttr zero_points)
      : storage_type(storage_type),
        expressed_type(expressed_type),
        block_shape(block_shape),
        scales(scales),
        zero_points(zero_points) {}

  bool operator==(const KeyTy& key) const {
    return key == KeyTy(storage_type, expressed_type, block_shape, scales,
                        zero_points);
  }

  static llvm::hash_code hashKey(const KeyTy& key) {
    return llvm::hash_combine(std::get<0>(key), std::get<1>(key),
                              std::get<2>(key), std::get<3>(key),
                              std::get<4>(key));
  }

  static BlockwiseQuantTypeStorage* construct(TypeStorageAllocator& allocator,
                                              const KeyTy& key) {
    return new (allocator.allocate<BlockwiseQuantTypeStorage>())
        BlockwiseQuantTypeStorage(std::get<0>(key), std::get<1>(key),
                                  allocator.copyInto(std::get<2>(key)),
                                  std::get<3>(key), std::get<4>(key));
  }

  Type storage_type;
  Type expressed_type;
  ArrayRef<int64_t> block_shape;
  ElementsAttr scales;
  ElementsAttr zero_points;
};
}  // namespace detail

// Element type of a tensor quantized with one set of parameters per block,
// where the blocks tile the tensor according to `block_shape`. This is the
// MLIR counterpart of `BlockwiseQuantization` with `block_shape` in the TFLite
// schema:
//
//   real = scale[b] * (q - zero_point[b]),  b = floor(index / block_shape)
//
// `block_shape` has one entry per dimension of the quantized tensor, and
// `scales` / `zero_points` are tensors of the same rank shaped like the block
// grid `ceil(tensor_shape / block_shape)`. A parameter tensor may have extent 1
// on any axis, in which case it broadcasts across the blocks of that axis.
// `zero_points` is optional and defaults to 0.
//
// Unlike `!quant.uniform`, the parameters are held as `ElementsAttr`s, so they
// may be backed by resource blobs, and their element types are independent of
// the storage and expressed types (e.g. `f8E8M0FNU` scales or `f32` zero
// points).
//
// Syntax:
//   !tfl.blockwise_quant<i4 -> f32, block_shape = [1, 32],
//                        scales = dense<...> : tensor<64x2xf32>,
//                        zero_points = dense<...> : tensor<64x2xi8>>
class BlockwiseQuantType
    : public Type::TypeBase<BlockwiseQuantType, Type,
                            detail::BlockwiseQuantTypeStorage> {
 public:
  using Base::Base;
  using Base::getChecked;
  static constexpr StringLiteral name = "tfl.blockwise_quant";

  static BlockwiseQuantType get(Type storage_type, Type expressed_type,
                                ArrayRef<int64_t> block_shape,
                                ElementsAttr scales,
                                ElementsAttr zero_points = nullptr) {
    return Base::get(storage_type.getContext(), storage_type, expressed_type,
                     block_shape, scales, zero_points);
  }

  static LogicalResult verifyInvariants(
      function_ref<InFlightDiagnostic()> emit_error, Type storage_type,
      Type expressed_type, ArrayRef<int64_t> block_shape, ElementsAttr scales,
      ElementsAttr zero_points);

  // The integer type the quantized values are stored as.
  Type getStorageType() const { return getImpl()->storage_type; }
  // The floating point type the values dequantize to.
  Type getExpressedType() const { return getImpl()->expressed_type; }
  ArrayRef<int64_t> getBlockShape() const { return getImpl()->block_shape; }
  ElementsAttr getScales() const { return getImpl()->scales; }
  // Null when the zero point is 0 everywhere.
  ElementsAttr getZeroPoints() const { return getImpl()->zero_points; }

  bool isSigned() const { return !getStorageType().isUnsignedInteger(); }
  unsigned getStorageTypeIntegralWidth() const {
    return getStorageType().getIntOrFloatBitWidth();
  }

  // Verifies that this type can be the element type of a tensor of `shape`,
  // i.e. that `block_shape` has one entry per dimension and the parameter
  // tensors are shaped like the block grid (or broadcast along it).
  LogicalResult verifyForShape(function_ref<InFlightDiagnostic()> emit_error,
                               ArrayRef<int64_t> shape) const;
};

// Discardable attribute naming a non-standard quantization contract that the
// operand types cannot express on their own. Read by the `quant_spec` derived
// attribute of `tfl.fully_connected`; see `FullyConnectedOptions.quant_spec` in
// schema.fbs.
inline constexpr StringLiteral kQuantSpecAttrName = "tfl.quant_spec";

#include "tensorflow/compiler/mlir/lite/ir/tfl_ops_interface.h.inc"

}  // end namespace TFL
}  // end namespace mlir

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h.inc"

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_IR_TFL_OPS_H_
