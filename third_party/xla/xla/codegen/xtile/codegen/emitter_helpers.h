/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_CODEGEN_XTILE_CODEGEN_EMITTER_HELPERS_H_
#define XLA_CODEGEN_XTILE_CODEGEN_EMITTER_HELPERS_H_

#include <cstdint>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/tiling/tiled_hlo_instruction.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape_util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::xtile {

using TensorValue = mlir::TypedValue<mlir::RankedTensorType>;
static constexpr auto kTritonDivisibilityAttr = "tt.divisibility";

// Returns a string representation of the given MLIR entity.
template <typename T>
std::string MlirToString(T&& value) {
  std::string result;
  llvm::raw_string_ostream os(result);
  value.print(os);
  return result;
}

// Constructs and holds information needed to construct a tile. This information
// is propagated to Extract/Insert ops to use them to load and store the correct
// tiles.
class TileInfo {
 public:
  static absl::StatusOr<TileInfo> Construct(
      mlir::ImplicitLocOpBuilder& b, mlir::Value pid,
      mlir::ValueRange runtime_values, const TiledHloInstruction& tiled_hlo);

  // Tile offsets. Its size is equal to the rank of the output shape.
  inline mlir::ValueRange offsets() const { return offsets_; }

  // Tile strides. Its size is equal to the rank of the output shape.
  inline mlir::ArrayRef<int64_t> tile_strides() const { return tile_strides_; }

  // The original shape of the tensor.
  inline mlir::ArrayRef<int64_t> original_shape() const {
    return original_shape_;
  }

  // Tile sizes after padding to a power of 2 (Triton requirement).
  inline mlir::ArrayRef<int64_t> padded_tile_sizes() const {
    return padded_tile_sizes_;
  }

  // The layout of the tensor in minor-to-major order.
  inline const llvm::SmallVector<int64_t>& minor_to_major_layout() const {
    return minor_to_major_layout_;
  }

  // The storage type of the tensor. This could be different from the element
  // type. e.g. predicates are stored as i8 instead of i1.
  mlir::Type storage_type() const { return storage_type_; }

 private:
  llvm::SmallVector<mlir::Value> offsets_;
  llvm::SmallVector<int64_t> tile_strides_;
  llvm::SmallVector<int64_t> original_shape_;
  llvm::SmallVector<int64_t> padded_tile_sizes_;
  llvm::SmallVector<int64_t> minor_to_major_layout_;
  mlir::Type storage_type_;

  inline TileInfo(llvm::SmallVector<mlir::Value> offsets,
                  llvm::SmallVector<int64_t> tile_strides,
                  llvm::SmallVector<int64_t> original_shape,
                  llvm::SmallVector<int64_t> padded_tile_sizes,
                  llvm::SmallVector<int64_t> minor_to_major_layout,
                  mlir::Type storage_type)
      : offsets_(std::move(offsets)),
        tile_strides_(std::move(tile_strides)),
        original_shape_(std::move(original_shape)),
        padded_tile_sizes_(std::move(padded_tile_sizes)),
        minor_to_major_layout_(std::move(minor_to_major_layout)),
        storage_type_(std::move(storage_type)) {}
};

// Triton requires that all block dimensions are a power of 2.
// TODO(b/353484968): Delete this function once we have constraints to only
// propagate tile sizes that are a power of 2.
llvm::SmallVector<int64_t> GetPaddedTileSizes(
    llvm::ArrayRef<int64_t> tile_sizes);

// XLA -> MLIR type conversions.
absl::StatusOr<mlir::Type> PrimitiveTypeToMlirType(
    mlir::ImplicitLocOpBuilder& b, PrimitiveType t);

// MLIR type -> XLA type conversions.
absl::StatusOr<PrimitiveType> GetPrimitiveType(mlir::Type t);

mlir::Type StorageType(mlir::Type t);

// Get the value of the scalar constant's literal in a C++ ty˝pe.
template <typename T>
T ScalarConstantValue(const HloInstruction& instr, PrimitiveType dst_type) {
  CHECK_EQ(instr.opcode(), HloOpcode::kConstant);
  CHECK(ShapeUtil::IsEffectiveScalar(instr.shape()));
  absl::StatusOr<Literal> converted = instr.literal().Convert(dst_type);
  CHECK_OK(converted.status());
  return converted.value().GetFirstElement<T>();
}

// Create a scalar constant.
template <typename T>
mlir::Value CreateConst(mlir::ImplicitLocOpBuilder& b, mlir::Type type,
                        T value) {
  if (mlir::isa<mlir::IntegerType>(type)) {
    return b.create<mlir::arith::ConstantOp>(b.getIntegerAttr(type, value));
  }

  if (mlir::isa<mlir::IndexType>(type)) {
    return b.create<mlir::arith::ConstantOp>(b.getIndexAttr(value));
  }

  if (mlir::isa<mlir::FloatType>(type)) {
    return b.create<mlir::arith::ConstantOp>(
        b.getFloatAttr(type, static_cast<double>(value)));
  }
  LOG(FATAL) << "Constant type not supported: " << llvm_ir::DumpToString(type);
}

// Create a tensor constant.
template <typename T>
mlir::TypedValue<mlir::RankedTensorType> CreateConst(
    mlir::ImplicitLocOpBuilder& b, mlir::Type type, T value,
    llvm::ArrayRef<int64_t> shape) {
  auto tensor_type = mlir::RankedTensorType::get(shape, type);
  if (auto int_type = mlir::dyn_cast<mlir::IntegerType>(type)) {
    mlir::Value result =
        b.create<mlir::arith::ConstantOp>(mlir::DenseElementsAttr::get(
            tensor_type,
            mlir::APInt(int_type.getIntOrFloatBitWidth(), value,
                        /*isSigned=*/false, /*implicitTrunc=*/true)));
    return mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(result);
  }
  if (auto float_type = mlir::dyn_cast<mlir::FloatType>(type)) {
    mlir::Value result =
        b.create<mlir::arith::ConstantOp>(mlir::DenseElementsAttr::get(
            tensor_type, b.getFloatAttr(type, static_cast<double>(value))));
    return mlir::cast<mlir::TypedValue<mlir::RankedTensorType>>(result);
  }
  LOG(FATAL) << "Constant type not supported: " << llvm_ir::DumpToString(type);
}

// Create a constant of the same shape as `like` but with a new type and value.
template <typename T>
mlir::Value ConstLike(mlir::ImplicitLocOpBuilder& b, mlir::Value like,
                      T new_value) {
  if (auto src_shaped_ty = mlir::dyn_cast<mlir::ShapedType>(like.getType())) {
    mlir::Type src_ty = src_shaped_ty.getElementType();
    return CreateConst(b, src_ty, new_value, src_shaped_ty.getShape());
  }
  return CreateConst(b, like.getType(), new_value);
}

inline mlir::Value ZerosLike(mlir::ImplicitLocOpBuilder& b, mlir::Value x) {
  return ConstLike(b, x, 0);
}

bool IsFp8Type(mlir::Type t);

// Triton type conversions.
mlir::Value Cast(mlir::ImplicitLocOpBuilder& b, mlir::Value value,
                 mlir::Type dst_element_ty);

// Emits a scalar constant.
absl::StatusOr<mlir::TypedValue<mlir::RankedTensorType>> EmitConstant(
    mlir::ImplicitLocOpBuilder& b, const HloInstruction& constant);

absl::StatusOr<mlir::Value> EmitElementwise(mlir::ImplicitLocOpBuilder& b,
                                            const HloInstruction& hlo,
                                            mlir::ValueRange inputs);

mlir::Value Bitcast(mlir::ImplicitLocOpBuilder& b, mlir::Value value,
                    mlir::Type type);

// Emits an xtile::ExtractTileOp for the given tile info and argument.
TensorValue EmitParameterExtract(mlir::ImplicitLocOpBuilder& b,
                                 const TileInfo& tile_info, mlir::Value arg);

// Emits a sequence of HLO instructions within a specific scope.
//
// This function traverses the provided `hlo_instructions` in a
// defined-before-use order and emits the corresponding MLIR operations using
// the given `mlir::ImplicitLocOpBuilder`. It uses `emitted_values` to look up
// already emitted results for instructions, typically parameters or results
// from outer scopes. New results are added to the `emitted_values` map.
//
// Example usage within [EmitReduce] includes using it to emit the body of the
// `HloInstruction::to_apply` computation.
absl::StatusOr<TensorValue> EmitScope(
    mlir::ImplicitLocOpBuilder& b,
    absl::Span<const HloInstruction* const> instructions,
    absl::flat_hash_map<const HloInstruction*, TensorValue>& values);

// Same as HLO BroadcastInDims. The sorted indices in `dims` specify the
// mapping of the input dimensions to the output dimensions.
TensorValue BroadcastInDims(mlir::ImplicitLocOpBuilder& b, TensorValue value,
                            ::mlir::ArrayRef<int64_t> output_shape,
                            ::mlir::ArrayRef<int64_t> dims);

TensorValue Splat(mlir::ImplicitLocOpBuilder& b, ::mlir::Value value,
                  ::mlir::ArrayRef<int64_t> output_shape);

// Returns a named attribute for divisibility of triton pointer function
// arguments.
inline mlir::NamedAttribute GetDivisibilityAttr(mlir::ImplicitLocOpBuilder& b) {
  return b.getNamedAttr(kTritonDivisibilityAttr,
                        b.getIntegerAttr(b.getI32Type(), 16));
}

mlir::Value UnsignedIntegerToSignlessInteger(mlir::OpBuilder& builder,
                                             mlir::Value value);

// Function to get the permutation vector from a MemRefType.
// The motivation for extracting it from getStridesAndOffset vs directly from
// xtile.layout is that when we fold memrefs (such as in a transpose) it
// will have a generic strided layout that does not directly encode the
// permutation.
absl::StatusOr<llvm::SmallVector<int64_t>> GetPermutationMinorToMajor(
    mlir::MemRefType memref);

}  // namespace xla::xtile

#endif  // XLA_CODEGEN_XTILE_CODEGEN_EMITTER_HELPERS_H_
