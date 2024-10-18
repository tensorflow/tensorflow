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

#ifndef XLA_SERVICE_GPU_FUSIONS_TRITON_EMITTER_HELPERS_H_
#define XLA_SERVICE_GPU_FUSIONS_TRITON_EMITTER_HELPERS_H_

#include <cstdint>
#include <variant>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/literal.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"
#include "tsl/platform/status.h"

namespace xla::gpu::triton {

// This is a wrapper around mlir::Value that can hold either a scalar or a
// non-0D tensor. An attempt to use this class with 0D tensors will CHECK-fail
// because 0D tensors are not supported by Triton.
class ScalarOrTensor {
 public:
  ScalarOrTensor() = default;

  // Wraps the given value in a ScalarOrTensor. CHECK-fails if the
  // value is a 0D tensor, because Triton does not support 0D tensors.
  explicit ScalarOrTensor(mlir::Value value);

  bool IsScalar() const { return std::holds_alternative<ScalarValue>(value_); }
  bool IsTensor() const { return std::holds_alternative<TensorValue>(value_); }

  mlir::Value UnwrapScalar() {
    CHECK(IsScalar());
    return std::get<ScalarValue>(value_).scalar_value;
  }

  mlir::Value UnwrapTensor() {
    CHECK(IsTensor());
    return std::get<TensorValue>(value_).tensor_value;
  }

  // Returns the underlying value regardless of whether it is a scalar or a
  // tensor. Only call this method in contexts where the consumer of the result
  // both needs to use an `mlir::Value` and functions identically for scalars
  // and tensors. In other cases, prefer to use the `UnwrapScalar` or
  // `UnwrapTensor` methods.
  mlir::Value UnwrapUnsafe() {
    if (auto* scalar = std::get_if<ScalarValue>(&value_)) {
      return scalar->scalar_value;
    }
    return std::get<TensorValue>(value_).tensor_value;
  }

  mlir::Type Type() { return UnwrapUnsafe().getType(); }

 private:
  struct ScalarValue {
    mlir::Value scalar_value;
  };

  struct TensorValue {
    mlir::Value tensor_value;
  };

  std::variant<ScalarValue, TensorValue> value_;
};

// Triton requires that all block dimensions are a power of 2.
// TODO(b/353484968): Delete this function once we have constraints to only
// propagate tile sizes that are a power of 2.
llvm::SmallVector<int64_t> GetPaddedTileSizes(
    llvm::ArrayRef<int64_t> tile_sizes);

// XLA -> Triton type conversions.
absl::StatusOr<mlir::Type> TritonType(mlir::OpBuilder b, PrimitiveType t);

mlir::Type StorageType(mlir::OpBuilder b, mlir::Type t);

// Get the value of the scalar constant's literal in a C++ type.
template <typename T>
T ScalarConstantValue(const HloInstruction& instr, PrimitiveType dst_type) {
  CHECK_EQ(instr.opcode(), HloOpcode::kConstant);
  CHECK(ShapeUtil::IsEffectiveScalar(instr.shape()));
  absl::StatusOr<Literal> converted = instr.literal().Convert(dst_type);
  TF_CHECK_OK(converted.status());
  return converted.value().GetFirstElement<T>();
}

// Create a scalar constant.
template <typename T>
ScalarOrTensor CreateConst(mlir::ImplicitLocOpBuilder b, mlir::Type type,
                           T value) {
  if (mlir::isa<mlir::IntegerType>(type)) {
    auto result =
        b.create<mlir::arith::ConstantOp>(b.getIntegerAttr(type, value));
    return ScalarOrTensor(result);
  }
  if (mlir::isa<mlir::FloatType>(type)) {
    auto result = b.create<mlir::arith::ConstantOp>(
        b.getFloatAttr(type, static_cast<double>(value)));
    return ScalarOrTensor(result);
  }
  LOG(FATAL) << "Constant type not supported: " << llvm_ir::DumpToString(type);
}

// Create a tensor constant.
template <typename T>
ScalarOrTensor CreateConst(mlir::ImplicitLocOpBuilder& b, mlir::Type type,
                           T value, llvm::ArrayRef<int64_t> shape) {
  if (shape.empty()) {
    return CreateConst<T>(b, type, value);
  }
  auto tensor_type = mlir::RankedTensorType::get(shape, type);
  if (auto int_type = mlir::dyn_cast<mlir::IntegerType>(type)) {
    auto result =
        b.create<mlir::arith::ConstantOp>(mlir::DenseElementsAttr::get(
            tensor_type, mlir::APInt(int_type.getIntOrFloatBitWidth(), value)));
    return ScalarOrTensor(result);
  }
  if (auto float_type = mlir::dyn_cast<mlir::FloatType>(type)) {
    auto result =
        b.create<mlir::arith::ConstantOp>(mlir::DenseElementsAttr::get(
            tensor_type, b.getFloatAttr(type, static_cast<double>(value))));
    return ScalarOrTensor(result);
  }
  LOG(FATAL) << "Constant type not supported: " << llvm_ir::DumpToString(type);
}

// Create a constant of the same shape as `like` but with a new type and value.
template <typename T>
mlir::Value ConstLike(mlir::ImplicitLocOpBuilder& b, mlir::Value like,
                      T new_value) {
  if (auto src_shaped_ty = mlir::dyn_cast<mlir::ShapedType>(like.getType())) {
    mlir::Type src_ty = src_shaped_ty.getElementType();
    return CreateConst(b, src_ty, new_value, src_shaped_ty.getShape())
        .UnwrapUnsafe();
  }
  return CreateConst(b, like.getType(), new_value).UnwrapUnsafe();
}

inline mlir::Value ZerosLike(mlir::ImplicitLocOpBuilder& b, mlir::Value x) {
  return ConstLike(b, x, 0);
}

inline mlir::Value OnesLike(mlir::ImplicitLocOpBuilder& b, mlir::Value x) {
  return ConstLike(b, x, 1);
}

bool IsFp8Type(mlir::Type t);

ScalarOrTensor Splat(mlir::ImplicitLocOpBuilder& b, ScalarOrTensor value,
                     llvm::ArrayRef<int64_t> shape);

// Triton type conversions.
mlir::Value Cast(mlir::ImplicitLocOpBuilder& b, mlir::Value value,
                 mlir::Type dst_element_ty);

// Emits a scalar constant.
absl::StatusOr<ScalarOrTensor> EmitConstant(mlir::ImplicitLocOpBuilder& b,
                                            const HloInstruction& constant);

absl::StatusOr<mlir::Value> EmitElementwise(
    mlir::ImplicitLocOpBuilder& b, absl::string_view libdevice_path,
    const se::DeviceDescription& device_info, const HloInstruction& hlo,
    mlir::ValueRange inputs);

// Emit sequence of operations for unpacking 2xi4 -> i8.
absl::StatusOr<mlir::Value> EmitUnpackInt4(mlir::ImplicitLocOpBuilder& b,
                                           const HloInstruction* hlo,
                                           int64_t unpack_dim_idx,
                                           mlir::Value value);
}  // namespace xla::gpu::triton

#endif  // XLA_SERVICE_GPU_FUSIONS_TRITON_EMITTER_HELPERS_H_
