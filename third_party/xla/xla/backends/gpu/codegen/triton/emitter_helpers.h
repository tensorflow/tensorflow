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

#ifndef XLA_BACKENDS_GPU_CODEGEN_TRITON_EMITTER_HELPERS_H_
#define XLA_BACKENDS_GPU_CODEGEN_TRITON_EMITTER_HELPERS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/emitter_loc_op_builder.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/tma_metadata.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/tsl/platform/status.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace xla::gpu::triton {

// Returns a string representation of the given MLIR entity.
template <typename T>
std::string MlirToString(T&& value) {
  std::string result;
  llvm::raw_string_ostream os(result);
  value.print(os);
  return result;
}

// Triton requires that all block dimensions are a power of 2.
// TODO(b/353484968): Delete this function once we have constraints to only
// propagate tile sizes that are a power of 2.
llvm::SmallVector<int64_t> GetPaddedTileSizes(
    llvm::ArrayRef<int64_t> tile_sizes);

// XLA -> Triton type conversions.
absl::StatusOr<mlir::Type> TritonType(EmitterLocOpBuilder& b, PrimitiveType t);

// Triton type -> XLA type conversions.
absl::StatusOr<PrimitiveType> GetPrimitiveType(mlir::Type t);

mlir::Type StorageType(mlir::Type t);

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
mlir::Value CreateConst(EmitterLocOpBuilder& b, mlir::Type type, T value) {
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
    EmitterLocOpBuilder& b, mlir::Type type, T value,
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
mlir::Value ConstLike(EmitterLocOpBuilder& b, mlir::Value like, T new_value) {
  if (auto src_shaped_ty = mlir::dyn_cast<mlir::ShapedType>(like.getType())) {
    mlir::Type src_ty = src_shaped_ty.getElementType();
    return CreateConst(b, src_ty, new_value, src_shaped_ty.getShape());
  }
  return CreateConst(b, like.getType(), new_value);
}

inline mlir::Value ZerosLike(EmitterLocOpBuilder& b, mlir::Value x) {
  return ConstLike(b, x, 0);
}

inline mlir::Value OnesLike(EmitterLocOpBuilder& b, mlir::Value x) {
  return ConstLike(b, x, 1);
}

bool IsFp8Type(mlir::Type t);

// Triton type conversions.
mlir::Value Cast(EmitterLocOpBuilder& b, mlir::Value value,
                 mlir::Type dst_element_ty);

// Emits a scalar constant.
absl::StatusOr<mlir::TypedValue<mlir::RankedTensorType>> EmitConstant(
    EmitterLocOpBuilder& b, const HloInstruction& constant);

bool IsSupportedElementwiseLibdeviceFunction(const HloInstruction& hlo);

// Should only be called if IsSupportedElementwiseLibdeviceFunction() returns
// true for `hlo`, otherwise an error is returned.
absl::StatusOr<mlir::Value> EmitElementwiseLibdeviceFunction(
    EmitterLocOpBuilder& b, absl::string_view libdevice_path,
    const se::DeviceDescription& device_info, const HloInstruction& hlo,
    mlir::ValueRange inputs);

absl::StatusOr<mlir::Value> EmitElementwise(
    EmitterLocOpBuilder& b, const se::DeviceDescription& device_info,
    const HloInstruction& hlo, mlir::ValueRange inputs);

mlir::Value Bitcast(EmitterLocOpBuilder& b, mlir::Value value, mlir::Type type);

// Extracts NVVM annotations from the Triton module.
std::vector<llvm::Metadata*> ExtractNvvmAnnotations(
    llvm::Module* ll_triton_module);

// Extracts TMA metadata information from LLVM generated by the Triton
// compilation. The underlying map will be empty if TMA is not used.
absl::StatusOr<stream_executor::gpu::TmaMetadata> ExtractTmaMetadata(
    mlir::LLVM::LLVMFuncOp func_op);

// Extracts thread dimensions from Triton module attributes.
absl::StatusOr<stream_executor::ThreadDim> ExtractThreadDims(
    mlir::ModuleOp triton_module, mlir::LLVM::LLVMFuncOp func_op);

// Returns the triton pointer type with global memory space and the given
// element type.
::mlir::triton::PointerType GetGlobalPointerType(mlir::Type element_type);

}  // namespace xla::gpu::triton

#endif  // XLA_BACKENDS_GPU_CODEGEN_TRITON_EMITTER_HELPERS_H_
