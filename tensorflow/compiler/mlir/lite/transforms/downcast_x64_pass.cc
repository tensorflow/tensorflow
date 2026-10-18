/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/transforms/downcast_x64_pass.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {

namespace {

// Recursively converts f64 -> f32 and i64 -> i32 for scalars and tensors.
mlir::Type ConvertX64TypeToX32(mlir::Type type) {
  if (auto tensor_type = mlir::dyn_cast<mlir::TensorType>(type)) {
    mlir::Type element_type = tensor_type.getElementType();
    mlir::Type new_element_type = ConvertX64TypeToX32(element_type);
    if (element_type == new_element_type) {
      return type;
    }

    if (auto ranked_ty = mlir::dyn_cast<mlir::RankedTensorType>(tensor_type)) {
      return mlir::RankedTensorType::get(ranked_ty.getShape(),
                                         new_element_type);
    }
    return mlir::UnrankedTensorType::get(new_element_type);
  }

  if (type.isF64()) {
    return mlir::Float32Type::get(type.getContext());
  }
  // Preserve signedness so ui64 narrows to ui32 rather than signless i32.
  if (auto int_type = mlir::dyn_cast<mlir::IntegerType>(type);
      int_type && int_type.getWidth() == 64) {
    return mlir::IntegerType::get(type.getContext(), 32,
                                  int_type.getSignedness());
  }
  return type;
}

// Emits the saturation warning on `op`.
void EmitSaturationWarning(mlir::Operation* op, bool is_unsigned) {
  op->emitWarning()
      << "enable_x64=false: 64-bit integer constant has values outside the "
         "32-bit range; saturating them to "
      << (is_unsigned ? "[0, UINT32_MAX]" : "[INT32_MIN, INT32_MAX]")
      << ". Pass enable_x64=true to keep 64-bit types.";
}

// Narrows a `DenseIntElementsAttr` of 64-bit integers to 32 bits, clamping
// out-of-range values. Sets `*clamped` if any value was clamped.
mlir::DenseIntElementsAttr NarrowDenseI64(mlir::DenseIntElementsAttr value,
                                          mlir::Type new_element_type,
                                          bool* clamped) {
  const bool is_unsigned = value.getElementType().isUnsignedInteger();
  const APInt min_value =
      is_unsigned ? APInt::getZero(64) : APInt::getSignedMinValue(32).sext(64);
  const APInt max_value = is_unsigned ? APInt::getMaxValue(32).zext(64)
                                      : APInt::getSignedMaxValue(32).sext(64);
  auto less_than = [is_unsigned](const APInt& a, const APInt& b) {
    return is_unsigned ? a.ult(b) : a.slt(b);
  };
  return mlir::cast<mlir::DenseIntElementsAttr>(
      value.mapValues(new_element_type, [&](const APInt& v) -> APInt {
        if (less_than(v, min_value)) {
          *clamped = true;
          return min_value.trunc(32);
        }
        if (less_than(max_value, v)) {
          *clamped = true;
          return max_value.trunc(32);
        }
        return v.trunc(32);
      }));
}

// Narrows a `DenseResourceElementsAttr` of 64-bit integers to 32 bits by
// allocating a new blob, clamping out-of-range values. Returns a null
// attribute if the resource's blob is unavailable. Sets `*clamped` if any
// value was clamped.
mlir::DenseResourceElementsAttr NarrowResourceI64(
    mlir::DenseResourceElementsAttr value, mlir::ShapedType new_type,
    bool* clamped) {
  mlir::AsmResourceBlob* blob = value.getRawHandle().getBlob();
  if (!blob) {
    if (auto* resource = value.getRawHandle().getResource()) {
      blob = resource->getBlob();
    }
  }
  if (!blob) return nullptr;

  const bool is_unsigned = value.getElementType().isUnsignedInteger();
  llvm::ArrayRef<char> src_bytes = blob->getData();
  const int64_t num_elements = new_type.getNumElements();
  if (src_bytes.size() != num_elements * sizeof(int64_t)) return nullptr;
  mlir::AsmResourceBlob new_blob = mlir::HeapAsmResourceBlob::allocate(
      num_elements * sizeof(int32_t), /*align=*/64, /*dataIsMutable=*/true);
  llvm::MutableArrayRef<char> dst_bytes = new_blob.getMutableData();

  // Copy element-wise through memcpy: the source blob's alignment is not
  // guaranteed to be suitable for a direct int64_t view.
  for (int64_t i = 0; i < num_elements; ++i) {
    const char* src = src_bytes.data() + i * sizeof(int64_t);
    char* dst = dst_bytes.data() + i * sizeof(int32_t);
    if (is_unsigned) {
      uint64_t v;
      std::memcpy(&v, src, sizeof(v));
      if (v > std::numeric_limits<uint32_t>::max()) {
        *clamped = true;
        v = std::numeric_limits<uint32_t>::max();
      }
      const uint32_t narrowed = static_cast<uint32_t>(v);
      std::memcpy(dst, &narrowed, sizeof(narrowed));
    } else {
      int64_t v;
      std::memcpy(&v, src, sizeof(v));
      if (v < std::numeric_limits<int32_t>::min()) {
        *clamped = true;
        v = std::numeric_limits<int32_t>::min();
      } else if (v > std::numeric_limits<int32_t>::max()) {
        *clamped = true;
        v = std::numeric_limits<int32_t>::max();
      }
      const int32_t narrowed = static_cast<int32_t>(v);
      std::memcpy(dst, &narrowed, sizeof(narrowed));
    }
  }

  std::string new_key =
      absl::StrCat(value.getRawHandle().getKey().str(), "_x32");
  return mlir::DenseResourceElementsAttr::get(new_type, new_key,
                                              std::move(new_blob));
}

// If `op` is a 64-bit integer constant, narrows its `value` attribute and
// result type to 32 bits in place and returns true. Values outside the 32-bit
// range are clamped to that range first, with a warning, so the constant
// saturates instead of wrapping. Handles both `DenseIntElementsAttr` and
// `DenseResourceElementsAttr` (the latter is what `lightweight_conversion`
// produces for large constants). Returns false for any other constant (e.g.
// f64), leaving it for the cast path.
//
// Signed and signless constants clamp to [INT32_MIN, INT32_MAX]; unsigned
// constants clamp to [0, UINT32_MAX] using unsigned comparisons, so a value
// with the top bit set is treated as large rather than negative. Narrowing
// directly (rather than through `tfl.cast`) also keeps ui64 working, since
// `tfl.cast` does not accept a ui64 input.
//
// PyTorch exports routinely use INT64_MAX / INT64_MIN as sentinels (e.g. the
// fill value of `torch.where(mask, x, iinfo(long).max).min()` or
// `masked_fill(~m, iinfo(long).min).argmax()`). Truncating those to -1 / 0
// would silently change the result; saturating keeps their meaning.
bool NarrowI64Constant(mlir::Operation* op) {
  if (op->getNumResults() != 1) return false;
  auto value = op->getAttrOfType<mlir::ElementsAttr>("value");
  if (!value || !value.getElementType().isInteger(64)) return false;
  if (value.getType() != op->getResult(0).getType()) return false;

  const bool is_unsigned = value.getElementType().isUnsignedInteger();
  auto new_type =
      mlir::cast<mlir::ShapedType>(ConvertX64TypeToX32(value.getType()));
  bool clamped = false;
  mlir::ElementsAttr new_value;
  if (auto dense = mlir::dyn_cast<mlir::DenseIntElementsAttr>(value)) {
    new_value = NarrowDenseI64(dense, new_type.getElementType(), &clamped);
  } else if (auto resource =
                 mlir::dyn_cast<mlir::DenseResourceElementsAttr>(value)) {
    new_value = NarrowResourceI64(resource, new_type, &clamped);
  }
  if (!new_value) return false;

  if (clamped) EmitSaturationWarning(op, is_unsigned);
  op->setAttr("value", new_value);
  op->getResult(0).setType(new_type);
  return true;
}

}  // namespace

void DowncastX64Pass::runOnOperation() {
  mlir::MLIRContext* ctx = &getContext();
  func::FuncOp func_op = getOperation();

  // Update Function signatures and entry block arguments.
  func_op->walk([&](mlir::func::FuncOp func_op) {
    mlir::FunctionType old_func_type = func_op.getFunctionType();

    mlir::SmallVector<mlir::Type> new_input_types;
    for (mlir::Type input_type : old_func_type.getInputs()) {
      new_input_types.push_back(ConvertX64TypeToX32(input_type));
    }

    mlir::SmallVector<mlir::Type> new_result_types;
    for (mlir::Type result_type : old_func_type.getResults()) {
      new_result_types.push_back(ConvertX64TypeToX32(result_type));
    }

    func_op.setType(
        mlir::FunctionType::get(ctx, new_input_types, new_result_types));

    // Update block arguments within the function.
    if (!func_op.empty()) {
      for (auto [arg, new_type] :
           llvm::zip(func_op.getArguments(), new_input_types)) {
        arg.setType(new_type);
      }
    }
  });

  // Update all operation results and handle Constant Attributes.
  func_op->walk([&](mlir::Operation* op) {
    // Skip the function itself as we handled it above.
    if (mlir::isa<mlir::func::FuncOp>(op)) {
      return;
    }

    // Handle constant-like operations. 64-bit integer constants (dense or
    // resource-backed) are narrowed in place; anything else gets a CastOp for
    // the constant folder to resolve.
    if (op->hasTrait<mlir::OpTrait::ConstantLike>()) {
      if (NarrowI64Constant(op)) return;
      for (mlir::OpResult result : op->getResults()) {
        mlir::Type old_type = result.getType();
        mlir::Type new_type = ConvertX64TypeToX32(old_type);

        if (new_type != old_type) {
          // Insert CastOp to let the constant folder handle the narrowing.
          mlir::OpBuilder builder(op->getContext());
          builder.setInsertionPointAfter(op);

          auto cast_op = mlir::TFL::CastOp::create(builder, op->getLoc(),
                                                   new_type, result);
          result.replaceAllUsesExcept(cast_op.getResult(), cast_op);
        }
      }
      return;
    }

    // Update result types for all operations.
    for (mlir::Value result : op->getResults()) {
      result.setType(ConvertX64TypeToX32(result.getType()));
    }

    // Update block arguments of any nested regions (e.g. the cond/body of a
    // region-based `tfl.while`, or a residual `stablehlo.while`/`reduce`), so
    // they stay consistent with the narrowed operands feeding them.
    for (mlir::Region& region : op->getRegions()) {
      for (mlir::Block& block : region) {
        for (mlir::BlockArgument arg : block.getArguments()) {
          arg.setType(ConvertX64TypeToX32(arg.getType()));
        }
      }
    }
  });
}

std::unique_ptr<mlir::OperationPass<func::FuncOp>> CreateDowncastX64Pass() {
  return std::make_unique<DowncastX64Pass>();
}

}  // end namespace TFL
}  // end namespace mlir
