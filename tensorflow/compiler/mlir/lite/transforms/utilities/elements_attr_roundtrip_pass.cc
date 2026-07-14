/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/transforms/utilities/elements_attr_roundtrip_pass.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/DialectResourceBlobManager.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace TFL {
namespace {

static bool ShouldConvertElementsToResource(ElementsAttr attr) {
  if (llvm::isa<DenseElementsAttr>(attr)) {
    // DenseElementsAttr encodes arbitrary dimension
    // splats whereas DenseResourceElementsAttr does not.
    return !attr.isSplat();
  }

  return false;
}

template <typename ElementType, unsigned numBits = sizeof(ElementType) * 8>
static void CopyIntAttrIntoBlob(AsmResourceBlob& blob,
                                DenseIntElementsAttr attr) {
  ArrayRef<ElementType> data = blob.getDataAs<ElementType>();
  MutableArrayRef<ElementType> rw_data = MutableArrayRef<ElementType>(
      const_cast<ElementType*>(data.data()), data.size());
  ArrayRef<char> raw_src_data = attr.getRawData();
  if (raw_src_data.size() == blob.getData().size()) {
    // Memcpy.
    std::memcpy(rw_data.data(), raw_src_data.data(), raw_src_data.size());
  } else {
    // Slow.
    size_t index = 0;
    for (APInt value : attr.getValues<APInt>()) {
      rw_data[index++] = value.extractBitsAsZExtValue(numBits, 0);
    }
  }
}

template <typename ElementType, unsigned num_bits = sizeof(ElementType) * 8>
static void CopyFPAttrIntoBlob(AsmResourceBlob& blob,
                               DenseFPElementsAttr attr) {
  ArrayRef<ElementType> data = blob.getDataAs<ElementType>();
  MutableArrayRef<ElementType> rw_data = MutableArrayRef<ElementType>(
      const_cast<ElementType*>(data.data()), data.size());
  ArrayRef<char> raw_src_data = attr.getRawData();
  if (raw_src_data.size() == blob.getData().size()) {
    // Memcpy.
    std::memcpy(rw_data.data(), raw_src_data.data(), raw_src_data.size());
  } else {
    // Slow.
    size_t index = 0;
    for (APFloat value : attr.getValues<APFloat>()) {
      rw_data[index++] =
          value.bitcastToAPInt().extractBitsAsZExtValue(num_bits, 0);
    }
  }
}

static absl::StatusOr<ElementsAttr> ConvertDenseElementsAttr(
    ElementsAttr elements_attr) {
  auto shaped_type = llvm::cast<ShapedType>(elements_attr.getType());
  auto element_type = shaped_type.getElementType();
  auto num_elements = elements_attr.getNumElements();
  auto bit_width = element_type.getIntOrFloatBitWidth();

  if (!shaped_type.hasStaticShape()) {
    return absl::InvalidArgumentError(
        "DenseElementsAttr to be converted has dynamic shape");
  }

  AsmResourceBlob blob;
  if (auto attr = llvm::dyn_cast<DenseIntElementsAttr>(elements_attr)) {
    switch (bit_width) {
      case 1:
        blob = HeapAsmResourceBlob::allocate(num_elements, /*align=*/64,
                                             /*dataIsMutable=*/true);
        CopyIntAttrIntoBlob<uint8_t, /*numBits=*/1>(blob, attr);
        return DenseResourceElementsAttr::get(shaped_type, "dense_elements_i1",
                                              std::move(blob));
      case 8:
        blob = HeapAsmResourceBlob::allocate(num_elements, /*align=*/64,
                                             /*dataIsMutable=*/true);
        CopyIntAttrIntoBlob<uint8_t>(blob, attr);
        return DenseResourceElementsAttr::get(shaped_type, "dense_elements_i8",
                                              std::move(blob));
      case 16:
        blob = HeapAsmResourceBlob::allocate(2 * num_elements, /*align=*/64,
                                             /*dataIsMutable=*/true);
        CopyIntAttrIntoBlob<uint16_t>(blob, attr);
        return DenseResourceElementsAttr::get(shaped_type, "dense_elements_i16",
                                              std::move(blob));
      case 32:
        blob = HeapAsmResourceBlob::allocate(4 * num_elements, /*align=*/64,
                                             /*dataIsMutable=*/true);
        CopyIntAttrIntoBlob<uint32_t>(blob, attr);
        return DenseResourceElementsAttr::get(shaped_type, "dense_elements_i32",
                                              std::move(blob));
      case 64:
        blob = HeapAsmResourceBlob::allocate(8 * num_elements, /*align=*/64,
                                             /*dataIsMutable=*/true);
        CopyIntAttrIntoBlob<uint64_t>(blob, attr);
        return DenseResourceElementsAttr::get(shaped_type, "dense_elements_i64",
                                              std::move(blob));
      default:
        return absl::InvalidArgumentError("Unsupported bit width");
    }
  } else if (auto attr = llvm::dyn_cast<DenseFPElementsAttr>(elements_attr)) {
    AsmResourceBlob blob;
    switch (bit_width) {
      case 8:
        blob = HeapAsmResourceBlob::allocate(num_elements, /*align=*/64,
                                             /*dataIsMutable=*/true);
        CopyFPAttrIntoBlob<uint8_t>(blob, attr);
        return DenseResourceElementsAttr::get(shaped_type, "dense_elements_f8",
                                              std::move(blob));
      case 16:
        blob = HeapAsmResourceBlob::allocate(2 * num_elements, /*align=*/64,
                                             /*dataIsMutable=*/true);
        CopyFPAttrIntoBlob<uint16_t>(blob, attr);
        return DenseResourceElementsAttr::get(shaped_type, "dense_elements_f16",
                                              std::move(blob));
      case 32:
        blob = HeapAsmResourceBlob::allocate(4 * num_elements, /*align=*/64,
                                             /*dataIsMutable=*/true);
        CopyFPAttrIntoBlob<uint32_t>(blob, attr);
        return DenseResourceElementsAttr::get(shaped_type, "dense_elements_f32",
                                              std::move(blob));
      case 64:
        blob = HeapAsmResourceBlob::allocate(8 * num_elements, /*align=*/64,
                                             /*dataIsMutable=*/true);
        CopyFPAttrIntoBlob<uint64_t>(blob, attr);
        return DenseResourceElementsAttr::get(shaped_type, "dense_elements_f64",
                                              std::move(blob));
      default:
        return absl::InvalidArgumentError("Unsupported bit width");
    }
  }
  return absl::InvalidArgumentError("Unsupported element type");
}

static absl::StatusOr<ElementsAttr> ConvertDenseResourceElementsAttr(
    ElementsAttr elements_attr) {
  auto shaped_type = llvm::cast<ShapedType>(elements_attr.getType());
  auto element_type = shaped_type.getElementType();

  if (!shaped_type.hasStaticShape()) {
    return absl::InvalidArgumentError(
        "DenseResourceElementsAttr to be converted has dynamic shape");
  }

  auto dense_resource_elements_attr =
      llvm::dyn_cast_or_null<DenseResourceElementsAttr>(elements_attr);
  if (!dense_resource_elements_attr)
    return absl::InvalidArgumentError("DenseResourceElementsAttr is null");

  AsmResourceBlob* blob = dense_resource_elements_attr.getRawHandle().getBlob();
  if (!blob) return absl::InvalidArgumentError("AsmResourceBlob is null");

  // Special handling for i1 because we store it as bytes in the resource
  // (unpacked), but DenseElementsAttr expects bits (packed).
  if (element_type.isInteger(1)) {
    return DenseElementsAttr::get(shaped_type,
                                  ArrayRef<bool>(blob->getDataAs<bool>()));
  }

  // For other types, the resource blob contains the raw data matching
  // DenseElementsAttr's expected layout.
  if (element_type.isIntOrFloat()) {
    return DenseElementsAttr::getFromRawBuffer(shaped_type, blob->getData());
  }

  return absl::InvalidArgumentError("Unsupported element type");
}

}  // namespace

void DenseToDenseResourceElementsPass::runOnOperation() {
  ModuleOp module_op = getOperation();

  module_op.walk([&](Operation* op) {
    if (auto const_op = llvm::dyn_cast<arith::ConstantOp>(op)) {
      mlir::ElementsAttr elements_attr;
      if (mlir::detail::constant_op_binder<mlir::ElementsAttr>(&elements_attr)
              .match(const_op)) {
        // Convert.
        if (ShouldConvertElementsToResource(elements_attr)) {
          if (auto replacement = ConvertDenseElementsAttr(elements_attr);
              replacement.ok()) {
            const_op.setValueAttr(replacement.value());
          }
        }
      }
    }
  });
}

void DenseResourceToDenseElementsPass::runOnOperation() {
  ModuleOp module_op = getOperation();

  module_op.walk([&](Operation* op) {
    if (auto const_op = llvm::dyn_cast<arith::ConstantOp>(op)) {
      mlir::ElementsAttr elements_attr;
      if (mlir::detail::constant_op_binder<mlir::ElementsAttr>(&elements_attr)
              .match(const_op)) {
        // Convert.
        if (llvm::isa<DenseResourceElementsAttr>(elements_attr)) {
          if (auto replacement =
                  ConvertDenseResourceElementsAttr(elements_attr);
              replacement.ok()) {
            const_op.setValueAttr(replacement.value());
          }
        }
      }
    }
  });
}

}  // namespace TFL
}  // namespace mlir
