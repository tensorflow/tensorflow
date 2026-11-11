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

#include "tensorflow/compiler/mlir/lite/export/constants.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Quant/IR/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/AsmState.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/export/payloads.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/schema/mutable/schema_generated.h"
#include "tensorflow/compiler/mlir/lite/utils/low_bit_utils.h"

namespace mlir::TFL::fb_export {
namespace {

// Elements per chunk when converting elements on the fly.
constexpr int64_t kChunkElements = 64 * 1024;

absl::string_view ToStringView(ArrayRef<char> data) {
  return absl::string_view(data.data(), data.size());
}

AsmResourceBlob* GetBlob(DenseResourceElementsAttr attr) {
  AsmResourceBlob* blob = attr.getRawHandle().getBlob();
  if (blob == nullptr && attr.getRawHandle().getResource() != nullptr) {
    blob = attr.getRawHandle().getResource()->getBlob();
  }
  return blob;
}

// Returns the number of bytes MLIR stores one element of `type` in.
int64_t GetStorageBytes(Type type) {
  if (auto complex = dyn_cast<ComplexType>(type)) {
    return 2 * GetStorageBytes(complex.getElementType());
  }
  if (auto quantized = dyn_cast<quant::QuantizedType>(type)) {
    return GetStorageBytes(quantized.getStorageType());
  }
  if (auto blockwise = dyn_cast<BlockwiseQuantType>(type)) {
    return GetStorageBytes(blockwise.getStorageType());
  }
  return llvm::divideCeil(type.getIntOrFloatBitWidth(), 8);
}

// Returns the dense storage of `value`, or std::nullopt if it doesn't store
// one value per element (splats) or isn't byte addressable (booleans).
std::optional<ArrayRef<char>> GetRawData(ElementsAttr value) {
  if (auto resource = dyn_cast<DenseResourceElementsAttr>(value)) {
    AsmResourceBlob* blob = GetBlob(resource);
    if (blob == nullptr) return std::nullopt;
    return blob->getData();
  }
  if (auto dense = dyn_cast<DenseElementsAttr>(value)) {
    if (dense.isSplat() || dense.getElementType().isInteger(1)) {
      return std::nullopt;
    }
    return dense.getRawData();
  }
  return std::nullopt;
}

absl::Status UnexpectedStorage() {
  return absl::InvalidArgumentError(
      "constant storage doesn't match its element type");
}

// Writes `count` copies of `element`.
absl::Status WriteRepeated(absl::string_view element, int64_t count,
                           ByteSink sink) {
  const int64_t per_chunk = std::min(count, kChunkElements);
  std::string chunk;
  chunk.reserve(per_chunk * element.size());
  for (int64_t i = 0; i < per_chunk; ++i) chunk.append(element);
  for (int64_t written = 0; written < count; written += per_chunk) {
    const int64_t n = std::min(per_chunk, count - written);
    if (absl::Status s =
            sink(absl::string_view(chunk).substr(0, n * element.size()));
        !s.ok()) {
      return s;
    }
  }
  return absl::OkStatus();
}

absl::Status WriteRaw(ElementsAttr value, ByteSink sink) {
  const int64_t num_elements = value.getNumElements();
  const Type element_type = value.getElementType();
  if (std::optional<ArrayRef<char>> raw = GetRawData(value)) {
    if (raw->size() != num_elements * GetStorageBytes(element_type)) {
      return UnexpectedStorage();
    }
    return sink(ToStringView(*raw));
  }
  auto dense = dyn_cast<DenseElementsAttr>(value);
  if (!dense) return UnexpectedStorage();
  if (element_type.isInteger(1)) {
    std::string bytes;
    bytes.reserve(num_elements);
    for (bool b : dense.getValues<bool>()) bytes.push_back(b ? 1 : 0);
    return sink(bytes);
  }
  return WriteRepeated(ToStringView(dense.getRawData()), num_elements, sink);
}

template <int kBitWidth>
absl::Status WritePacked(ElementsAttr value, ByteSink sink) {
  const int64_t num_elements = value.getNumElements();
  if (std::optional<ArrayRef<char>> raw = GetRawData(value)) {
    // MLIR stores one element per byte. For a single element this can't be
    // told apart from packed storage, which packing leaves unchanged anyway.
    if (raw->size() == num_elements) {
      return tflite::StreamPackLowBitValues8Bit<kBitWidth>(*raw, sink);
    }
    // Already packed.
    if (raw->size() == llvm::divideCeil(num_elements * kBitWidth, 8)) {
      return sink(ToStringView(*raw));
    }
    return UnexpectedStorage();
  }
  auto dense = dyn_cast<DenseElementsAttr>(value);
  if (!dense) return UnexpectedStorage();
  return tflite::StreamPackLowBitValues<kBitWidth>(dense.getValues<APInt>(),
                                                   sink);
}

float HalfToFloat(uint16_t bits, bool is_bf16) {
  if (is_bf16) {
    const uint32_t f32_bits = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &f32_bits, sizeof(f));
    return f;
  }
  return APFloat(APFloat::IEEEhalf(), APInt(16, bits)).convertToFloat();
}

absl::Status WriteWidenedToF32(ElementsAttr value, ByteSink sink) {
  const int64_t num_elements = value.getNumElements();
  const bool is_bf16 = value.getElementType().isBF16();
  auto as_bytes = [](const float* data, int64_t n) {
    return absl::string_view(reinterpret_cast<const char*>(data),
                             n * sizeof(float));
  };

  std::optional<ArrayRef<char>> raw = GetRawData(value);
  if (!raw.has_value()) {
    auto dense = dyn_cast<DenseElementsAttr>(value);
    if (!dense || !dense.isSplat()) return UnexpectedStorage();
    uint16_t bits;
    std::memcpy(&bits, dense.getRawData().data(), sizeof(bits));
    const float f = HalfToFloat(bits, is_bf16);
    return WriteRepeated(as_bytes(&f, 1), num_elements, sink);
  }
  if (raw->size() != num_elements * sizeof(uint16_t)) {
    return UnexpectedStorage();
  }

  std::vector<float> chunk(std::min(num_elements, kChunkElements));
  for (int64_t start = 0; start < num_elements; start += chunk.size()) {
    const int64_t n = std::min<int64_t>(chunk.size(), num_elements - start);
    for (int64_t i = 0; i < n; ++i) {
      uint16_t bits;
      std::memcpy(&bits, raw->data() + (start + i) * sizeof(bits),
                  sizeof(bits));
      chunk[i] = HalfToFloat(bits, is_bf16);
    }
    if (absl::Status s = sink(as_bytes(chunk.data(), n)); !s.ok()) return s;
  }
  return absl::OkStatus();
}

// Returns the value of a constant op.
std::optional<ElementsAttr> GetConstantValue(Operation* op) {
  if (op == nullptr) return std::nullopt;
  if (auto constant = dyn_cast<ConstOp>(op)) return constant.getValue();
  if (auto constant = dyn_cast<QConstOp>(op)) return constant.getValue();
  if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
    if (auto value = dyn_cast<ElementsAttr>(constant.getValue())) return value;
    return std::nullopt;
  }
  if (auto constant = dyn_cast<vhlo::ConstantOpV1>(op)) {
    return cast<vhlo::TensorV1Attr>(constant.getValue()).getData();
  }
  return std::nullopt;
}

// Returns true for a cast of a f16/bf16 constant to f32.
bool IsFoldableCast(CastOp cast) {
  auto input_type = dyn_cast<ShapedType>(cast.getInput().getType());
  auto output_type = dyn_cast<ShapedType>(cast.getType());
  if (!input_type || !output_type || !input_type.hasStaticShape() ||
      !output_type.hasStaticShape()) {
    return false;
  }
  const Type input_element = input_type.getElementType();
  return (input_element.isF16() || input_element.isBF16()) &&
         output_type.getElementType().isF32() &&
         GetConstantValue(cast.getInput().getDefiningOp()).has_value();
}

// Reads the elements of `attr` as `APInt`s of the element bit width, expanding
// splats. Like `DenseElementsAttr`, this expects every element to be stored in
// a whole number of bytes, e.g. one byte per element for sub-byte types.
absl::StatusOr<std::vector<APInt>> ReadElementsAsAPInts(ElementsAttr attr) {
  const unsigned bit_width = attr.getShapedType().getElementTypeBitWidth();
  const int64_t num_elements = attr.getNumElements();
  const size_t stride = llvm::divideCeil(bit_width, 8);
  if (bit_width == 1) return UnexpectedStorage();

  ArrayRef<char> raw;
  bool is_splat = false;
  if (auto dense = dyn_cast<DenseElementsAttr>(attr)) {
    raw = dense.getRawData();
    is_splat = dense.isSplat();
  } else if (std::optional<ArrayRef<char>> data = GetRawData(attr)) {
    raw = *data;
  } else {
    return UnexpectedStorage();
  }
  if (raw.size() < (is_splat ? stride : stride * num_elements)) {
    return UnexpectedStorage();
  }

  std::vector<APInt> values;
  values.reserve(num_elements);
  for (int64_t i = 0; i < num_elements; ++i) {
    uint64_t bits = 0;
    std::memcpy(&bits, raw.data() + (is_splat ? 0 : i * stride),
                std::min<size_t>(stride, sizeof(bits)));
    values.emplace_back(bit_width, bits, /*isSigned=*/false,
                        /*implicitTrunc=*/true);
  }
  return values;
}

template <typename T>
void Append(std::string& data, T value) {
  data.append(reinterpret_cast<const char*>(&value), sizeof(value));
}

float ToFloat(const llvm::fltSemantics& semantics, const APInt& bits) {
  APFloat value(semantics, bits);
  bool loses_info = false;
  value.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven,
                &loses_info);
  return value.convertToFloat();
}

BlockwiseParameter MakeParameter(ElementsAttr attr, tflite::TensorType type) {
  BlockwiseParameter parameter;
  parameter.type = type;
  for (int64_t dim : attr.getShapedType().getShape()) {
    parameter.shape.push_back(static_cast<int32_t>(dim));
  }
  return parameter;
}

}  // namespace

std::optional<Constant> GetConstant(Operation* op) {
  if (auto cast = dyn_cast_or_null<CastOp>(op)) {
    if (!IsFoldableCast(cast)) return std::nullopt;
    return Constant{*GetConstantValue(cast.getInput().getDefiningOp()),
                    ConstantEncoding::kWidenedToF32};
  }
  if (std::optional<ElementsAttr> value = GetConstantValue(op)) {
    return Constant{*value};
  }
  return std::nullopt;
}

bool IsOnlyUsedByFoldedCasts(Operation* op) {
  if (!GetConstantValue(op).has_value() || op->use_empty()) return false;
  return llvm::all_of(op->getUsers(), [](Operation* user) {
    auto cast = dyn_cast<CastOp>(user);
    return cast && IsFoldableCast(cast);
  });
}

ConstantEncoding GetEncoding(tflite::TensorType type) {
  switch (type) {
    case tflite::TensorType_INT4:
    case tflite::TensorType_UINT4:
      return ConstantEncoding::kPacked4Bit;
    case tflite::TensorType_INT2:
      return ConstantEncoding::kPacked2Bit;
    default:
      return ConstantEncoding::kRaw;
  }
}

uint64_t GetFingerprint(const Constant& constant) {
  llvm::hash_code data_hash;
  if (auto resource = dyn_cast<DenseResourceElementsAttr>(constant.value)) {
    // Distinct resources may hold the same data, e.g. a weight materialized
    // once per signature.
    AsmResourceBlob* blob = GetBlob(resource);
    if (blob != nullptr && !blob->getData().empty()) {
      ArrayRef<char> data = blob->getData();
      data_hash = llvm::hash_code(llvm::xxh3_64bits(ArrayRef<uint8_t>(
          reinterpret_cast<const uint8_t*>(data.data()), data.size())));
    } else {
      data_hash = llvm::hash_value(resource.getRawHandle().getKey());
    }
    data_hash = llvm::hash_combine(data_hash, resource.getType());
  } else {
    // Other attributes are uniqued, so equal data means equal attributes.
    data_hash = hash_value(constant.value);
  }
  return llvm::hash_combine(data_hash, static_cast<int>(constant.encoding));
}

uint64_t GetByteSize(const Constant& constant) {
  const int64_t num_elements = constant.value.getNumElements();
  switch (constant.encoding) {
    case ConstantEncoding::kRaw:
      return num_elements * GetStorageBytes(constant.value.getElementType());
    case ConstantEncoding::kPacked4Bit:
      return llvm::divideCeil(num_elements, 2);
    case ConstantEncoding::kPacked2Bit:
      return llvm::divideCeil(num_elements, 4);
    case ConstantEncoding::kWidenedToF32:
      return num_elements * sizeof(float);
  }
  return 0;
}

AsmResourceBlob* GetResourceBlob(const Constant& constant) {
  if (auto resource = dyn_cast<DenseResourceElementsAttr>(constant.value)) {
    return GetBlob(resource);
  }
  return nullptr;
}

absl::Status CheckConstant(const Constant& constant) {
  const ElementsAttr value = constant.value;
  const int64_t num_elements = value.getNumElements();
  if (num_elements == 0) return absl::OkStatus();
  const std::optional<ArrayRef<char>> raw = GetRawData(value);
  // Dense splats and booleans are expanded while writing.
  if (!raw.has_value()) {
    return isa<DenseElementsAttr>(value) ? absl::OkStatus()
                                         : UnexpectedStorage();
  }
  bool valid = false;
  switch (constant.encoding) {
    case ConstantEncoding::kRaw:
      valid =
          raw->size() == num_elements * GetStorageBytes(value.getElementType());
      break;
    case ConstantEncoding::kPacked4Bit:
      valid = raw->size() == num_elements ||
              raw->size() == llvm::divideCeil(num_elements * 4, 8);
      break;
    case ConstantEncoding::kPacked2Bit:
      valid = raw->size() == num_elements ||
              raw->size() == llvm::divideCeil(num_elements * 2, 8);
      break;
    case ConstantEncoding::kWidenedToF32:
      valid = raw->size() == num_elements * sizeof(uint16_t);
      break;
  }
  return valid ? absl::OkStatus() : UnexpectedStorage();
}

absl::Status WriteConstant(const Constant& constant, ByteSink sink) {
  if (constant.value.getNumElements() == 0) return absl::OkStatus();
  switch (constant.encoding) {
    case ConstantEncoding::kRaw:
      return WriteRaw(constant.value, sink);
    case ConstantEncoding::kPacked4Bit:
      return WritePacked<4>(constant.value, sink);
    case ConstantEncoding::kPacked2Bit:
      return WritePacked<2>(constant.value, sink);
    case ConstantEncoding::kWidenedToF32:
      return WriteWidenedToF32(constant.value, sink);
  }
  return absl::InternalError("unknown constant encoding");
}

absl::StatusOr<BlockwiseParameter> SerializeBlockwiseScales(
    ElementsAttr scales) {
  auto float_type = dyn_cast<FloatType>(scales.getElementType());
  if (!float_type) {
    return absl::InvalidArgumentError("blockwise scales must be floats");
  }
  absl::StatusOr<std::vector<APInt>> bits = ReadElementsAsAPInts(scales);
  if (!bits.ok()) return bits.status();

  if (float_type.isF16() || float_type.isBF16()) {
    BlockwiseParameter parameter =
        MakeParameter(scales, float_type.isF16() ? tflite::TensorType_FLOAT16
                                                 : tflite::TensorType_BFLOAT16);
    for (const APInt& value : *bits) {
      Append(parameter.data, static_cast<uint16_t>(value.getZExtValue()));
    }
    return parameter;
  }
  BlockwiseParameter parameter =
      MakeParameter(scales, tflite::TensorType_FLOAT32);
  for (const APInt& value : *bits) {
    Append(parameter.data, ToFloat(float_type.getFloatSemantics(), value));
  }
  return parameter;
}

absl::StatusOr<BlockwiseParameter> SerializeBlockwiseZeroPoints(
    ElementsAttr zero_points) {
  const Type element_type = zero_points.getElementType();
  absl::StatusOr<std::vector<APInt>> bits = ReadElementsAsAPInts(zero_points);
  if (!bits.ok()) return bits.status();

  if (auto float_type = dyn_cast<FloatType>(element_type)) {
    BlockwiseParameter parameter =
        MakeParameter(zero_points, tflite::TensorType_FLOAT32);
    for (const APInt& value : *bits) {
      Append(parameter.data, ToFloat(float_type.getFloatSemantics(), value));
    }
    return parameter;
  }
  auto int_type = dyn_cast<IntegerType>(element_type);
  if (!int_type) {
    return absl::InvalidArgumentError(
        "blockwise zero points must be integers or floats");
  }
  BlockwiseParameter parameter =
      MakeParameter(zero_points, tflite::TensorType_INT32);
  for (const APInt& value : *bits) {
    const int64_t wide =
        int_type.isUnsigned() ? value.getZExtValue() : value.getSExtValue();
    if (wide < std::numeric_limits<int32_t>::min() ||
        wide > std::numeric_limits<int32_t>::max()) {
      return absl::InvalidArgumentError(
          "blockwise zero point out of the int32 range");
    }
    Append(parameter.data, static_cast<int32_t>(wide));
  }
  return parameter;
}

}  // namespace mlir::TFL::fb_export
