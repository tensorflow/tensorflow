/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/utils/const_tensor_utils.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "Eigen/Core"  // from @eigen_archive
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"
#include "tensorflow/compiler/mlir/lite/utils/low_bit_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/string_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tsl/platform/statusor.h"

namespace mlir {
namespace TFL {
namespace {

using ::absl::StatusOr;
using ::mlir::Builder;
using ::mlir::quant::QuantizedType;
using ::tflite::TensorT;

// The buffers in TFLite flatbuffers have their contents stored as a vector of
// bytes that represent host endianness values.
// The read_size parameter is present to allow reading both float16 and float32
// without a case split.
template <typename T>
llvm::SmallVector<mlir::APInt> ReadAsHostEndian(ArrayRef<uint8_t> bytes) {
  llvm::SmallVector<mlir::APInt> ret;
  size_t read_size = sizeof(T);
  int bytes_len = bytes.size();
  assert(bytes_len % read_size == 0);

  int elem_count = bytes_len / read_size;
  ret.reserve(elem_count);

  const char* data_ptr = reinterpret_cast<const char*>(bytes.data());
  for (int i = 0; i < elem_count; i++) {
    T val = llvm::support::endian::readNext<T, llvm::endianness::native,
                                            llvm::support::unaligned>(data_ptr);
    ret.push_back(mlir::APInt(sizeof(T) * 8, val));
  }
  return ret;
}

// If the values in the buffer can be clamped to a bitwidth, truncate
// and return the new clamped integer width.
void truncateLimitedIntegerAPInt(llvm::SmallVector<mlir::APInt>& values) {
  mlir::APInt min = values[0];
  mlir::APInt max = values[0];
  for (mlir::APInt& val : values) {
    min = llvm::APIntOps::smin(val, min);
    max = llvm::APIntOps::smax(val, max);
  }

  for (int64_t bw = 8; bw < min.getBitWidth(); bw += bw) {
    mlir::APInt limitMin =
        mlir::APInt::getSignedMinValue(bw).sext(min.getBitWidth());
    mlir::APInt limitMax =
        mlir::APInt::getSignedMaxValue(bw).sext(min.getBitWidth());

    // Skips to the next bitwidth if the min and max values are out of the range
    // for the current bitwidth.
    if (min.sle(limitMin) || max.sle(limitMin) || min.sge(limitMax) ||
        max.sge(limitMax)) {
      continue;
    }

    for (mlir::APInt& val : values) {
      val = val.trunc(bw);
    }
    break;
  }
}

}  // namespace

bool IsQuantized(const TensorT& tensor) {
  return (tensor.quantization != nullptr) &&
         !tensor.quantization->zero_point.empty();
}

// Returns the correct type for a quantized tensor.
// We have a special case for constants since they have a higher minimum value.
StatusOr<QuantizedType> GetQuantizedType(const TensorT& tensor, Builder builder,
                                         bool is_constant,
                                         mlir::Type storage_type) {
  tflite::QuantizationParametersT& quant_params = *tensor.quantization;
  if (quant_params.details.AsCustomQuantization()) {
    return absl::UnimplementedError("Cannot handle experimental quantization");
  }

  bool is_signed = true;
  if (tensor.type == tflite::TensorType_UINT8) {
    is_signed = false;
    storage_type = mlir::IntegerType::get(builder.getContext(), 8);
  }

  if (!storage_type) {
    const mlir::Type raw_elem_type = ConvertElementType(tensor.type, builder);
    if (!mlir::isa<mlir::IntegerType>(raw_elem_type)) {
      return absl::InvalidArgumentError(
          "Quantized tensors must be stored as integers");
    }
    storage_type = mlir::cast<mlir::IntegerType>(raw_elem_type);
  }

  // TFlite uses narrow-range [u]int8 for constant buffers of quantized weights.
  // Since we don't know which ones are weights, we represent this optimization
  // as a change in the storage bounds for the type for all constants of this
  // type.
  const int bitwidth = storage_type.getIntOrFloatBitWidth();
  const bool is_weight_buffer = is_constant && (bitwidth == 8);

  int64_t storage_min =
      QuantizedType::getDefaultMinimumForInteger(is_signed, bitwidth) +
      static_cast<int>(is_weight_buffer);
  int64_t storage_max =
      QuantizedType::getDefaultMaximumForInteger(is_signed, bitwidth);
  uint32_t flags =
      is_signed ? mlir::quant::QuantizationFlags::FlagValue::Signed : 0;

  // Zero scales we make the minimum fp value, this is because some flatbuffers
  // contain zero scale for zero values.
  llvm::SmallVector<double> scales;
  for (float scale : quant_params.scale) {
    if (scale == 0) {
      scales.push_back(std::numeric_limits<float>::min());
      continue;
    }
    scales.push_back(scale);
  }

  // Scale size can't be zero as it is checked before.
  if (quant_params.scale.size() != 1) {
    return mlir::quant::UniformQuantizedPerAxisType::get(
        flags, storage_type, builder.getF32Type(), scales,
        quant_params.zero_point, quant_params.quantized_dimension, storage_min,
        storage_max);
  }
  return mlir::quant::UniformQuantizedType::get(
      flags, storage_type, builder.getF32Type(), scales[0],
      quant_params.zero_point.at(0), storage_min, storage_max);
}

StatusOr<QuantizedType> GetCalibratedQuantizedType(const TensorT& tensor,
                                                   Builder builder) {
  if (tensor.quantization == nullptr) {
    return absl::InvalidArgumentError("The tensor is not quantized.");
  }
  mlir::Type raw_elem_type = ConvertElementType(tensor.type, builder);
  float min = tensor.quantization->min[0];
  float max = tensor.quantization->max[0];
  return mlir::quant::CalibratedQuantizedType::get(raw_elem_type, min, max);
}

StatusOr<mlir::TensorType> GetTensorType(const TensorT& tensor, Builder builder,
                                         bool is_constant, bool is_intermediate,
                                         bool get_storage) {
  mlir::Type elem_type = ConvertElementType(tensor.type, builder);
  if (tensor.type == tflite::TensorType_VARIANT) {
    llvm::SmallVector<mlir::TensorType> tensor_types;
    if (tensor.variant_tensors.size() > 1) {
      return absl::InvalidArgumentError(
          "Have more than one nested type in `variant_tensors`.");
    }
    for (const auto& nested_tensor : tensor.variant_tensors) {
      mlir::Type nested_elem_type =
          ConvertElementType(nested_tensor->type, builder);
      if (nested_tensor->has_rank) {
        llvm::SmallVector<int64_t> shape(nested_tensor->shape.begin(),
                                         nested_tensor->shape.end());
        tensor_types.push_back(
            tensorflow::GetTypeFromTFTensorShape(shape, nested_elem_type));
      } else {
        tensor_types.push_back(UnrankedTensorType::get(nested_elem_type));
      }
    }
    elem_type = mlir::TF::VariantType::get(tensor_types, builder.getContext());
  }
  if (IsQuantized(tensor) && !get_storage) {
    TF_ASSIGN_OR_RETURN(elem_type,
                        GetQuantizedType(tensor, builder, is_constant));
  } else if (IsQuantized(tensor) && get_storage) {
    // If the type is quantized we strip the signedness from the storage type.
    elem_type = mlir::IntegerType::get(elem_type.getContext(),
                                       elem_type.getIntOrFloatBitWidth());
  }

  // Intermediate tensors with calibration value (but not scale and zero points)
  // should return calibrated quantized type.
  if (is_intermediate && tensor.quantization != nullptr &&
      !IsQuantized(tensor)) {
    TF_ASSIGN_OR_RETURN(elem_type, GetCalibratedQuantizedType(tensor, builder));
  }

  if (tensor.shape.empty() && (is_constant || tensor.has_rank)) {
    return RankedTensorType::get({}, elem_type);
  }

  if (!tensor.shape_signature.empty()) {
    llvm::SmallVector<int64_t, 4> shape(tensor.shape_signature.begin(),
                                        tensor.shape_signature.end());
    return tensorflow::GetTypeFromTFTensorShape(shape, elem_type);
  }

  if (!tensor.shape.empty()) {
    llvm::SmallVector<int64_t, 4> shape(tensor.shape.begin(),
                                        tensor.shape.end());
    return tensorflow::GetTypeFromTFTensorShape(shape, elem_type);
  }

  return UnrankedTensorType::get(elem_type);
}

mlir::ElementsAttr GetSplat(RankedTensorType type, int unique_index,
                            Builder builder) {
  mlir::Type element_ty = getElementTypeOrSelf(type);

  if (element_ty.isSignlessInteger())
    return DenseElementsAttr::get(
        type, builder.getIntegerAttr(element_ty, unique_index));

  if (mlir::isa<mlir::FloatType>(element_ty))
    return DenseElementsAttr::get(
        type, builder.getFloatAttr(element_ty, unique_index));

  if (auto qtype = mlir::dyn_cast<QuantizedType>(element_ty)) {
    mlir::RankedTensorType new_type = tensorflow::GetTypeFromTFTensorShape(
        type.getShape(), qtype.getStorageType());
    return DenseElementsAttr::get(
        new_type, builder.getIntegerAttr(qtype.getStorageType(), unique_index));
  }
  llvm_unreachable("unhandled element type");
}

StatusOr<mlir::ElementsAttr> ConvertIntBuffer(
    mlir::RankedTensorType shaped_type, const std::vector<uint8_t>& buffer,
    bool truncate) {
  mlir::Type elem_type = shaped_type.getElementType();
  unsigned bit_width;
  if (auto itype = mlir::dyn_cast<mlir::IntegerType>(elem_type)) {
    bit_width = itype.getWidth();
  } else if (auto qtype =
                 mlir::dyn_cast<mlir::quant::QuantizedType>(elem_type)) {
    bit_width = qtype.getStorageTypeIntegralWidth();
    shaped_type = tensorflow::GetTypeFromTFTensorShape(shaped_type.getShape(),
                                                       qtype.getStorageType());
  } else {
    return absl::InvalidArgumentError("unsupported integer constant type");
  }

  llvm::SmallVector<mlir::APInt> values;
  switch (bit_width) {
    case 1: {
      // vector<bool> doesn't convert to an ArrayRef
      llvm::SmallVector<bool, 8> boolValues;
      boolValues.reserve(buffer.size());
      for (auto b : buffer) {
        boolValues.emplace_back(b != 0);
      }
      return mlir::ElementsAttr(
          DenseElementsAttr::get(shaped_type, ArrayRef<bool>(boolValues)));
    }
    case 4: {
      auto i4Values =
          tflite::UnpackDenseInt4IntoInt8(buffer, shaped_type.getNumElements());
      // Use `getFromRawBuffer()` instead of `get()` to bypass a templated size
      // check which doesn't work with int4 because int4_t doesn't exist.
      return mlir::ElementsAttr(DenseElementsAttr::getFromRawBuffer(
          shaped_type, ArrayRef<char>(i4Values)));
    }
    case 8: {
      return mlir::ElementsAttr(
          DenseElementsAttr::get(shaped_type, ArrayRef<uint8_t>(buffer)));
    }
    case 16: {
      values = ReadAsHostEndian<uint16_t>(buffer);
      break;
    }
    case 32: {
      values = ReadAsHostEndian<uint32_t>(buffer);
      break;
    }
    case 64: {
      values = ReadAsHostEndian<uint64_t>(buffer);
      break;
    }
    default:
      return absl::UnimplementedError(
          absl::StrCat("Cannot handle bit width ", bit_width));
  }

  if (truncate) {
    truncateLimitedIntegerAPInt(values);
    auto sign = mlir::cast<mlir::IntegerType>(shaped_type.getElementType())
                    .getSignedness();
    auto ety = mlir::IntegerType::get(shaped_type.getContext(),
                                      values[0].getBitWidth(), sign);
    shaped_type =
        tensorflow::GetTypeFromTFTensorShape(shaped_type.getShape(), ety);
  }

  return mlir::ElementsAttr(DenseElementsAttr::get(shaped_type, values));
}

StatusOr<mlir::ElementsAttr> ConvertFloatBuffer(
    mlir::RankedTensorType shaped_type, const std::vector<uint8_t>& buffer) {
  size_t bytes_len = buffer.size();
  mlir::Type elem_type = shaped_type.getElementType();

  // The bytes of floats are stored little-endian.
  switch (elem_type.getIntOrFloatBitWidth()) {
    case 16: {
      assert(bytes_len % 2 == 0);
      // Supports both BF16 and F16.
      assert(elem_type.isF16() || elem_type.isBF16());
      int elem_count = bytes_len / 2;

      if (elem_type.isF16()) {
        std::vector<Eigen::half> values;
        values.reserve(elem_count);

        const char* data = reinterpret_cast<const char*>(buffer.data());

        for (int i = 0; i < elem_count; i++) {
          uint16_t bit_repr = llvm::support::endian::readNext<
              uint16_t, llvm::endianness::native, llvm::support::unaligned>(
              data);
          values.push_back(Eigen::numext::bit_cast<Eigen::half>(bit_repr));
        }

        return mlir::ElementsAttr(
            DenseElementsAttr::get(shaped_type, ArrayRef<Eigen::half>(values)));
      } else {
        std::vector<Eigen::bfloat16> values;
        values.reserve(elem_count);

        const char* data = reinterpret_cast<const char*>(buffer.data());

        for (int i = 0; i < elem_count; i++) {
          uint16_t bit_repr = llvm::support::endian::readNext<
              uint16_t, llvm::endianness::native, llvm::support::unaligned>(
              data);
          values.push_back(Eigen::numext::bit_cast<Eigen::bfloat16>(bit_repr));
        }

        return mlir::ElementsAttr(DenseElementsAttr::get(
            shaped_type, ArrayRef<Eigen::bfloat16>(values)));
      }
    }
    case 32: {
      assert(bytes_len % 4 == 0);
      int elem_count = bytes_len / 4;
      std::vector<float> values;
      values.reserve(elem_count);

      const char* data = reinterpret_cast<const char*>(buffer.data());

      for (int i = 0; i < elem_count; i++) {
        uint32_t bit_repr =
            llvm::support::endian::readNext<uint32_t, llvm::endianness::native,
                                            llvm::support::unaligned>(data);
        values.push_back(absl::bit_cast<float>(bit_repr));
      }
      return mlir::ElementsAttr(
          DenseElementsAttr::get(shaped_type, ArrayRef<float>(values)));
    }
    case 64: {
      assert(bytes_len % 8 == 0);
      int elem_count = bytes_len / 8;
      std::vector<double> values;
      values.reserve(elem_count);

      const char* data = reinterpret_cast<const char*>(buffer.data());

      for (int i = 0; i < elem_count; i++) {
        uint64_t bit_repr =
            llvm::support::endian::readNext<uint64_t, llvm::endianness::native,
                                            llvm::support::unaligned>(data);
        values.push_back(absl::bit_cast<double>(bit_repr));
      }
      return mlir::ElementsAttr(
          DenseElementsAttr::get(shaped_type, ArrayRef<double>(values)));
    }
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "unsupported bit width ", elem_type.getIntOrFloatBitWidth()));
}

tensorflow::TensorProto ConvertTfliteConstTensor(
    const TensorT& tensor, const std::vector<uint8_t>& buffer) {
  tensorflow::TensorProto ret;
  ret.set_dtype(TflTypeToTfType(tensor.type));

  tensorflow::TensorShapeProto* shape = ret.mutable_tensor_shape();
  shape->set_unknown_rank(false);
  for (auto dim : tensor.shape) {
    shape->add_dim()->set_size(int64_t{dim});
  }
  // TensorFlow Lite uses tflite::DynamicBufer to encode vector of strings.
  if (tensor.type == tflite::TensorType_STRING) {
    for (int i = 0; i < mlir::TFL::GetStringCount(buffer.data()); ++i) {
      mlir::TFL::StringRef str = mlir::TFL::GetString(buffer.data(), i);
      ret.add_string_val(str.str, str.len);
    }
    return ret;
  }
  std::string content;
  content.assign(reinterpret_cast<const char*>(buffer.data()), buffer.size());
  ret.set_tensor_content(content);
  return ret;
}

int64_t GetSizeInBits(mlir::ShapedType shaped_type) {
  return GetSizeInBits(shaped_type.getElementType()) *
         shaped_type.getNumElements();
}

int64_t GetSizeInBits(mlir::quant::QuantizedType quant_type) {
  const int64_t bits = std::max(quant_type.getStorageTypeIntegralWidth(),
                                static_cast<uint32_t>(CHAR_BIT));
  assert(IsPowerOfTwo(bits));
  return bits;
}

int64_t GetSizeInBits(mlir::Type type) {
  if (type.isIntOrFloat()) {
    const int64_t bits =
        std::max(type.getIntOrFloatBitWidth(), static_cast<uint32_t>(CHAR_BIT));
    assert(IsPowerOfTwo(bits));
    return bits;
  }
  if (mlir::isa<mlir::ShapedType>(type)) {
    auto shaped_type = mlir::cast<mlir::ShapedType>(type);
    if (mlir::isa<mlir::ComplexType>(shaped_type.getElementType())) {
      auto complex_type =
          mlir::cast<mlir::ComplexType>(shaped_type.getElementType());
      return GetSizeInBits(complex_type.getElementType()) * 2;
    } else if (mlir::isa<mlir::quant::QuantizedType>(
                   shaped_type.getElementType())) {
      auto quant_type =
          mlir::cast<mlir::quant::QuantizedType>(shaped_type.getElementType());
      return GetSizeInBits(quant_type);
    } else {
      return GetSizeInBits(shaped_type);
    }
  }

  return 0;
}

int64_t GetSizeInBytes(mlir::Type type) {
  return ExactIntegerDivide(GetSizeInBits(type), CHAR_BIT);
}

}  // namespace TFL
}  // namespace mlir
