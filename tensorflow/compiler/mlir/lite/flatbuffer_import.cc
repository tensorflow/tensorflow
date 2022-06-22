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

#include "tensorflow/compiler/mlir/lite/flatbuffer_import.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/flatbuffer_operator.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "tensorflow/lite/string_util.h"

using llvm::ArrayRef;
using mlir::Builder;
using mlir::DenseElementsAttr;
using mlir::Location;
using mlir::MLIRContext;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OperationState;
using mlir::OwningOpRef;
using mlir::RankedTensorType;
using mlir::UnrankedTensorType;
using mlir::Value;
using mlir::func::FuncOp;
using mlir::quant::QuantizedType;
using tflite::OperatorT;
using tflite::TensorT;
using xla::Status;
using xla::StatusOr;

namespace errors = tensorflow::errors;
namespace tfl = mlir::TFL;

namespace {
bool IsScalar(const TensorT& tensor) {
  // TODO(b/138222071) We can't distinguish scalars and unranked tensors
  // Work out a way to handle this and stub out the code until then
  return tensor.shape.empty() && false;
}

bool IsQuantized(const TensorT& tensor) {
  return (tensor.quantization != nullptr) &&
         !tensor.quantization->zero_point.empty();
}

// Create the MLIR NamedLoc location corresponding to a given tensor
Location TensorLoc(const TensorT& tensor, Builder builder, Location base) {
  if (tensor.name.empty()) {
    return base;
  }
  return mlir::NameLoc::get(builder.getStringAttr(tensor.name), base);
}

// Create the MLIR Location corresponding to a given op. This is an
// experimental/debugging feature and production code should not rely on names
// of intermediate tensors since importer doesn't guarantee to preserve tensor
// names except output tensors.
Location OpLoc(const OperatorT& op,
               const std::vector<std::unique_ptr<tflite::TensorT>>& tensors,
               Builder builder, Location base) {
  if (op.outputs.empty()) return base;

  llvm::SmallVector<Location, 4> locations;
  locations.reserve(op.outputs.size());
  for (auto tensor_index : op.outputs) {
    locations.push_back(TensorLoc(*tensors[tensor_index], builder, base));
  }
  return mlir::FusedLoc::get(builder.getContext(), locations);
}

// Returns the correct type for a quantized tensor
// We have a special case for constants since they have a higher minimum value.
StatusOr<QuantizedType> GetQuantizedType(const TensorT& tensor, Builder builder,
                                         bool is_constant = false) {
  tflite::QuantizationParametersT& quant_params = *tensor.quantization;
  if (quant_params.details.AsCustomQuantization()) {
    return errors::Unimplemented("Cannot handle experimental quantization");
  }

  bool is_signed = true;
  mlir::IntegerType storage_type;
  if (tensor.type == tflite::TensorType_UINT8) {
    is_signed = false;
    storage_type = builder.getIntegerType(8);
  } else {
    auto raw_elem_type = ConvertElementType(tensor.type, builder);
    if (!raw_elem_type.isa<mlir::IntegerType>()) {
      return errors::InvalidArgument(
          "Quantized tensors must be stored as integers");
    }
    storage_type = raw_elem_type.cast<mlir::IntegerType>();
  }

  // TFlite uses narrow-range [u]int8 for constant buffers of quantized weights.
  // Since we don't know which ones are weights, we represent this optimization
  // as a change in the storage bounds for the type for all constants of this
  // type.
  bool is_weight_buffer = is_constant && (storage_type.getWidth() == 8);

  int64_t storage_min = QuantizedType::getDefaultMinimumForInteger(
                            is_signed, storage_type.getWidth()) +
                        static_cast<int>(is_weight_buffer);
  int64_t storage_max = QuantizedType::getDefaultMaximumForInteger(
      is_signed, storage_type.getWidth());
  uint32_t flags =
      is_signed ? mlir::quant::QuantizationFlags::FlagValue::Signed : 0;

  // Rejects if quantized tensors have zero scales.
  for (float scale : quant_params.scale) {
    if (scale == 0) {
      return errors::InvalidArgument(
          "Quantized tensors must have non-zero scales");
    }
  }

  // Scale size can't be zero as it is checked before.
  if (quant_params.scale.size() != 1) {
    llvm::SmallVector<double, 4> scales(quant_params.scale.begin(),
                                        quant_params.scale.end());
    return mlir::quant::UniformQuantizedPerAxisType::get(
        flags, storage_type, builder.getF32Type(), scales,
        quant_params.zero_point, quant_params.quantized_dimension, storage_min,
        storage_max);
  }
  return mlir::quant::UniformQuantizedType::get(
      flags, storage_type, builder.getF32Type(), quant_params.scale.at(0),
      quant_params.zero_point.at(0), storage_min, storage_max);
}

// import float tensor with calibration value into calibrated quantized type.
StatusOr<QuantizedType> GetCalibratedQuantizedType(const TensorT& tensor,
                                                   Builder builder) {
  if (tensor.quantization == nullptr) {
    return errors::InvalidArgument("The tensor is not quantized.");
  }
  auto raw_elem_type = ConvertElementType(tensor.type, builder);
  float min = tensor.quantization->min[0];
  float max = tensor.quantization->max[0];
  return mlir::quant::CalibratedQuantizedType::get(raw_elem_type, min, max);
}

// TODO(b/138222071) Remove shapeless_are_scalars once we can reliably
// make that distinction and don't have to rely on context
// (input to main and constants must have static shape)
StatusOr<mlir::TensorType> GetTensorType(const TensorT& tensor, Builder builder,
                                         bool shapeless_are_scalars = false,
                                         bool is_constant = false,
                                         bool is_intermediate = false) {
  mlir::Type elem_type = ConvertElementType(tensor.type, builder);
  if (IsQuantized(tensor)) {
    TF_ASSIGN_OR_RETURN(elem_type,
                        GetQuantizedType(tensor, builder, is_constant));
  }

  // Intermediate tensors with calibration value (but not scale and zero points)
  // should return calibrated quantized type.
  if (is_intermediate && tensor.quantization != nullptr &&
      !IsQuantized(tensor)) {
    TF_ASSIGN_OR_RETURN(elem_type, GetCalibratedQuantizedType(tensor, builder));
  }

  if (IsScalar(tensor) || (shapeless_are_scalars && tensor.shape.empty())) {
    return RankedTensorType::get({}, elem_type);
  }

  if (!tensor.shape_signature.empty()) {
    llvm::SmallVector<int64_t, 4> shape(tensor.shape_signature.begin(),
                                        tensor.shape_signature.end());
    return RankedTensorType::get(shape, elem_type);
  }

  if (!tensor.shape.empty()) {
    llvm::SmallVector<int64_t, 4> shape(tensor.shape.begin(),
                                        tensor.shape.end());
    return RankedTensorType::get(shape, elem_type);
  }

  return UnrankedTensorType::get(elem_type);
}

// Extract the min max information in the tensor and create the quant stats op.
// If the input `tensor` has scale/zero_point, `res` should have quantized
// type, thus none stats op is required and nullptr is retruned.
// If the min max information is invalid, nullptr is returned.
mlir::Operation* ConvertMinMaxToStatsOp(const TensorT& tensor, OpBuilder b,
                                        Value res) {
  // If the `tensor` has scale/zero_point, it must have been quantized, then the
  // min/max stats is just for comments, so ignore it.
  if (!tensor.quantization || IsQuantized(tensor)) return nullptr;
  // If the result isn't float and unquantizable, the min/max is ignored.
  if (!res.getType()
           .cast<mlir::ShapedType>()
           .getElementType()
           .isa<mlir::FloatType>()) {
    return nullptr;
  }
  auto mins = tensor.quantization->min;
  auto maxs = tensor.quantization->max;
  if (mins.size() != maxs.size() || mins.empty()) return nullptr;

  llvm::SmallVector<llvm::APFloat, 4> min_maxs;
  min_maxs.reserve(mins.size() * 2);
  for (int i = 0, end = mins.size(); i < end; ++i) {
    llvm::APFloat min(mins[i]);
    llvm::APFloat max(maxs[i]);
    min_maxs.push_back(min);
    min_maxs.push_back(max);
  }
  // The layer stats contain only the first min/max pairs.
  mlir::ElementsAttr layer_stats = mlir::DenseFPElementsAttr::get(
      mlir::RankedTensorType::get({2}, b.getF32Type()),
      {min_maxs[0], min_maxs[1]});
  mlir::ElementsAttr axis_stats;
  mlir::IntegerAttr axis;
  if (mins.size() > 1) {
    llvm::SmallVector<int64_t, 4> axis_stats_shape{
        static_cast<int64_t>(mins.size()), 2};
    axis_stats = mlir::DenseFPElementsAttr::get(
        mlir::RankedTensorType::get(axis_stats_shape, b.getF32Type()),
        min_maxs);
    // TODO(fengliuai): this quantization dimension isn't correct.
    axis = b.getI64IntegerAttr(tensor.quantization->quantized_dimension);
  }
  return b.create<mlir::quant::StatisticsOp>(b.getUnknownLoc(), res,
                                             layer_stats, axis_stats, axis);
}

// Returns true if this is a basic LSTM op.
bool IsBasicLSTMOp(tflite::BuiltinOptionsUnion op_union) {
  if (const auto* op = op_union.AsLSTMOptions()) {
    return op->kernel_type == tflite::LSTMKernelType_BASIC;
  } else {
    return false;
  }
}

// Gets the MLIR op name with the dialect name for the flatbuffer operator.
std::string GetMlirOpName(const tflite::OperatorT& op,
                          const tflite::OperatorCodeT& op_code) {
  if (IsBasicLSTMOp(op.builtin_options)) {
    return std::string("tfl.basic_lstm");
  }
  return mlir::GetMlirOpNameFromOpCode(op_code);
}

// The buffers in TFLite flatbuffers have their contents stored as a vector of
// bytes that represent little-endian values.
// The read_size parameter is present to allow reading both float16 and float32s
// without a case split.
template <typename T>
std::vector<T> ReadAsLittleEndian(ArrayRef<uint8_t> bytes) {
  std::vector<T> ret;
  size_t read_size = sizeof(T);
  int bytes_len = bytes.size();
  assert(bytes_len % read_size == 0);

  int elem_count = bytes_len / read_size;
  ret.reserve(elem_count);

  const char* data_ptr = reinterpret_cast<const char*>(bytes.data());
  for (int i = 0; i < elem_count; i++) {
    ret.push_back(
        llvm::support::endian::readNext<T, llvm::support::little,
                                        llvm::support::unaligned>(data_ptr));
  }
  return ret;
}

tensorflow::TensorProto ConvertTfliteConstTensor(
    const tflite::TensorT& tensor, const std::vector<uint8_t>& buffer) {
  tensorflow::TensorProto ret;
  ret.set_dtype(TflTypeToTfType(tensor.type));

  tensorflow::TensorShapeProto* shape = ret.mutable_tensor_shape();
  shape->set_unknown_rank(false);
  for (auto dim : tensor.shape) {
    shape->add_dim()->set_size(int64_t{dim});
  }
  // TensorFlow Lite uses tflite::DynamicBufer to encode vector of strings.
  if (tensor.type == tflite::TensorType_STRING) {
    for (int i = 0; i < tflite::GetStringCount(buffer.data()); ++i) {
      tflite::StringRef str = tflite::GetString(buffer.data(), i);
      ret.add_string_val(str.str, str.len);
    }
    return ret;
  }
  std::string content;
  content.assign(reinterpret_cast<const char*>(buffer.data()), buffer.size());
  ret.set_tensor_content(content);
  return ret;
}

StatusOr<mlir::ElementsAttr> ConvertFloatBuffer(
    mlir::RankedTensorType shaped_type, mlir::FloatType elem_type,
    const std::vector<uint8_t>& buffer) {
  size_t bytes_len = buffer.size();

  // The bytes of floats are stored little-endian.
  switch (elem_type.getWidth()) {
    case 16: {
      assert(bytes_len % 2 == 0);
      int elem_count = bytes_len / 2;
      std::vector<llvm::APFloat> values;
      values.reserve(elem_count);

      const char* data = reinterpret_cast<const char*>(buffer.data());
      auto& semantics = elem_type.getFloatSemantics();

      for (int i = 0; i < elem_count; i++) {
        uint16_t bit_repr =
            llvm::support::endian::readNext<uint16_t, llvm::support::little,
                                            llvm::support::unaligned>(data);
        llvm::APInt int_repr(16, bit_repr);
        values.emplace_back(semantics, int_repr);
      }

      return mlir::ElementsAttr(DenseElementsAttr::get(shaped_type, values));
    }
    case 32: {
      assert(bytes_len % 4 == 0);
      int elem_count = bytes_len / 4;
      std::vector<float> values;
      values.reserve(elem_count);

      const char* data = reinterpret_cast<const char*>(buffer.data());

      for (int i = 0; i < elem_count; i++) {
        uint32_t bit_repr =
            llvm::support::endian::readNext<uint32_t, llvm::support::little,
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
            llvm::support::endian::readNext<uint64_t, llvm::support::little,
                                            llvm::support::unaligned>(data);
        values.push_back(absl::bit_cast<double>(bit_repr));
      }
      return mlir::ElementsAttr(
          DenseElementsAttr::get(shaped_type, ArrayRef<double>(values)));
    }
  }
  return errors::InvalidArgument("unsupported bit width", elem_type.getWidth());
}

StatusOr<mlir::ElementsAttr> ConvertIntBuffer(
    mlir::RankedTensorType shaped_type, mlir::Type elem_type,
    const std::vector<uint8_t>& buffer) {
  unsigned bit_width;
  if (auto itype = elem_type.dyn_cast<mlir::IntegerType>()) {
    bit_width = itype.getWidth();
  } else if (auto qtype = elem_type.dyn_cast<QuantizedType>()) {
    bit_width = qtype.getStorageTypeIntegralWidth();
    shaped_type = mlir::RankedTensorType::get(shaped_type.getShape(),
                                              qtype.getStorageType());
  } else {
    return errors::InvalidArgument("unsupported integer constant type");
  }

  switch (bit_width) {
    case 1: {
      // vector<bool> doesn't convert to an ArrayRef
      llvm::SmallVector<bool, 8> values;
      values.reserve(buffer.size());
      for (auto b : buffer) {
        values.emplace_back(b != 0);
      }
      return mlir::ElementsAttr(
          DenseElementsAttr::get(shaped_type, ArrayRef<bool>(values)));
    }
    case 8: {
      return mlir::ElementsAttr(
          DenseElementsAttr::get(shaped_type, ArrayRef<uint8_t>(buffer)));
    }
    case 16: {
      auto values = ReadAsLittleEndian<uint16_t>(buffer);
      return mlir::ElementsAttr(
          DenseElementsAttr::get(shaped_type, ArrayRef<uint16_t>(values)));
    }
    case 32: {
      auto values = ReadAsLittleEndian<uint32_t>(buffer);
      return mlir::ElementsAttr(
          DenseElementsAttr::get(shaped_type, ArrayRef<uint32_t>(values)));
    }
    case 64: {
      auto values = ReadAsLittleEndian<uint64_t>(buffer);
      return mlir::ElementsAttr(
          DenseElementsAttr::get(shaped_type, ArrayRef<uint64_t>(values)));
    }
    default:
      return errors::Unimplemented("Cannot handle bit width ", bit_width);
  }
}

StatusOr<Operation*> BuildExternalConstOp(const tflite::TensorT& tensor,
                                          int32_t buffer_index,
                                          OpBuilder builder, Location loc) {
  TF_ASSIGN_OR_RETURN(auto type, GetTensorType(tensor, builder,
                                               /*shapeless_are_scalars=*/true,
                                               /*is_constant=*/true));
  auto shaped_type = type.dyn_cast<mlir::RankedTensorType>();
  if (!shaped_type) {
    return errors::Internal("Constant doesn't have a shape");
  }
  auto op = builder.create<tfl::ExternalConstOp>(
      loc, shaped_type, builder.getI32IntegerAttr(buffer_index));
  return op.getOperation();
}

// Gets a constant splat for the given value of type. Requires value to be of
// type static shaped RankedTensorType. `unique_index` is used to get the unique
// value for the attribute.
static mlir::ElementsAttr GetSplat(RankedTensorType type, int unique_index,
                                   OpBuilder builder) {
  mlir::Type element_ty = getElementTypeOrSelf(type);

  if (element_ty.isSignlessInteger())
    return DenseElementsAttr::get(
        type, builder.getIntegerAttr(element_ty, unique_index));

  if (element_ty.isa<mlir::FloatType>())
    return DenseElementsAttr::get(
        type, builder.getFloatAttr(element_ty, unique_index));

  if (auto qtype = element_ty.dyn_cast<QuantizedType>()) {
    mlir::RankedTensorType new_type =
        RankedTensorType::get(type.getShape(), qtype.getStorageType());
    return DenseElementsAttr::get(
        new_type, builder.getIntegerAttr(qtype.getStorageType(), unique_index));
  }
  llvm_unreachable("unhandled element type");
}

// TODO(b/172664358): Creates a new op instead of reusing constant op.
// Creates a constant op to represent stateful variable. The function static
// variable `stateful_variable_idx` is used as a unique value for each constant
// to avoid CSEed. `tensor` is the data structure of flatbuffer. `shaped_type`
// is the ShapedType for the const op.
Operation* BuildVariableOp(const tflite::TensorT& tensor,
                           mlir::RankedTensorType shaped_type,
                           OpBuilder builder, Location loc) {
  static int stateful_variable_idx = 0;
  mlir::ElementsAttr value =
      GetSplat(shaped_type, stateful_variable_idx++, builder);
  if (IsQuantized(tensor)) {
    auto op = builder.create<tfl::QConstOp>(
        loc, mlir::TypeAttr::get(shaped_type), value);
    return op.getOperation();
  }
  auto op = builder.create<tfl::ConstOp>(loc, value);
  if (tensor.quantization && !tensor.quantization->min.empty()) {
    if (auto stats_op =
            ConvertMinMaxToStatsOp(tensor, builder, op.getResult())) {
      return stats_op;
    }
  }
  return op.getOperation();
}

static StatusOr<mlir::ArrayAttr> ConvertSparseIndexVector(
    const tflite::SparseIndexVectorUnion& sparse_index_vector,
    OpBuilder& builder) {
  if (sparse_index_vector.type == tflite::SparseIndexVector_Int32Vector) {
    return builder.getI32ArrayAttr(sparse_index_vector.AsInt32Vector()->values);
  } else if (sparse_index_vector.type ==
             tflite::SparseIndexVector_Uint16Vector) {
    const auto& inputs = sparse_index_vector.AsUint16Vector()->values;
    std::vector<int32_t> outputs(inputs.size());
    std::transform(inputs.begin(), inputs.end(), outputs.begin(),
                   [](auto x) { return static_cast<int32_t>(x); });
    return builder.getI32ArrayAttr(outputs);
  } else if (sparse_index_vector.type ==
             tflite::SparseIndexVector_Uint8Vector) {
    const auto& inputs = sparse_index_vector.AsUint8Vector()->values;
    std::vector<int32_t> outputs(inputs.size());
    std::transform(inputs.begin(), inputs.end(), outputs.begin(),
                   [](auto x) { return static_cast<int32_t>(x); });
    return builder.getI32ArrayAttr(outputs);
  } else {
    return errors::Unimplemented("Unsupported SparseIndexVector type");
  }
}

static StatusOr<Operation*> BuildSparseConstOp(
    const tflite::TensorT& tensor, const std::vector<uint8_t>& buffer,
    const mlir::RankedTensorType shaped_type, OpBuilder& builder,
    Location loc) {
  tensorflow::TensorProto repr = ConvertTfliteConstTensor(tensor, buffer);
  repr.clear_tensor_shape();
  if (IsQuantized(tensor)) {
    repr.mutable_tensor_shape()->add_dim()->set_size(buffer.size());
    repr.set_dtype(tensorflow::DT_INT8);
  } else {
    repr.mutable_tensor_shape()->add_dim()->set_size(
        buffer.size() / (shaped_type.getElementTypeBitWidth() / CHAR_BIT));
  }
  TF_ASSIGN_OR_RETURN(mlir::ElementsAttr compressed_data,
                      tensorflow::ConvertTensorProto(repr, &builder));

  const int dim_metadata_size = tensor.sparsity->dim_metadata.size();
  std::vector<mlir::Attribute> dim_metadata(dim_metadata_size);
  for (int i = 0; i < dim_metadata_size; i++) {
    if (tensor.sparsity->dim_metadata[i]->format ==
        tflite::DimensionType_DENSE) {
      dim_metadata[i] = tfl::DimensionMetadataAttr::get(
          mlir::TFL::DimensionTypeAttr::get(builder.getContext(),
                                            tfl::DimensionType::DENSE),
          builder.getI32IntegerAttr(
              tensor.sparsity->dim_metadata[i]->dense_size),
          builder.getI32ArrayAttr({}), builder.getI32ArrayAttr({}),
          builder.getContext());
    } else if (tensor.sparsity->dim_metadata[i]->format ==
               tflite::DimensionType_SPARSE_CSR) {
      TF_ASSIGN_OR_RETURN(
          mlir::ArrayAttr segments,
          ConvertSparseIndexVector(
              tensor.sparsity->dim_metadata[i]->array_segments, builder));
      TF_ASSIGN_OR_RETURN(
          mlir::ArrayAttr indices,
          ConvertSparseIndexVector(
              tensor.sparsity->dim_metadata[i]->array_indices, builder));
      dim_metadata[i] = tfl::DimensionMetadataAttr::get(
          mlir::TFL::DimensionTypeAttr::get(builder.getContext(),
                                            tfl::DimensionType::SPARSE_CSR),
          builder.getI32IntegerAttr(0), segments, indices,
          builder.getContext());
    } else {
      return errors::Unimplemented("Unsupported dimension metadata type");
    }
  }
  auto s_param = tfl::SparsityParameterAttr::get(
      builder.getI32ArrayAttr(tensor.sparsity->traversal_order),
      builder.getI32ArrayAttr(tensor.sparsity->block_map),
      builder.getArrayAttr(dim_metadata), builder.getContext());

  auto value_type = shaped_type;
  if (IsQuantized(tensor)) {
    value_type = RankedTensorType::get(
        shaped_type.getShape(), shaped_type.getElementType()
                                    .dyn_cast<mlir::quant::QuantizedType>()
                                    .getStorageType());
  }
  std::vector<char> dense_buffer(
      value_type.getElementType().getIntOrFloatBitWidth() / CHAR_BIT);
  mlir::Attribute dummy_value =
      mlir::DenseIntOrFPElementsAttr::getFromRawBuffer(value_type,
                                                       dense_buffer);

  if (IsQuantized(tensor)) {
    return builder
        .create<tfl::SparseQConstOp>(loc, mlir::TypeAttr::get(shaped_type),
                                     dummy_value, s_param, compressed_data)
        .getOperation();
  }
  return builder
      .create<tfl::SparseConstOp>(loc, dummy_value, s_param, compressed_data)
      .getOperation();
}

StatusOr<Operation*> BuildConstOp(const tflite::TensorT& tensor,
                                  const std::vector<uint8_t>& buffer,
                                  bool is_variable, OpBuilder builder,
                                  Location loc) {
  TF_ASSIGN_OR_RETURN(auto type, GetTensorType(tensor, builder,
                                               /*shapeless_are_scalars=*/true,
                                               /*is_constant=*/true));
  auto shaped_type = type.dyn_cast<mlir::RankedTensorType>();
  if (!shaped_type) {
    return errors::Internal("Constant doesn't have a shape");
  }

  if (tensor.sparsity != nullptr) {
    return BuildSparseConstOp(tensor, buffer, shaped_type, builder, loc);
  }

  auto elem_type = shaped_type.getElementType();

  mlir::ElementsAttr value;
  if (is_variable) {
    return BuildVariableOp(tensor, shaped_type, builder, loc);
  } else if (auto float_type = elem_type.dyn_cast<mlir::FloatType>()) {
    TF_ASSIGN_OR_RETURN(value,
                        ConvertFloatBuffer(shaped_type, float_type, buffer));
  } else if (elem_type.isa<mlir::IntegerType, QuantizedType>()) {
    TF_ASSIGN_OR_RETURN(value,
                        ConvertIntBuffer(shaped_type, elem_type, buffer));
  } else if (elem_type.isa<mlir::TF::StringType>()) {
    tensorflow::TensorProto repr = ConvertTfliteConstTensor(tensor, buffer);
    std::vector<llvm::StringRef> refs;
    refs.reserve(repr.string_val_size());

    for (const auto& ref : repr.string_val())
      refs.push_back({ref.data(), ref.size()});

    value = mlir::DenseStringElementsAttr::get(shaped_type, refs);
  } else if (elem_type.isa<mlir::ComplexType, mlir::TF::TensorFlowType>()) {
    auto dialect = elem_type.getContext()->getLoadedDialect("tf");
    tensorflow::TensorProto repr = ConvertTfliteConstTensor(tensor, buffer);
    std::string mangled = tensorflow::mangling_util::MangleTensor(repr);

    value = mlir::OpaqueElementsAttr::get(dialect, shaped_type, mangled);
  } else {
    return errors::Unimplemented("Constant of unsupported type");
  }

  if (IsQuantized(tensor)) {
    auto op = builder.create<tfl::QConstOp>(
        loc, mlir::TypeAttr::get(shaped_type), value);
    return op.getOperation();
  }
  auto op = builder.create<tfl::ConstOp>(loc, value);
  return op.getOperation();
}

StatusOr<llvm::SmallVector<mlir::NamedAttribute, 4>>
ConvertSubgraphIdxsToFunctionAttrs(tflite::BuiltinOptionsUnion options,
                                   const std::vector<std::string>& func_names,
                                   Builder builder) {
  if (auto* opts = options.AsCallOnceOptions()) {
    uint32_t init_idx = opts->init_subgraph_index;
    if (init_idx >= func_names.size()) {
      return errors::InvalidArgument("subgraph with index not found: ",
                                     init_idx);
    }
    auto init_attr = builder.getStringAttr(func_names.at(init_idx));

    return llvm::SmallVector<mlir::NamedAttribute, 4>{
        builder.getNamedAttr("session_init_function", init_attr)};
  }
  if (auto* opts = options.AsIfOptions()) {
    uint32_t then_idx = opts->then_subgraph_index;
    if (then_idx >= func_names.size()) {
      return errors::InvalidArgument("subgraph with index not found: ",
                                     then_idx);
    }
    auto then_attr =
        mlir::SymbolRefAttr::get(builder.getContext(), func_names.at(then_idx));
    uint32_t else_idx = opts->else_subgraph_index;
    if (else_idx >= func_names.size()) {
      return errors::InvalidArgument("subgraph with index not found: ",
                                     else_idx);
    }
    auto else_attr =
        mlir::SymbolRefAttr::get(builder.getContext(), func_names.at(else_idx));

    return llvm::SmallVector<mlir::NamedAttribute, 4>{
        builder.getNamedAttr("then_branch", then_attr),
        builder.getNamedAttr("else_branch", else_attr),
        // TODO(b/139667752): Analyze statelessness correctly
        builder.getNamedAttr("is_stateless", builder.getBoolAttr(false))};
  }
  if (auto* opts = options.AsWhileOptions()) {
    uint32_t cond_idx = opts->cond_subgraph_index;
    if (cond_idx >= func_names.size()) {
      return errors::InvalidArgument("subgraph with index not found: ",
                                     cond_idx);
    }
    auto cond_attr =
        mlir::SymbolRefAttr::get(builder.getContext(), func_names.at(cond_idx));
    uint32_t body_idx = opts->body_subgraph_index;
    if (body_idx >= func_names.size()) {
      return errors::InvalidArgument("subgraph with index not found: ",
                                     body_idx);
    }
    auto body_attr =
        mlir::SymbolRefAttr::get(builder.getContext(), func_names.at(body_idx));

    return llvm::SmallVector<mlir::NamedAttribute, 4>{
        builder.getNamedAttr("cond", cond_attr),
        builder.getNamedAttr("body", body_attr)};
  }
  return llvm::SmallVector<mlir::NamedAttribute, 4>{};
}

Status AddOpIntermediatesForLstm(
    const tflite::OperatorT& op,
    const std::vector<mlir::TensorType>& intermediate_types,
    OperationState& op_state, Location loc, OpBuilder& builder) {
  if (!op.intermediates.empty()) {
    if (op.intermediates.size() != 5) {
      auto err = errors::InvalidArgument(
          "operator has intermediate tensors but the number of them is not "
          "five.");
      return emitError(loc, err.ToString()), err;
    }
    // Create intermediate value

    const llvm::SmallVector<llvm::StringRef, 5> kIntermediateNames = {
        "input_to_input_intermediate", "input_to_forget_intermediate",
        "input_to_cell_intermediate", "input_to_output_intermediate",
        "effective_hidden_scale_intermediate"};
    for (auto type_and_name :
         llvm::zip(intermediate_types, kIntermediateNames)) {
      mlir::TypeAttr type_attr =
          mlir::TypeAttr::get(std::get<0>(type_and_name));
      auto named_attr =
          builder.getNamedAttr(std::get<1>(type_and_name), type_attr);
      op_state.addAttribute(named_attr.getName(), named_attr.getValue());
    }
  }
  return ::tensorflow::OkStatus();
}

// TODO(krzysd) Handle function calls
StatusOr<Operation*> ConvertOp(
    const tflite::OperatorT& op, const std::vector<Value>& vals_map,
    const std::vector<mlir::TensorType>& intermediate_types,
    Value optional_arg_marker,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& op_codes,
    const std::vector<std::string>& func_names,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tensors, Location loc,
    OpBuilder builder) {
  llvm::SmallVector<Value, 4> operands;
  llvm::SmallVector<mlir::Type, 2> outputTypes;

  const tflite::OperatorCodeT& op_code = *op_codes.at(op.opcode_index);

  const std::string op_name = GetMlirOpName(op, op_code);

  OperationState op_state(loc, op_name);

  for (auto input_num : op.inputs) {
    if (input_num == -1) {
      assert(optional_arg_marker != nullptr);
      op_state.addOperands({optional_arg_marker});
    } else {
      op_state.addOperands({vals_map.at(input_num)});
    }
  }

  for (auto output_num : op.outputs) {
    auto& tensor = *tensors.at(output_num);
    auto type_or_err = GetTensorType(tensor, builder);
    if (!type_or_err.ok()) {
      return emitError(loc, type_or_err.status().ToString()),
             type_or_err.status();
    }
    auto type = type_or_err.ConsumeValueOrDie();

    if (op_name == "tfl.quantize") {
      // Special case for quantize: return type must also be in qtype attribute
      op_state.addAttribute("qtype", mlir::TypeAttr::get(type));
    } else if (op_name == "tfl.reshape" && op_state.operands.size() == 1) {
      // Special case for reshape: the second op is optional in the old
      // converter and kernel, so we create the second operand, which is
      // required by the new converter, from the reshape op's option.
      auto new_shape = op.builtin_options.AsReshapeOptions()->new_shape;
      auto shape_type = RankedTensorType::get(
          {static_cast<int64_t>(new_shape.size())}, builder.getIntegerType(32));

      mlir::SmallVector<mlir::Attribute, 4> shape;
      for (auto s : new_shape) {
        shape.push_back(builder.getI32IntegerAttr(static_cast<int32_t>(s)));
      }
      auto output_shape = DenseElementsAttr::get(shape_type, shape);
      auto shape_op = builder.create<tfl::ConstOp>(loc, output_shape);
      op_state.addOperands({shape_op});
    }

    op_state.addTypes({type});
  }

  // While the last several tensors could be optional tensors for an tfl op, the
  // number of input operands could vary. Gets the min/max number of
  // operands from tflite op name.
  // Also, since the above code special-handles the `tfl.reshape` op and add an
  // additional input, we put these function block here.
  llvm::MinMax input_min_max = mlir::OperandNumbersMinMax(op_name);
  int input_max_num = input_min_max.Max;
  int op_input_num = op_state.operands.size();
  if (input_max_num != 0 && input_max_num > op_input_num) {
    // If the number of current inputs is less than the op definition, fill in
    // with `none` value,
    llvm::SmallVector<Value, 4> none_operands(
        input_max_num - op_input_num,
        builder.create<mlir::TFL::NoValueOp>(loc, builder.getNoneType(),
                                             builder.getUnitAttr()));
    op_state.addOperands(ArrayRef<Value>(none_operands));
  }

  if (op_name == "tfl.lstm") {
    // TODO(b/147587779): add the right region if region is empty.
    op_state.addRegion();
    TF_CHECK_OK(AddOpIntermediatesForLstm(op, intermediate_types, op_state, loc,
                                          builder));
  }
  if (op_name == "tfl.while") {
    // Adds two empty regions for "tfl.while". We will fill the regions after
    // creating the callee functions because the "tfl.while" input/output types
    // may be different with the callee functions, and the call ops need to sync
    // with callee function types.
    op_state.addRegion();
    op_state.addRegion();
  }
  if (op_name == "tfl.unidirectional_sequence_lstm") {
    TF_CHECK_OK(AddOpIntermediatesForLstm(op, intermediate_types, op_state, loc,
                                          builder));
  }
  if (op_name == "tfl.reshape") {
    // Flattern reshape ops when more than one dimension shape operand is given.
    mlir::DenseIntElementsAttr shape_attr;
    if (matchPattern(op_state.operands[1], m_Constant(&shape_attr))) {
      auto shape_ty =
          op_state.operands[1].getType().dyn_cast<RankedTensorType>();
      if (shape_ty != nullptr && shape_ty.hasRank() && shape_ty.getRank() > 1) {
        llvm::SmallVector<mlir::Attribute, 4> shape;
        int32_t dim_size = 0;
        for (const auto& dim :
             llvm::enumerate(shape_attr.getValues<llvm::APInt>())) {
          const int64_t size = dim.value().getSExtValue();
          shape.push_back(
              builder.getI32IntegerAttr(static_cast<int32_t>(size)));
          ++dim_size;
        }
        auto shape_type = RankedTensorType::get(
            {static_cast<int32_t>(dim_size)}, builder.getIntegerType(32));
        auto output_shape = mlir::DenseElementsAttr::get(shape_type, shape);
        auto shape_op = builder.create<tfl::ConstOp>(loc, output_shape);
        op_state.operands[1] = shape_op;
      }
    }
  }

  llvm::SmallVector<mlir::NamedAttribute, 2> attrs;
  auto builtin_code = tflite::GetBuiltinCode(&op_code);
  if (builtin_code == tflite::BuiltinOperator_CUSTOM) {
    auto status = mlir::CustomOptionsToAttributes(
        op_code.custom_code, op.custom_options, builder, loc, &attrs);
    if (!status.ok()) {
      return emitError(loc, status.ToString()), status;
    }
  } else {
    mlir::BuiltinOptionsToAttributes(op.builtin_options, builder, attrs);
  }
  op_state.addAttributes(attrs);

  // Handle the conversion from subgraph index to functions for If and While. We
  // will add CallOps in the region to call the functions later for While.
  TF_ASSIGN_OR_RETURN(auto function_ref_attrs,
                      ConvertSubgraphIdxsToFunctionAttrs(op.builtin_options,
                                                         func_names, builder));
  op_state.addAttributes(function_ref_attrs);

  return builder.create(op_state);
}

// Returns indices of the given tensors in the subgraph. Returns error if a
// tensor name cannot be found in the subgraph.
StatusOr<std::vector<int>> GetTensorIndices(
    const tflite::SubGraphT& subgraph,
    const std::vector<std::string>& tensor_names) {
  absl::flat_hash_map<std::string, int> name_to_index;
  for (const auto& index_and_tensor : llvm::enumerate(subgraph.tensors)) {
    name_to_index[index_and_tensor.value()->name] = index_and_tensor.index();
  }

  std::vector<int> indices;
  indices.reserve(tensor_names.size());

  for (const auto& name : tensor_names) {
    auto found = name_to_index.find(name);
    if (found != name_to_index.end()) {
      indices.push_back(found->second);
    } else {
      return errors::InvalidArgument("could not find tensor in subgraph: ",
                                     name);
    }
  }

  return indices;
}

// Given a list of tensor indices, returns a string of concatenated tensor names
// wrapped in a NamedAttribute.
template <typename ContainerType>
mlir::NamedAttribute BuildTFEntryFunctionAttribute(
    const tflite::SubGraphT& subgraph, Builder* builder, const std::string name,
    const ContainerType indices) {
  auto tensor_names = llvm::map_range(
      indices, [&](int i) { return subgraph.tensors.at(i)->name; });
  return builder->getNamedAttr(
      name, builder->getStringAttr(llvm::join(tensor_names, ",")));
}

// Traverses the subgraph from output_indices to input_indices and returns the
// set of ops that are visited.
StatusOr<absl::flat_hash_set<const tflite::OperatorT*>> PruneSubgraph(
    const tflite::SubGraphT& subgraph, ArrayRef<int32_t> input_indices,
    ArrayRef<int32_t> output_indices) {
  // Create a map from tensor index to defining op.
  absl::flat_hash_map<int32_t, const tflite::OperatorT*> defining_op;
  for (const auto& op : subgraph.operators) {
    for (int32_t output : op->outputs) {
      if (!llvm::is_contained(input_indices, output)) {
        defining_op[output] = op.get();
      }
    }
  }

  std::vector<const tflite::OperatorT*> queue;
  for (int32_t output : output_indices) {
    if (auto& op = defining_op[output]) {
      queue.push_back(op);
    }
  }

  // Traverse the graph towards inputs.
  absl::flat_hash_set<const tflite::OperatorT*> visited;
  while (!queue.empty()) {
    const tflite::OperatorT* op = queue.back();
    queue.pop_back();
    if (!visited.insert(op).second) {
      // The node has already been visited.
      continue;
    }

    for (int32_t input : op->inputs) {
      // Input tensor may not have a defining op in case it is a subgraph input
      // or a constant tensor.
      if (auto& op = defining_op[input]) {
        queue.push_back(op);
      }
    }
  }

  return visited;
}

// We want to adjust the func op according to some cross ops information.
static StatusOr<FuncOp> PostProcessFuncOp(FuncOp func) {
  OpBuilder builder(func);
  // When a quantized constant is imported, its quantization parameter is set
  // to be narrow range. Here revert to be the fully range if the user doesn't
  // require narrow range.
  func.walk([&](tfl::QConstOp cst) {
    Value value = cst.getResult();
    Value full_range_const = value;
    auto qtype = mlir::quant::UniformQuantizedType::getQuantizedElementType(
        value.getType());
    // Only the 8-bit constants are imported with narrow range.
    if (!qtype || qtype.getStorageTypeIntegralWidth() != 8 ||
        !(qtype.isa<mlir::quant::UniformQuantizedType>() ||
          qtype.isa<mlir::quant::UniformQuantizedPerAxisType>())) {
      return;
    }
    for (auto& use : value.getUses()) {
      Operation* user = use.getOwner();
      if (user->hasTrait<mlir::OpTrait::IsTerminator>()) continue;

      auto affine_user = llvm::dyn_cast<mlir::AffineQuantizedOpInterface>(user);
      if (affine_user &&
          affine_user.GetAffineOperandIndex() == use.getOperandNumber() &&
          affine_user.RequiredNarrowRangeAffineOperand())
        continue;
      // Create a fully range quantized constant.
      if (full_range_const == value) {
        mlir::quant::QuantizedType new_qtype;
        if (auto per_axis =
                qtype.dyn_cast<mlir::quant::UniformQuantizedPerAxisType>()) {
          new_qtype = mlir::quant::UniformQuantizedPerAxisType::get(
              per_axis.getFlags(), per_axis.getStorageType(),
              per_axis.getExpressedType(), per_axis.getScales(),
              per_axis.getZeroPoints(), per_axis.getQuantizedDimension(),
              per_axis.getStorageTypeMin() - 1, per_axis.getStorageTypeMax());
        } else if (auto per_tensor =
                       qtype.dyn_cast<mlir::quant::UniformQuantizedType>()) {
          new_qtype = mlir::quant::UniformQuantizedType::get(
              per_tensor.getFlags(), per_tensor.getStorageType(),
              per_tensor.getExpressedType(), per_tensor.getScale(),
              per_tensor.getZeroPoint(), per_tensor.getStorageTypeMin() - 1,
              per_tensor.getStorageTypeMax());
        } else {
          return;  // Should not reach here, as it's already checked.
        }
        auto new_output_type = new_qtype.castFromExpressedType(
            mlir::quant::UniformQuantizedType::castToExpressedType(
                value.getType()));
        builder.setInsertionPointAfter(cst.getOperation());
        auto new_op = builder.create<tfl::QConstOp>(
            cst.getLoc(), new_output_type, mlir::TypeAttr::get(new_output_type),
            cst.valueAttr());
        full_range_const = new_op.output();
      }
      use.set(full_range_const);
    }
    if (cst.use_empty()) cst.erase();
  });
  return func;
}

// Helper method that returns the index of the tensor with name 'tensor_name'
// in the list of tensor names 'tensors'. It allows excluding some indices.
int GetTensorIndex(const std::string& tensor_name,
                   llvm::SmallVector<llvm::StringRef, 2> tensors,
                   const std::set<int>& exclude_indices = {}) {
  for (const auto& tensor_index_pair : llvm::enumerate(tensors)) {
    if (tensor_index_pair.value() == tensor_name &&
        exclude_indices.find(tensor_index_pair.index()) ==
            exclude_indices.end())
      return tensor_index_pair.index();
  }
  return -1;
}

// Helper method that returns list of all strings in a StringAttr identified
// by 'attr_key' and values are separated by a comma.
llvm::SmallVector<llvm::StringRef, 2> GetStringsFromAttrWithSeparator(
    mlir::DictionaryAttr attr, const std::string& attr_key) {
  llvm::SmallVector<llvm::StringRef, 2> result;
  if (auto str = attr.get(attr_key).dyn_cast_or_null<mlir::StringAttr>()) {
    str.getValue().split(result, ',', /*MaxSplit=*/-1,
                         /*KeepEmpty=*/false);
  }
  return result;
}

// Sets signature attributes on the function.
void SetSignature(
    FuncOp func, const tflite::SignatureDefT* signature,
    const std::vector<std::unique_ptr<tflite::TensorT>>& tensors) {
  auto* context = func->getContext();
  static const char kSignatureDefIndexPath[] = "tf_saved_model.index_path";
  static const char kExportedNameAttr[] = "tf_saved_model.exported_names";
  static const char kEntryFunctionAttributes[] = "tf.entry_function";

  auto dict_attr =
      func->getAttrOfType<mlir::DictionaryAttr>(kEntryFunctionAttributes);
  if (!dict_attr) return;

  // Get Input and output tensor names from attribute.
  llvm::SmallVector<llvm::StringRef, 2> input_names =
      GetStringsFromAttrWithSeparator(dict_attr, /*attr_key=*/"inputs");
  llvm::SmallVector<llvm::StringRef, 2> output_names =
      GetStringsFromAttrWithSeparator(dict_attr, /*attr_key=*/"outputs");

  for (const auto& input_pair : llvm::enumerate(signature->inputs)) {
    const int arg_index = GetTensorIndex(
        tensors[input_pair.value()->tensor_index]->name, input_names);
    if (arg_index == -1) {
      func->emitWarning("Invalid signature tensors specified.");
      return;
    }
    func.setArgAttr(
        arg_index, kSignatureDefIndexPath,
        mlir::ArrayAttr::get(context, {mlir::StringAttr::get(
                                          context, input_pair.value()->name)}));
  }
  // Multiple signature outputs can refer to the same tensor. Avoid setting
  // signature output attribute at the same index by maintaining a set.
  std::set<int> seen_indices;
  for (const auto& output_pair : llvm::enumerate(signature->outputs)) {
    const int arg_index =
        GetTensorIndex(tensors[output_pair.value()->tensor_index]->name,
                       output_names, seen_indices);
    if (arg_index == -1) {
      func->emitWarning("Invalid signature tensors specified.");
      return;
    }
    func.setResultAttr(arg_index, kSignatureDefIndexPath,
                       mlir::ArrayAttr::get(
                           context, {mlir::StringAttr::get(
                                        context, output_pair.value()->name)}));
    seen_indices.insert(arg_index);
  }
  func->setAttr(
      kExportedNameAttr,
      mlir::ArrayAttr::get(
          context, {mlir::StringAttr::get(context, signature->signature_key)}));
}

// Build a FuncOp from a tflite SubGraph
// The buffers are directly taken
// from the deserialized flatbuffer as we do not have the type information to
// interpret them until this point. The base_loc parameter is the location of
// the flatbuffer as a whole (usually a file). The is_entry_point flag
// controls whether shapeless types are treated as scalars. If
// ordered_output_arrays is not empty, then the imported mlir function will only
// return nodes in ordered_output_arrays in the same order.
// If signature is not null, then the inputs/outputs in signature will be
// attached to the FuncOp.
StatusOr<FuncOp> ConvertSubgraph(
    const tflite::SubGraphT& subgraph, llvm::StringRef name,
    const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& op_codes,
    const std::vector<std::string>& func_names,
    const std::vector<std::unique_ptr<tflite::BufferT>>& buffers,
    Location base_loc, Builder builder, bool is_entry_point,
    bool use_external_constant,
    const std::vector<std::string>& ordered_input_arrays,
    const std::vector<std::string>& ordered_output_arrays,
    bool experimental_prune_unreachable_nodes_unconditionally,
    const tflite::SignatureDefT* signature) {
  llvm::SmallVector<mlir::Type, 2> ret_types;
  llvm::SmallVector<mlir::Type, 4> input_types;

  auto func_loc = mlir::NameLoc::get(builder.getStringAttr(name), base_loc);

  std::vector<int> func_inputs = subgraph.inputs;
  if (is_entry_point && !ordered_input_arrays.empty()) {
    if (!experimental_prune_unreachable_nodes_unconditionally) {
      // TODO(b/149922113): Resolve input-arrays/pruning flags interaction.
      return errors::InvalidArgument(
          "input-arrays should be used with experimental pruning flag");
    }
    TF_ASSIGN_OR_RETURN(func_inputs,
                        GetTensorIndices(subgraph, ordered_input_arrays));
  }

  for (int input : func_inputs) {
    auto& tensor = *subgraph.tensors.at(input);
    // TODO(b/138222071) Graph inputs must have static shape per the exporter,
    // but we cannot differentiate scalars from unranked tensors.
    // Here we reverse the default assumption that shape = [] means unranked.
    // when processing main()
    auto type_or_err = GetTensorType(tensor, builder,
                                     /*shapeless_are_scalars=*/is_entry_point,
                                     /*is_constant=*/false);
    if (!type_or_err.ok()) {
      emitError(func_loc, "error reading argument types")
          << type_or_err.status().ToString();
      return type_or_err.status();
    }
    auto type = type_or_err.ConsumeValueOrDie();
    input_types.push_back(type);
  }

  llvm::SmallVector<bool, 16> is_op_output(subgraph.tensors.size(), false);
  for (auto& op : subgraph.operators) {
    for (auto output : op->outputs) {
      is_op_output[output] = true;
    }
  }

  std::vector<int> func_outputs = subgraph.outputs;
  if (is_entry_point && !ordered_output_arrays.empty()) {
    TF_ASSIGN_OR_RETURN(func_outputs,
                        GetTensorIndices(subgraph, ordered_output_arrays));
  }

  for (auto output : func_outputs) {
    const bool is_func_input = std::find(func_inputs.begin(), func_inputs.end(),
                                         output) != func_inputs.end();
    bool is_constant = !is_op_output[output] && !is_func_input;
    // There are 2 cases tensor is scalar when it doesn't have a shape in
    // flatbuffer:
    // 1. `is_constant` = true, means this tensor is created from a constant op.
    // 2. `is_func_input` = true and `is_entry_point` = true, which means this
    // tensor is function input and function input type is a scalar tensor.
    const bool shapeless_is_scalar =
        is_constant || (is_func_input && is_entry_point);
    auto type_or_err = GetTensorType(*subgraph.tensors.at(output), builder,
                                     shapeless_is_scalar,
                                     /*is_constant=*/is_constant);
    if (!type_or_err.ok()) {
      emitError(func_loc, "error reading return types")
          << type_or_err.status().ToString();
      return type_or_err.status();
    }
    auto type = type_or_err.ConsumeValueOrDie();
    ret_types.push_back(type);
  }
  auto func_type = builder.getFunctionType(input_types, ret_types);

  // Construct function object
  auto func = FuncOp::create(func_loc, name, func_type, /* attrs= */ {});
  func.addEntryBlock();
  auto& body = func.getBody();
  OpBuilder op_builder{body};

  std::vector<Value> vals_map(subgraph.tensors.size(), nullptr);
  Value maybe_optional_arg_marker = nullptr;

  // Get or construct MLIR values for each input
  for (int i = 0, e = func_inputs.size(); i < e; i++) {
    auto input_tensor = func_inputs[i];
    const auto& tensor = *subgraph.tensors.at(input_tensor);
    auto loc = TensorLoc(tensor, builder, base_loc);
    if (vals_map[input_tensor]) {
      auto err = errors::FailedPrecondition("duplicate input arguments");
      return emitError(loc, err.ToString()), err;
    }
    Value input_value = func.getArgument(i);

    // If the `tensor` has min/max and doesn't have scale/zero_point
    // information, a stats op is created to use the input_value, then the
    // `tensor` should be mapped to the result of this new stats op.
    if (auto stats_op =
            ConvertMinMaxToStatsOp(tensor, op_builder, input_value)) {
      vals_map[input_tensor] = stats_op->getResult(0);
    } else {
      vals_map[input_tensor] = input_value;
    }
  }

  // Set tf.entry_function attribute
  if (is_entry_point) {
    llvm::SmallVector<mlir::NamedAttribute, 2> attributes;
    if (!func_inputs.empty()) {
      attributes.push_back(BuildTFEntryFunctionAttribute(
          subgraph, &builder, "inputs", func_inputs));
    }
    if (!func_outputs.empty()) {
      attributes.push_back(BuildTFEntryFunctionAttribute(
          subgraph, &builder, "outputs", func_outputs));
    }
    func->setAttr("tf.entry_function", builder.getDictionaryAttr(attributes));
  } else {
    func.setPrivate();
  }

  // Set signature on function.
  if (signature) {
    SetSignature(func, signature, subgraph.tensors);
  }

  absl::flat_hash_set<const tflite::OperatorT*> pruned_subgraph_ops;
  if (experimental_prune_unreachable_nodes_unconditionally) {
    TF_ASSIGN_OR_RETURN(pruned_subgraph_ops,
                        PruneSubgraph(subgraph, func_inputs, func_outputs));
  }

  // Construct MLIR operators from TFLite operators
  for (auto& op : subgraph.operators) {
    if (experimental_prune_unreachable_nodes_unconditionally &&
        !pruned_subgraph_ops.contains(op)) {
      continue;
    }

    for (auto input_num : op->inputs) {
      // The operators in a graph are topologically sorted
      // and so if no previous operation has produced a tensor
      // it must be a constant.
      if (input_num == -1) {
        if (maybe_optional_arg_marker == nullptr) {
          maybe_optional_arg_marker =
              op_builder
                  .create<mlir::TFL::NoValueOp>(base_loc, builder.getNoneType(),
                                                builder.getUnitAttr())
                  .getResult();
        }
      } else if (!vals_map.at(input_num)) {
        auto& const_tensor = *subgraph.tensors[input_num];
        auto const_loc = TensorLoc(const_tensor, builder, base_loc);
        auto op_or_err =
            use_external_constant
                ? BuildExternalConstOp(const_tensor, const_tensor.buffer,
                                       op_builder, const_loc)
                : BuildConstOp(const_tensor, buffers[const_tensor.buffer]->data,
                               const_tensor.is_variable, op_builder, const_loc);
        if (!op_or_err.ok()) {
          return emitError(const_loc, op_or_err.status().ToString()),
                 op_or_err.status();
        }
        vals_map[input_num] = op_or_err.ValueOrDie()->getResult(0);
      }
    }

    // Intermediate tensors for LSTMs are used to carry quantization range
    // in their types, so we only need and extract their types.
    std::vector<mlir::TensorType> intermediate_types;
    intermediate_types.reserve(5);
    for (auto intermediate : op->intermediates) {
      TF_ASSIGN_OR_RETURN(
          auto type, GetTensorType(*subgraph.tensors[intermediate], builder,
                                   /*shapeless_are_scalars=*/true,
                                   /*is_constant=*/false,
                                   /*is_intermediate=*/true));
      intermediate_types.emplace_back(type);
    }

    auto op_loc = OpLoc(*op, subgraph.tensors, builder, base_loc);

    // If there's an optional argument, maybe_optional_arg_marker has been set
    // to a valid Value
    TF_ASSIGN_OR_RETURN(
        auto* mlir_op,
        ConvertOp(*op, vals_map, intermediate_types, maybe_optional_arg_marker,
                  op_codes, func_names, subgraph.tensors, op_loc, op_builder));

    // Add the results to the value maps. There are two cases: 1. the result
    // tensor does not have min/max values, the original op result is used
    // directly; 2. the result tensor has some min/max values, a stats op is
    // created, then the result of the stats op is used.
    for (const auto& pair : llvm::enumerate(mlir_op->getResults())) {
      int output_tensor_index = op->outputs[pair.index()];
      auto& tensor = *subgraph.tensors[output_tensor_index];
      if (auto stats_op =
              ConvertMinMaxToStatsOp(tensor, op_builder, pair.value())) {
        vals_map[output_tensor_index] = stats_op->getResult(0);
      } else {
        vals_map[output_tensor_index] = pair.value();
      }
    }
  }

  // Construct return values
  llvm::SmallVector<Value, 4> return_operands;
  for (auto index : func_outputs) {
    if (!vals_map.at(index)) {
      auto& const_tensor = *subgraph.tensors[index];
      auto const_loc = TensorLoc(const_tensor, builder, base_loc);
      auto op_or_err =
          use_external_constant
              ? BuildExternalConstOp(const_tensor, const_tensor.buffer,
                                     op_builder, const_loc)
              : BuildConstOp(const_tensor, buffers[const_tensor.buffer]->data,
                             const_tensor.is_variable, op_builder, const_loc);
      if (!op_or_err.ok()) {
        return emitError(const_loc, op_or_err.status().ToString()),
               op_or_err.status();
      }
      vals_map[index] = op_or_err.ValueOrDie()->getResult(0);
    }
    return_operands.push_back(vals_map[index]);
  }

  op_builder.create<mlir::func::ReturnOp>(base_loc, return_operands);

  return PostProcessFuncOp(func);
}

// TFLite subgraphs do not necessarily have names, though MLIR functions must
// have them, so we generate a name for subgraphs that are missing one here.
// Note: in TFLite, the first subgraph is the entry point, and in MLIR that
// represents TFLite, this entry point must be called "main"
std::string SubgraphName(bool set_implicit_main_func, unsigned index,
                         const tflite::SubGraphT& subgraph) {
  if (index == 0 && set_implicit_main_func) {
    return "main";
  }
  if (subgraph.name.empty()) {
    return llvm::formatv("fn_{0}", index).str();
  }
  return subgraph.name;
}

// Adds a CallOp in `region` to call the `func` and returns the results of
// CallOp.
void AddCallOpInWhileOpRegion(mlir::Region& region, mlir::func::FuncOp func) {
  OpBuilder op_builder{region};
  region.push_back(new mlir::Block());
  Location loc = region.getLoc();
  auto inputs = func.getFunctionType().getInputs();
  region.addArguments(inputs, mlir::SmallVector<Location>(inputs.size(), loc));
  op_builder.setInsertionPointToStart(&region.front());
  auto call_op = op_builder.create<mlir::func::CallOp>(
      loc, func.getFunctionType().getResults(), func.getSymName(),
      region.getArguments());
  op_builder.create<mlir::TFL::YieldOp>(loc, call_op.getResults());
}

// TFL::WhileOp has regions, so we add CallOp to call the FuncOp in the regions
// if we have while ops.
void AddRegionsForTflWhileOp(mlir::ModuleOp module) {
  mlir::SymbolTable symbol_table(module);
  module.walk([&](mlir::TFL::WhileOp while_op) {
    auto cond = symbol_table.lookup<mlir::func::FuncOp>(
        while_op->getAttr("cond").cast<mlir::FlatSymbolRefAttr>().getValue());
    AddCallOpInWhileOpRegion(while_op.cond(), cond);
    while_op->removeAttr("cond");
    auto body = symbol_table.lookup<mlir::func::FuncOp>(
        while_op->getAttr("body").cast<mlir::FlatSymbolRefAttr>().getValue());
    AddCallOpInWhileOpRegion(while_op.body(), body);
    while_op->removeAttr("body");
  });
}
}  // namespace

OwningOpRef<mlir::ModuleOp> tflite::FlatBufferToMlir(
    absl::string_view buffer, MLIRContext* context, Location base_loc,
    bool use_external_constant,
    const std::vector<std::string>& ordered_input_arrays,
    const std::vector<std::string>& ordered_output_arrays,
    bool experimental_prune_unreachable_nodes_unconditionally) {
  context->loadDialect<mlir::arith::ArithmeticDialect, mlir::func::FuncDialect,
                       mlir::quant::QuantizationDialect,
                       mlir::TFL::TensorFlowLiteDialect,
                       mlir::TF::TensorFlowDialect>();

  auto model_ptr =
      FlatBufferModel::VerifyAndBuildFromBuffer(buffer.data(), buffer.length());
  if (nullptr == model_ptr) {
    return emitError(base_loc, "couldn't parse flatbuffer"), nullptr;
  }

  std::unique_ptr<ModelT> model(model_ptr->GetModel()->UnPack());

  auto builder = Builder(context);

  std::vector<std::string> func_names;
  for (auto& subgraph : model->subgraphs) {
    func_names.push_back(subgraph->name);
  }

  auto module = mlir::ModuleOp::create(base_loc);
  // We currently don't use this to make decisions, but we could
  // use it in exports or if there are breaking changes
  module->setAttr("tfl.schema_version",
                  builder.getI32IntegerAttr(model->version));
  if (!model->description.empty()) {
    module->setAttr("tfl.description",
                    builder.getStringAttr(model->description));
  }

  if (!model->signature_defs.empty()) {
    module->setAttr("tf_saved_model.semantics",
                    mlir::UnitAttr::get(builder.getContext()));
  }

  absl::flat_hash_map<uint32_t, tflite::SignatureDefT*>
      subgraph_to_signature_map;
  for (int i = 0; i < model->signature_defs.size(); i++) {
    auto* signature_def = model->signature_defs[i].get();
    const uint32_t subgraph_index = signature_def->subgraph_index;
    subgraph_to_signature_map[subgraph_index] = signature_def;
  }

  const bool set_implicit_main_func = subgraph_to_signature_map.size() <= 1;
  for (const auto& e : llvm::enumerate(model->subgraphs)) {
    auto& subgraph = e.value();
    std::string name =
        SubgraphName(set_implicit_main_func, e.index(), *subgraph);
    uint32_t subgraph_index = static_cast<uint32_t>(e.index());
    auto func_or_error = ConvertSubgraph(
        *subgraph, name, model->operator_codes, func_names, model->buffers,
        base_loc, builder,
        /*is_entry_point=*/
        set_implicit_main_func
            ? e.index() == 0
            : subgraph_to_signature_map.contains(subgraph_index),
        /*use_external_constant=*/use_external_constant, ordered_input_arrays,
        ordered_output_arrays,
        experimental_prune_unreachable_nodes_unconditionally,
        subgraph_to_signature_map.contains(subgraph_index)
            ? subgraph_to_signature_map.at(subgraph_index)
            : nullptr);
    if (!func_or_error.ok()) {
      return emitError(base_loc, "could not translate function ")
                 << subgraph->name << ": "
                 << func_or_error.status().error_message(),
             nullptr;
    }
    module.push_back(func_or_error.ConsumeValueOrDie());
  }
  AddRegionsForTflWhileOp(module);
  return OwningOpRef<mlir::ModuleOp>(module);
}
