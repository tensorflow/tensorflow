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

#include "tensorflow/compiler/mlir/lite/flatbuffer_operator.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "flatbuffers/string.h"  // from @flatbuffers
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/AttrTypeSubElements.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/dynamic_shape_utils.h"
#include "xla/statusor.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/kernel_utils.h"
#include "tensorflow/lite/schema/mutable/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"
#include "tsl/platform/status.h"

namespace {

using ::absl::StatusOr;
using ::tensorflow::Status;
using ::tensorflow::errors::InvalidArgument;

StatusOr<mlir::StringAttr> GetPaddingAttr(TfLitePadding pad_params,
                                          mlir::Builder builder,
                                          mlir::Location loc) {
  auto padding = tflite::Padding::Padding_VALID;
  if (pad_params == TfLitePadding::kTfLitePaddingSame) {
    padding = tflite::Padding_SAME;
  } else if (pad_params == TfLitePadding::kTfLitePaddingValid) {
    padding = tflite::Padding_VALID;
  } else {
    return InvalidArgument(
        absl::StrCat("Invalid padding type", std::to_string(pad_params)));
  }

  const char* option_name = tflite::EnumNamePadding(padding);
  return builder.getStringAttr(option_name);
}

}  // namespace

bool mlir::IsStablehloOp(const tflite::OperatorCodeT& op_code) {
  llvm::StringRef op_name(
      tflite::EnumNameBuiltinOperator(tflite::GetBuiltinCode(&op_code)));
  return op_name.starts_with("STABLEHLO_");
}

std::string mlir::GetMlirOpNameFromOpCode(
    const tflite::OperatorCodeT& op_code) {
  auto builtin_code = tflite::GetBuiltinCode(&op_code);
  if (builtin_code == tflite::BuiltinOperator_IF) {
    return std::string("tf.If");
  }

  llvm::StringRef op_name(tflite::EnumNameBuiltinOperator(builtin_code));

  // If the Op name contains stablehlo
  if (IsStablehloOp(op_code)) {
    return llvm::Twine(
               llvm::Twine("vhlo.", op_name.drop_front(10).lower()).str(),
               "_v1")
        .str();
  }
  return llvm::Twine("tfl.", op_name.lower()).str();
}

// TODO(jpienaar): This is a placeholder. This should be done in more efficient
// way when part of the translation of module.
static tflite::ActivationFunctionType ConvertTFL_AFAttrForOptionWriter(
    llvm::StringRef str, flatbuffers::FlatBufferBuilder* builder) {
  return llvm::StringSwitch<tflite::ActivationFunctionType>(str)
      .Case("NONE", tflite::ActivationFunctionType_NONE)
      .Case("RELU", tflite::ActivationFunctionType_RELU)
      .Case("RELU_N1_TO_1", tflite::ActivationFunctionType_RELU_N1_TO_1)
      .Case("RELU6", tflite::ActivationFunctionType_RELU6)
      .Case("TANH", tflite::ActivationFunctionType_TANH)
      .Case("SIGN_BIT", tflite::ActivationFunctionType_SIGN_BIT);
}

static tflite::TensorType ConvertDerivedTFLiteTypeAttrForOptionWriter(
    tflite::TensorType type, flatbuffers::FlatBufferBuilder* builder) {
  if (type == tflite::TensorType_INT64) {
    return tflite::TensorType_INT64;
  } else if (type == tflite::TensorType_INT32) {
    return tflite::TensorType_INT32;
  }
  llvm_unreachable("invalid type in conversion.");
}

static tflite::Padding ConvertTFL_PaddingAttrForOptionWriter(
    llvm::StringRef str, flatbuffers::FlatBufferBuilder* builder) {
  return llvm::StringSwitch<tflite::Padding>(str)
      .Case("SAME", tflite::Padding_SAME)
      .Case("VALID", tflite::Padding_VALID);
}

static tflite::MirrorPadMode ConvertTFL_MirrorPaddingAttrForOptionWriter(
    mlir::TFL::MirrorPaddingType padding,
    flatbuffers::FlatBufferBuilder* builder) {
  switch (padding) {
    case mlir::TFL::MirrorPaddingType::REFLECT:
      return tflite::MirrorPadMode_REFLECT;
    case mlir::TFL::MirrorPaddingType::SYMMETRIC:
      return tflite::MirrorPadMode_SYMMETRIC;
  }
  llvm_unreachable("invalid mirror_pad_enum in conversion.");
}

static tflite::TensorType ConvertDerivedTypeAttrForOptionWriter(
    mlir::Type type, flatbuffers::FlatBufferBuilder* builder) {
  return tflite::ConvertTypeToTensorType(type);
}

// I32Attr already returns an int as required by flatbuffer builders.
static int ConvertI32AttrForOptionWriter(
    int i, flatbuffers::FlatBufferBuilder* builder) {
  return i;
}

// I64Attr already returns a int64_t as required by flatbuffer builders.
static int64_t ConvertI64AttrForOptionWriter(
    int64_t i, flatbuffers::FlatBufferBuilder* builder) {
  return i;
}

static int ConvertPositiveI32AttrForOptionWriter(
    int i, flatbuffers::FlatBufferBuilder* builder) {
  return ConvertI32AttrForOptionWriter(i, builder);
}

static flatbuffers::Offset<flatbuffers::Vector<int32_t>>
ConvertI64ArrayAttrForOptionWriter(mlir::ArrayAttr attrArray,
                                   flatbuffers::FlatBufferBuilder* builder) {
  std::vector<int32_t> intVec;
  intVec.reserve(attrArray.getValue().size());
  for (auto attr : attrArray.getValue()) {
    intVec.push_back(mlir::cast<mlir::IntegerAttr>(attr).getInt());
  }
  return builder->CreateVector(intVec);
}

static flatbuffers::Offset<flatbuffers::Vector<float>>
ConvertF32ArrayAttrForOptionWriter(mlir::ArrayAttr attrArray,
                                   flatbuffers::FlatBufferBuilder* builder) {
  std::vector<float> floatVec;
  floatVec.reserve(attrArray.getValue().size());
  for (auto attr : attrArray.getValue()) {
    floatVec.push_back(
        mlir::cast<mlir::FloatAttr>(attr).getValue().convertToFloat());
  }
  return builder->CreateVector(floatVec);
}

// F32Attr already returns a float as required by flatbuffer builders.
static float ConvertF32AttrForOptionWriter(
    llvm::APFloat f, flatbuffers::FlatBufferBuilder* builder) {
  return f.convertToFloat();
}

// BoolAttr already returns a bool as required by flatbuffer builders.
static bool ConvertBoolAttrForOptionWriter(
    bool b, flatbuffers::FlatBufferBuilder* builder) {
  return b;
}

// Overloading of ConvertBoolAttrForOptionWriter which takes std::optional<bool>
// as an input. If value is not specified, false is set for the attribute.
static bool ConvertBoolAttrForOptionWriter(
    std::optional<bool> b, flatbuffers::FlatBufferBuilder* builder) {
  return b.has_value() ? b.value() : false;
}

static flatbuffers::Offset<flatbuffers::String> ConvertStrAttrForOptionWriter(
    llvm::StringRef str, flatbuffers::FlatBufferBuilder* builder) {
  return builder->CreateString(str.str());
}

static tflite::TensorType ConvertTypeAttrForOptionWriter(
    mlir::Type type, flatbuffers::FlatBufferBuilder* builder) {
  return tflite::ConvertTypeToTensorType(type);
}

static flatbuffers::Offset<flatbuffers::Vector<int32_t>>
ConvertDerivedShapeAttrForOptionWriter(
    llvm::ArrayRef<int64_t> r, flatbuffers::FlatBufferBuilder* builder) {
  std::vector<int> intVec(r.begin(), r.end());
  return builder->CreateVector(intVec);
}

static tflite::FullyConnectedOptionsWeightsFormat
ConvertTFL_FullyConnectedOptionsWeightFormatAttrForOptionWriter(
    llvm::StringRef str, flatbuffers::FlatBufferBuilder* builder) {
  return llvm::StringSwitch<tflite::FullyConnectedOptionsWeightsFormat>(str)
      .Case("DEFAULT", tflite::FullyConnectedOptionsWeightsFormat_DEFAULT)
      .Case("SHUFFLED4x16INT8",
            tflite::FullyConnectedOptionsWeightsFormat_SHUFFLED4x16INT8);
}

static tflite::LSTMKernelType ConvertTFL_LSTMKernelTypeAttrForOptionWriter(
    mlir::TFL::LSTMKernelType kernel_type,
    flatbuffers::FlatBufferBuilder* builder) {
  switch (kernel_type) {
    case mlir::TFL::LSTMKernelType::FULL:
      return tflite::LSTMKernelType_FULL;
    case mlir::TFL::LSTMKernelType::BASIC:
      return tflite::LSTMKernelType_BASIC;
  }
  llvm_unreachable("invalid lstm_kernel_type in conversion.");
}

static mlir::Attribute BuildBoolAttr(bool value, mlir::Builder builder) {
  return builder.getBoolAttr(value);
}

static mlir::Attribute BuildStrAttr(llvm::StringRef str,
                                    mlir::Builder builder) {
  return builder.getStringAttr(str);
}

static mlir::Attribute BuildF32Attr(float value, mlir::Builder builder) {
  return builder.getF32FloatAttr(value);
}

static mlir::Attribute BuildI32Attr(int32_t value, mlir::Builder builder) {
  return builder.getI32IntegerAttr(value);
}

static mlir::Attribute BuildI64Attr(int64_t value, mlir::Builder builder) {
  return builder.getI64IntegerAttr(value);
}

static mlir::Attribute BuildI64ArrayAttr(std::vector<int32_t> value,
                                         mlir::Builder builder) {
  std::vector<int64_t> typecast(value.begin(), value.end());
  return builder.getI64ArrayAttr(typecast);
}

static mlir::Attribute BuildVhloBooleanV1Attr(bool value,
                                              mlir::Builder builder) {
  return mlir::vhlo::BooleanV1Attr::get(builder.getContext(), value);
}

static mlir::Attribute BuildVhloIntV1Attr(int64_t value,
                                          mlir::Builder builder) {
  mlir::StablehloVhloTypeConverter type_converter;
  auto vhlo_type =
      type_converter.convertType(builder.getI64IntegerAttr(value).getType());
  return mlir::vhlo::IntegerV1Attr::get(builder.getContext(), vhlo_type,
                                        llvm::APInt(64, value));
}

static mlir::Attribute BuildVhloStringV1Attr(llvm::StringRef str,
                                             mlir::Builder builder) {
  return mlir::vhlo::StringV1Attr::get(builder.getContext(), str);
}

static mlir::Attribute BuildVhloArrayV1Attr(std::vector<mlir::Attribute> value,
                                            mlir::Builder builder) {
  return mlir::vhlo::ArrayV1Attr::get(builder.getContext(), value);
}

static mlir::Attribute BuildVhloDictionaryV1Attr(
    std::vector<std::pair<mlir::Attribute, mlir::Attribute>> value,
    mlir::Builder builder) {
  return mlir::vhlo::DictionaryV1Attr::get(builder.getContext(), value);
}

static mlir::Attribute BuildVhloFloatV1Attr(::llvm::APFloat value,
                                            mlir::Type type,
                                            mlir::Builder builder) {
  return mlir::vhlo::FloatV1Attr::get(builder.getContext(), type,
                                      std::move(value));
}

static mlir::Attribute BuildRankedTensorAttr(std::vector<int64_t> shape,
                                             std::vector<bool> value,
                                             mlir::Builder builder) {
  // The implementation of getBoolVectorAttr is flawed, so we bypass it here
  std::vector<llvm::APInt> extendVec;
  extendVec.resize(value.size());
  for (size_t i = 0; i < value.size(); ++i) {
    extendVec[i] = llvm::APInt(1, value[i]);
  }
  mlir::RankedTensorType ty =
      tensorflow::GetTypeFromTFTensorShape(shape, builder.getIntegerType(1));
  return mlir::DenseIntElementsAttr::get(ty, extendVec);
}

static mlir::Attribute BuildRankedTensorAttr(std::vector<int64_t> shape,
                                             std::vector<int64_t> value,
                                             mlir::Builder builder) {
  mlir::RankedTensorType ty =
      tensorflow::GetTypeFromTFTensorShape(shape, builder.getIntegerType(64));
  return mlir::DenseIntElementsAttr::get(ty, value);
}

static mlir::Attribute BuildVhloTensorV1Attr(std::vector<int64_t> shape,
                                             std::vector<int64_t> value,
                                             mlir::Builder builder) {
  mlir::StablehloVhloTypeConverter type_converter;
  auto builtin_attr = mlir::dyn_cast<mlir::DenseIntElementsAttr>(
      BuildRankedTensorAttr(shape, value, builder));
  auto vhlo_type = type_converter.convertType(builtin_attr.getType());
  return mlir::vhlo::TensorV1Attr::get(builder.getContext(), vhlo_type,
                                       builtin_attr.getRawData());
}

static mlir::Attribute BuildVhloTensorV1Attr(std::vector<int64_t> shape,
                                             std::vector<bool> value,
                                             mlir::Builder builder) {
  mlir::StablehloVhloTypeConverter type_converter;
  auto builtin_attr = mlir::dyn_cast<mlir::DenseIntElementsAttr>(
      BuildRankedTensorAttr(shape, value, builder));
  auto vhlo_type = type_converter.convertType(builtin_attr.getType());
  return mlir::vhlo::TensorV1Attr::get(builder.getContext(), vhlo_type,
                                       builtin_attr.getRawData());
}

static mlir::Attribute BuildVhloPrecisionConfigV1Attr(
    std::vector<tflite::StablehloPrecisionConfig> value,
    mlir::Builder builder) {
  llvm::SmallVector<mlir::Attribute> precision_attrs;
  for (size_t i = 0; i < value.size(); ++i) {
    precision_attrs.push_back(mlir::vhlo::PrecisionV1Attr::get(
        builder.getContext(),
        mlir::vhlo::symbolizePrecisionV1(static_cast<uint32_t>(value[i]))
            .value()));
  }
  return mlir::vhlo::ArrayV1Attr::get(builder.getContext(), precision_attrs);
}

static mlir::Attribute BuildF32ArrayAttr(std::vector<float> value,
                                         mlir::Builder builder) {
  std::vector<float> typecast(value.begin(), value.end());
  return builder.getF32ArrayAttr(typecast);
}

static mlir::Attribute BuildPositiveI32Attr(int32_t value,
                                            mlir::Builder builder) {
  return builder.getI32IntegerAttr(value);
}

static mlir::Attribute BuildTypeAttr(tflite::TensorType value,
                                     mlir::Builder builder) {
  return mlir::TypeAttr::get(ConvertElementType(value, builder));
}

static mlir::Attribute BuildTFL_AFAttr(tflite::ActivationFunctionType value,
                                       mlir::Builder builder) {
  const char* option_name = tflite::EnumNameActivationFunctionType(value);
  return builder.getStringAttr(option_name);
}

static mlir::Attribute BuildTFL_FullyConnectedOptionsWeightFormatAttr(
    tflite::FullyConnectedOptionsWeightsFormat value, mlir::Builder builder) {
  const char* option_name =
      tflite::EnumNameFullyConnectedOptionsWeightsFormat(value);
  return builder.getStringAttr(option_name);
}

static mlir::Attribute BuildTFL_LSTMKernelTypeAttr(tflite::LSTMKernelType value,
                                                   mlir::Builder builder) {
  mlir::TFL::LSTMKernelType kernel_type;
  switch (value) {
    case tflite::LSTMKernelType_FULL:
      kernel_type = mlir::TFL::LSTMKernelType::FULL;
      break;
    case tflite::LSTMKernelType_BASIC:
      kernel_type = mlir::TFL::LSTMKernelType::BASIC;
      break;
  }
  return mlir::TFL::LSTMKernelTypeAttr::get(builder.getContext(), kernel_type);
}

static mlir::Attribute BuildTFL_MirrorPaddingAttr(tflite::MirrorPadMode value,
                                                  mlir::Builder builder) {
  mlir::TFL::MirrorPaddingType padding;
  switch (value) {
    case tflite::MirrorPadMode_REFLECT:
      padding = mlir::TFL::MirrorPaddingType::REFLECT;
      break;
    case tflite::MirrorPadMode_SYMMETRIC:
    default:
      padding = mlir::TFL::MirrorPaddingType::SYMMETRIC;
      break;
  }
  return mlir::TFL::MirrorPaddingTypeAttr::get(builder.getContext(), padding);
}

static std::vector<mlir::Attribute> BuildAttributeVectorFromFlatbuffer(
    flexbuffers::Vector flatbuffer_vector, mlir::Builder builder) {
  std::vector<mlir::Attribute> mlir_vector;

  for (int i = 0; i < flatbuffer_vector.size(); ++i) {
    auto value = flatbuffer_vector[i];

    if (value.IsBool()) {
      mlir_vector.push_back(BuildVhloBooleanV1Attr(value.AsBool(), builder));
    } else if (value.IsString()) {
      mlir_vector.push_back(
          BuildVhloStringV1Attr(value.AsString().str(), builder));
    } else if (value.IsInt()) {
      mlir_vector.push_back(BuildVhloIntV1Attr(value.AsInt64(), builder));
    } else if (value.IsFloat()) {
      mlir_vector.push_back(BuildVhloFloatV1Attr(llvm::APFloat(value.AsFloat()),
                                                 mlir::Float32Type(), builder));
    } else if (value.IsVector()) {
      std::vector<mlir::Attribute> nested_mlir_vector =
          BuildAttributeVectorFromFlatbuffer(value.AsVector(), builder);
      mlir_vector.push_back(
          BuildVhloArrayV1Attr(std::move(nested_mlir_vector), builder));
    }
  }

  return mlir_vector;
}

static mlir::Attribute BuildTFL_PaddingAttr(tflite::Padding value,
                                            mlir::Builder builder) {
  const char* option_name = tflite::EnumNamePadding(value);
  return builder.getStringAttr(option_name);
}

static mlir::Attribute BuildStablehlo_PrecisionConfigAttr(
    std::vector<tflite::StablehloPrecisionConfig> value,
    mlir::Builder builder) {
  llvm::SmallVector<mlir::Attribute> precision_attrs;
  for (size_t i = 0; i < value.size(); ++i) {
    precision_attrs.push_back(mlir::stablehlo::PrecisionAttr::get(
        builder.getContext(),
        mlir::stablehlo::symbolizePrecision(static_cast<uint32_t>(value[i]))
            .value()));
  }
  return builder.getArrayAttr(precision_attrs);
}

Status mlir::CustomOptionsToAttributes(
    const std::string& custom_code, const std::vector<uint8_t>& custom_options,
    mlir::Builder builder, mlir::Location loc,
    llvm::SmallVectorImpl<mlir::NamedAttribute>* attributes) {
  attributes->emplace_back(
      builder.getNamedAttr("custom_code", builder.getStringAttr(custom_code)));
  std::string content;
  content.assign(reinterpret_cast<const char*>(custom_options.data()),
                 custom_options.size());
  attributes->emplace_back(builder.getNamedAttr(
      "custom_option",
      mlir::TFL::ConstBytesAttr::get(builder.getContext(), content)));

  return absl::OkStatus();
}

// TODO(zichuanwei@): Populate Builtin_options_2 manual for now, should
// automate these in the future
void BuiltinOptions2ToAttributesManual(
    tflite::BuiltinOptions2Union op_union, mlir::Builder builder,
    llvm::SmallVectorImpl<mlir::NamedAttribute>& attributes) {
  if (const auto* op = op_union.AsStablehloConcatenateOptions()) {
    attributes.emplace_back(builder.getNamedAttr(
        "dimension", BuildVhloIntV1Attr(op->dimension, builder)));
    return;
  }
  if (const auto* op = op_union.AsStablehloBroadcastInDimOptions()) {
    attributes.emplace_back(builder.getNamedAttr(
        "broadcast_dimensions",
        BuildVhloTensorV1Attr(
            {static_cast<int64_t>(op->broadcast_dimensions.size())},
            op->broadcast_dimensions, builder)));
    return;
  }
  if (const auto* op = op_union.AsStablehloSliceOptions()) {
    std::vector<int64_t> shape = {
        static_cast<int64_t>(op->start_indices.size())};
    attributes.emplace_back(builder.getNamedAttr(
        "start_indices",
        BuildVhloTensorV1Attr(shape, op->start_indices, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "limit_indices",
        BuildVhloTensorV1Attr(shape, op->limit_indices, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "strides", BuildVhloTensorV1Attr(shape, op->strides, builder)));
    return;
  }
  if (const auto* op = op_union.AsStablehloConvolutionOptions()) {
    if (!(op->window_strides.empty())) {
      std::vector<int64_t> shape;
      shape.push_back(static_cast<int64_t>(op->window_strides.size()));
      attributes.emplace_back(builder.getNamedAttr(
          "window_strides",
          BuildVhloTensorV1Attr(shape, op->window_strides, builder)));
    } else {
      std::vector<int64_t> data(op->input_spatial_dimensions.size(), 1);
      std::vector<int64_t> shape;
      shape.push_back(static_cast<int64_t>(data.size()));
      attributes.emplace_back(builder.getNamedAttr(
          "window_strides", BuildVhloTensorV1Attr(shape, data, builder)));
    }
    if (!(op->padding.empty())) {
      std::vector<int64_t> shape;
      shape.push_back(static_cast<int64_t>(op->padding.size()) / 2);
      shape.push_back(2);
      attributes.emplace_back(builder.getNamedAttr(
          "padding", BuildVhloTensorV1Attr(shape, op->padding, builder)));
    } else {
      std::vector<int64_t> data(op->input_spatial_dimensions.size() * 2, 0);
      std::vector<int64_t> shape;
      shape.push_back(static_cast<int64_t>(data.size()) / 2);
      shape.push_back(2);
      attributes.emplace_back(builder.getNamedAttr(
          "padding", BuildVhloTensorV1Attr(shape, data, builder)));
    }
    if (!(op->lhs_dilation.empty())) {
      std::vector<int64_t> shape;
      shape.push_back(static_cast<int64_t>(op->lhs_dilation.size()));
      attributes.emplace_back(builder.getNamedAttr(
          "lhs_dilation",
          BuildVhloTensorV1Attr(shape, op->lhs_dilation, builder)));
    } else {
      std::vector<int64_t> data(op->input_spatial_dimensions.size(), 1);
      std::vector<int64_t> shape;
      shape.push_back(static_cast<int64_t>(data.size()));
      attributes.emplace_back(builder.getNamedAttr(
          "lhs_dilation", BuildVhloTensorV1Attr(shape, data, builder)));
    }
    if (!(op->rhs_dilation.empty())) {
      std::vector<int64_t> shape;
      shape.push_back(static_cast<int64_t>(op->rhs_dilation.size()));
      attributes.emplace_back(builder.getNamedAttr(
          "rhs_dilation",
          BuildVhloTensorV1Attr(shape, op->rhs_dilation, builder)));
    } else {
      std::vector<int64_t> data(op->input_spatial_dimensions.size(), 1);
      std::vector<int64_t> shape;
      shape.push_back(static_cast<int64_t>(data.size()));
      attributes.emplace_back(builder.getNamedAttr(
          "rhs_dilation", BuildVhloTensorV1Attr(shape, data, builder)));
    }
    attributes.emplace_back(builder.getNamedAttr(
        "window_reversal",
        BuildVhloTensorV1Attr(
            {static_cast<int64_t>(op->window_reversal.size())},
            op->window_reversal, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "input_batch_dimension",
        BuildVhloIntV1Attr(op->input_batch_dimension, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "input_feature_dimension",
        BuildVhloIntV1Attr(op->input_feature_dimension, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "kernel_input_feature_dimension",
        BuildVhloIntV1Attr(op->kernel_input_feature_dimension, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "kernel_output_feature_dimension",
        BuildVhloIntV1Attr(op->kernel_output_feature_dimension, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "output_batch_dimension",
        BuildVhloIntV1Attr(op->output_batch_dimension, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "output_feature_dimension",
        BuildVhloIntV1Attr(op->output_feature_dimension, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "input_spatial_dimensions",
        BuildVhloTensorV1Attr(
            {static_cast<int64_t>(op->input_spatial_dimensions.size())},
            op->input_spatial_dimensions, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "kernel_spatial_dimensions",
        BuildVhloTensorV1Attr(
            {static_cast<int64_t>(op->kernel_spatial_dimensions.size())},
            op->kernel_spatial_dimensions, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "output_spatial_dimensions",
        BuildVhloTensorV1Attr(
            {static_cast<int64_t>(op->output_spatial_dimensions.size())},
            op->output_spatial_dimensions, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "feature_group_count",
        BuildVhloIntV1Attr(op->feature_group_count, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "batch_group_count",
        BuildVhloIntV1Attr(op->batch_group_count, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "precision_config",
        BuildVhloPrecisionConfigV1Attr(op->precision_config, builder)));

    return;
  }
  if (const auto* op = op_union.AsStablehloCustomCallOptions()) {
    // hard coding api version for now, we should rework this by updating the
    // STABLEHLO_CUSTOM_CALL definition
    attributes.emplace_back(builder.getNamedAttr(
        "api_version",
        mlir::vhlo::CustomCallApiVersionV1Attr::get(
            builder.getContext(),
            mlir::vhlo::symbolizeCustomCallApiVersionV1(op->api_version)
                .value())));
    attributes.emplace_back(builder.getNamedAttr(
        "call_target_name",
        BuildVhloStringV1Attr(op->call_target_name, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "backend_config", BuildVhloStringV1Attr(op->backend_config, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "called_computations", BuildVhloArrayV1Attr({}, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "operand_layouts", BuildVhloArrayV1Attr({}, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "output_operand_aliases", BuildVhloArrayV1Attr({}, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "result_layouts", BuildVhloArrayV1Attr({}, builder)));

    std::string has_side_effect_key = "has_side_effect";
    bool has_side_effect_set = false;
    const flexbuffers::Map& computation_map =
        flexbuffers::GetRoot(op->custom_attributes).AsMap();
    const auto& keys = computation_map.Keys();
    for (size_t i = 0; i < keys.size(); ++i) {
      const auto key = keys[i].AsKey();
      const auto& value = computation_map[key];
      if (has_side_effect_key == key) has_side_effect_set = true;
      if (value.IsBool()) {
        auto attr = value.AsBool();
        auto named_attr =
            builder.getNamedAttr(key, BuildVhloBooleanV1Attr(attr, builder));
        attributes.emplace_back(named_attr);
      }
      if (value.IsString()) {
        auto attr = value.AsString();
        auto named_attr = builder.getNamedAttr(
            key, BuildVhloStringV1Attr(attr.str(), builder));
        attributes.emplace_back(named_attr);
      }
    }
    if (!has_side_effect_set)
      attributes.emplace_back(builder.getNamedAttr(
          "has_side_effect", BuildVhloBooleanV1Attr(false, builder)));
    return;
  }
  if (const auto* op = op_union.AsStableHLOCompositeOptions()) {
    attributes.emplace_back(
        builder.getNamedAttr("name", BuildVhloStringV1Attr(op->name, builder)));

    attributes.emplace_back(builder.getNamedAttr(
        "version", BuildVhloIntV1Attr(op->version, builder)));

    auto composite_attribute_pairs =
        std::vector<std::pair<mlir::Attribute, mlir::Attribute>>();

    auto composite_attributes =
        flexbuffers::GetRoot(op->composite_attributes).AsMap();

    const auto& keys = composite_attributes.Keys();
    for (size_t i = 0; i < keys.size(); ++i) {
      const auto key = keys[i].AsKey();
      const auto& value = composite_attributes[key];

      std::pair<mlir::Attribute, mlir::Attribute> composite_attribute_pair;
      composite_attribute_pair.first = BuildVhloStringV1Attr(key, builder);

      if (value.IsBool()) {
        composite_attribute_pair.second =
            BuildVhloBooleanV1Attr(value.AsBool(), builder);
      }
      if (value.IsString()) {
        composite_attribute_pair.second =
            BuildVhloStringV1Attr(value.AsString().str(), builder);
      }
      if (value.IsInt()) {
        composite_attribute_pair.second =
            BuildVhloIntV1Attr(value.AsInt64(), builder);
      }
      if (value.IsFloat()) {
        composite_attribute_pair.second = BuildVhloFloatV1Attr(
            llvm::APFloat(value.AsFloat()), mlir::Float32Type(), builder);
      }

      if (value.IsVector()) {
        std::vector<mlir::Attribute> mlir_vector =
            BuildAttributeVectorFromFlatbuffer(value.AsVector(), builder);

        composite_attribute_pair.second =
            BuildVhloArrayV1Attr(std::move(mlir_vector), builder);
      }

      composite_attribute_pairs.emplace_back(composite_attribute_pair);
    }

    attributes.emplace_back(builder.getNamedAttr(
        "composite_attributes",
        BuildVhloDictionaryV1Attr(std::move(composite_attribute_pairs),
                                  builder)));
    return;
  }
  if (const auto* op = op_union.AsStablehloPadOptions()) {
    std::vector<int64_t> shape = {
        static_cast<int64_t>(op->edge_padding_low.size())};
    attributes.emplace_back(builder.getNamedAttr(
        "edge_padding_low",
        BuildVhloTensorV1Attr(shape, op->edge_padding_low, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "edge_padding_high",
        BuildVhloTensorV1Attr(shape, op->edge_padding_high, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "interior_padding",
        BuildVhloTensorV1Attr(shape, op->interior_padding, builder)));
    return;
  }
  if (const auto* op = op_union.AsStablehloDynamicSliceOptions()) {
    attributes.emplace_back(builder.getNamedAttr(
        "slice_sizes",
        BuildVhloTensorV1Attr({static_cast<int64_t>(op->slice_sizes.size())},
                              op->slice_sizes, builder)));
    return;
  }
  if (const auto* op = op_union.AsStablehloCompareOptions()) {
    attributes.emplace_back(builder.getNamedAttr(
        "comparison_direction",
        mlir::vhlo::ComparisonDirectionV1Attr::get(
            builder.getContext(),
            mlir::vhlo::symbolizeComparisonDirectionV1(op->comparison_direction)
                .value())));
    attributes.emplace_back(builder.getNamedAttr(
        "compare_type",
        mlir::vhlo::ComparisonTypeV1Attr::get(
            builder.getContext(),
            mlir::vhlo::symbolizeComparisonTypeV1(op->compare_type).value())));
    return;
  }
  if (const auto* op = op_union.AsStablehloIotaOptions()) {
    attributes.emplace_back(builder.getNamedAttr(
        "iota_dimension", BuildVhloIntV1Attr(op->iota_dimension, builder)));
    return;
  }
  if (const auto* op = op_union.AsStablehloReduceOptions()) {
    attributes.emplace_back(builder.getNamedAttr(
        "dimensions",
        BuildVhloTensorV1Attr({static_cast<int64_t>(op->dimensions.size())},
                              op->dimensions, builder)));
    return;
  }
  if (const auto* op = op_union.AsStablehloReduceWindowOptions()) {
    if (!op->window_dimensions.empty()) {
      attributes.emplace_back(builder.getNamedAttr(
          "window_dimensions",
          BuildVhloTensorV1Attr(
              {static_cast<int64_t>(op->window_dimensions.size())},
              op->window_dimensions, builder)));
    }
    if (!op->window_strides.empty()) {
      attributes.emplace_back(builder.getNamedAttr(
          "window_strides",
          BuildVhloTensorV1Attr(
              {static_cast<int64_t>(op->window_strides.size())},
              op->window_strides, builder)));
    }
    if (!op->base_dilations.empty()) {
      attributes.emplace_back(builder.getNamedAttr(
          "base_dilations",
          BuildVhloTensorV1Attr(
              {static_cast<int64_t>(op->base_dilations.size())},
              op->base_dilations, builder)));
    }
    if (!op->window_dilations.empty()) {
      attributes.emplace_back(builder.getNamedAttr(
          "window_dilations",
          BuildVhloTensorV1Attr(
              {static_cast<int64_t>(op->window_dilations.size())},
              op->window_dilations, builder)));
    }
    if (!op->padding.empty()) {
      attributes.emplace_back(builder.getNamedAttr(
          "padding", BuildVhloTensorV1Attr(
                         {static_cast<int64_t>(op->padding.size()) / 2, 2},
                         op->padding, builder)));
    }
    return;
  }
  if (const auto* op = op_union.AsStablehloDotGeneralOptions()) {
    attributes.emplace_back(builder.getNamedAttr(
        "lhs_batching_dimensions",
        BuildVhloTensorV1Attr(
            {static_cast<int64_t>(op->lhs_batching_dimensions.size())},
            op->lhs_batching_dimensions, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "rhs_batching_dimensions",
        BuildVhloTensorV1Attr(
            {static_cast<int64_t>(op->rhs_batching_dimensions.size())},
            op->rhs_batching_dimensions, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "lhs_contracting_dimensions",
        BuildVhloTensorV1Attr(
            {static_cast<int64_t>(op->lhs_contracting_dimensions.size())},
            op->lhs_contracting_dimensions, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "rhs_contracting_dimensions",
        BuildVhloTensorV1Attr(
            {static_cast<int64_t>(op->rhs_contracting_dimensions.size())},
            op->rhs_contracting_dimensions, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "precision_config",
        BuildVhloPrecisionConfigV1Attr(op->precision_config, builder)));
    return;
  }
  if (const auto* op = op_union.AsStablehloSortOptions()) {
    attributes.emplace_back(builder.getNamedAttr(
        "dimension", BuildVhloIntV1Attr(op->dimension, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "is_stable",
        mlir::vhlo::BooleanV1Attr::get(builder.getContext(), op->is_stable)));
    return;
  }
  if (const auto* op = op_union.AsStablehloScatterOptions()) {
    attributes.emplace_back(builder.getNamedAttr(
        "update_window_dims",
        BuildVhloTensorV1Attr(
            {static_cast<int64_t>(op->update_window_dims.size())},
            op->update_window_dims, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "inserted_window_dims",
        BuildVhloTensorV1Attr(
            {static_cast<int64_t>(op->inserted_window_dims.size())},
            op->inserted_window_dims, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "scatter_dims_to_operand_dims",
        BuildVhloTensorV1Attr(
            {static_cast<int64_t>(op->scatter_dims_to_operand_dims.size())},
            op->scatter_dims_to_operand_dims, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "unique_indices", mlir::vhlo::BooleanV1Attr::get(builder.getContext(),
                                                         op->unique_indices)));
    attributes.emplace_back(builder.getNamedAttr(
        "indices_are_sorted",
        mlir::vhlo::BooleanV1Attr::get(builder.getContext(),
                                       op->indices_are_sorted)));
    attributes.emplace_back(builder.getNamedAttr(
        "index_vector_dim", BuildVhloIntV1Attr(op->index_vector_dim, builder)));
    return;
  }
  if (const auto* op = op_union.AsStablehloGatherOptions()) {
    attributes.emplace_back(builder.getNamedAttr(
        "slice_sizes",
        BuildVhloTensorV1Attr({static_cast<int64_t>(op->slice_sizes.size())},
                              op->slice_sizes, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "collapsed_slice_dims",
        BuildVhloTensorV1Attr(
            {static_cast<int64_t>(op->collapsed_slice_dims.size())},
            op->collapsed_slice_dims, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "start_index_map",
        BuildVhloTensorV1Attr(
            {static_cast<int64_t>(op->start_index_map.size())},
            op->start_index_map, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "index_vector_dim", BuildVhloIntV1Attr(op->index_vector_dim, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "offset_dims",
        BuildVhloTensorV1Attr({static_cast<int64_t>(op->offset_dims.size())},
                              op->offset_dims, builder)));
    attributes.emplace_back(builder.getNamedAttr(
        "indices_are_sorted",
        mlir::vhlo::BooleanV1Attr::get(builder.getContext(),
                                       op->indices_are_sorted)));
    return;
  }
  if (const auto* op = op_union.AsStablehloTransposeOptions()) {
    if (!op->permutation.empty()) {
      attributes.emplace_back(builder.getNamedAttr(
          "permutation",
          BuildVhloTensorV1Attr({static_cast<int64_t>(op->permutation.size())},
                                op->permutation, builder)));
    }
    return;
  }
  if (const auto* op = op_union.AsStablehloRngBitGeneratorOptions()) {
    mlir::vhlo::RngAlgorithmV1 algorithm;
    switch (op->algorithm) {
      case tflite::RngAlgorithm_THREEFRY:
        algorithm = mlir::vhlo::RngAlgorithmV1::THREE_FRY;
        break;
      case tflite::RngAlgorithm_PHILOX:
        algorithm = mlir::vhlo::RngAlgorithmV1::PHILOX;
        break;
      case tflite::RngAlgorithm_DEFAULT:
        algorithm = mlir::vhlo::RngAlgorithmV1::DEFAULT;
    }
    auto attr =
        mlir::vhlo::RngAlgorithmV1Attr::get(builder.getContext(), algorithm);
    attributes.emplace_back(builder.getNamedAttr("rng_algorithm", attr));
    return;
  }
}

// Pull in FlatBuffer writers for TFLite generated using TableGen
#include "tensorflow/compiler/mlir/lite/operator_converters.inc"
