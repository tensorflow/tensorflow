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

#include <vector>

#include "absl/strings/str_cat.h"
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/lite/kernels/internal/kernel_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

using ::tensorflow::Status;
using ::tensorflow::errors::InvalidArgument;
using ::xla::StatusOr;

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
    llvm::StringRef str, flatbuffers::FlatBufferBuilder* builder) {
  return llvm::StringSwitch<tflite::MirrorPadMode>(str)
      .Case("REFLECT", tflite::MirrorPadMode_REFLECT)
      .Case("SYMMETRIC", tflite::MirrorPadMode_SYMMETRIC);
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
    intVec.push_back(attr.cast<mlir::IntegerAttr>().getInt());
  }
  return builder->CreateVector(intVec);
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
    llvm::StringRef str, flatbuffers::FlatBufferBuilder* builder) {
  return llvm::StringSwitch<tflite::LSTMKernelType>(str)
      .Case("FULL", tflite::LSTMKernelType_FULL)
      .Case("BASIC", tflite::LSTMKernelType_BASIC);
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

static mlir::Attribute BuildI64ArrayAttr(std::vector<int32_t> value,
                                         mlir::Builder builder) {
  std::vector<int64_t> typecast(value.begin(), value.end());
  return builder.getI64ArrayAttr(typecast);
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
  const char* option_name = tflite::EnumNameLSTMKernelType(value);
  return builder.getStringAttr(option_name);
}

static mlir::Attribute BuildTFL_MirrorPaddingAttr(tflite::MirrorPadMode value,
                                                  mlir::Builder builder) {
  const char* option_name = tflite::EnumNameMirrorPadMode(value);
  return builder.getStringAttr(option_name);
}

static mlir::Attribute BuildTFL_PaddingAttr(tflite::Padding value,
                                            mlir::Builder builder) {
  const char* option_name = tflite::EnumNamePadding(value);
  return builder.getStringAttr(option_name);
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
  ShapedType type = RankedTensorType::get(
      {static_cast<int64_t>(custom_options.size())}, builder.getIntegerType(8));
  attributes->emplace_back(builder.getNamedAttr(
      "custom_option",
      OpaqueElementsAttr::get(builder.getContext()->getLoadedDialect("tfl"),
                              type, content)));

  return Status::OK();
}

// Pull in FlatBuffer writers for TFLite generated using TableGen
#include "tensorflow/compiler/mlir/lite/operator_converters.inc"
