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

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "flatbuffers/string.h"  // from @flatbuffers
#include "flatbuffers/vector.h"  // from @flatbuffers
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/flatbuffer_operator_writers.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/schema/mutable/schema_generated.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"

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

// Serializes one `quant_spec` value. Only the kinds that the specs actually
// use are supported; anything else is a bug in the pass that built the
// dictionary, and dropping it would silently change the op's meaning.
//
// The key is taken by `const std::string&` rather than `llvm::StringRef`
// because the `flexbuffers::Builder` key overloads call `strlen` on it, and a
// `StringRef` is not guaranteed to be null-terminated.
static void SerializeQuantSpecValue(const std::string& key,
                                    mlir::Attribute value,
                                    flexbuffers::Builder* fbb) {
  if (auto str = mlir::dyn_cast<mlir::StringAttr>(value)) {
    fbb->String(key.c_str(), str.getValue().str());
  } else if (auto boolean = mlir::dyn_cast<mlir::BoolAttr>(value)) {
    fbb->Bool(key.c_str(), boolean.getValue());
  } else if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(value)) {
    fbb->Int(key.c_str(), integer.getInt());
  } else if (auto fp = mlir::dyn_cast<mlir::FloatAttr>(value)) {
    fbb->Double(key.c_str(), fp.getValueAsDouble());
  } else {
    llvm::report_fatal_error(llvm::Twine("quant_spec key '") + key +
                             "' has an unsupported attribute kind");
  }
}

static flatbuffers::Offset<flatbuffers::Vector<uint8_t>>
ConvertDerivedQuantSpecAttrForOptionWriter(
    mlir::DictionaryAttr attr, flatbuffers::FlatBufferBuilder* builder) {
  if (!attr || attr.empty()) return 0;

  flexbuffers::Builder fbb;
  const size_t map_start = fbb.StartMap();
  for (const mlir::NamedAttribute& named : attr) {
    SerializeQuantSpecValue(named.getName().str(), named.getValue(), &fbb);
  }
  fbb.EndMap(map_start);
  fbb.Finish();
  return builder->CreateVector(fbb.GetBuffer());
}

// Pull in FlatBuffer writers for TFLite generated using TableGen
#include "tensorflow/compiler/mlir/lite/operator_converters.inc"
