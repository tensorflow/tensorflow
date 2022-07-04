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

#include "tensorflow/compiler/mlir/tensorflow/utils/mangling_util.h"

#include <string>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/parse_text_proto.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace mangling_util {
namespace {

const char kAttributePrefix[] = "tf.";
const char kDataTypePrefix[] = "tfdtype$";
const char kTensorShapePrefix[] = "tfshape$";
const char kTensorPrefix[] = "tftensor$";

std::string PrintShortTextProto(
    const ::tensorflow::protobuf::MessageLite& message) {
  // proto2::TextFormat::Printer::PrintToString does not have
  // a overload for MessageLite so here to be consistent with the existing
  // behavior we use MessageLite::ShortDebugString().
  return message.ShortDebugString();
}

std::string PrintShortTextProto(
    const ::tensorflow::protobuf::Message& message) {
  std::string message_short_text;

  ::tensorflow::protobuf::TextFormat::Printer printer;
  printer.SetSingleLineMode(true);
  printer.SetExpandAny(true);

  printer.PrintToString(message, &message_short_text);
  // Single line mode currently might have an extra space at the end.
  if (!message_short_text.empty() && message_short_text.back() == ' ') {
    message_short_text.pop_back();
  }

  return message_short_text;
}

}  // namespace

string MangleAttributeName(absl::string_view str) {
  return absl::StrCat(kAttributePrefix, str);
}

bool IsMangledAttributeName(absl::string_view str) {
  return absl::StartsWith(str, kAttributePrefix);
}

absl::string_view DemangleAttributeName(absl::string_view str) {
  DCHECK(IsMangledAttributeName(str));
  return str.substr(std::strlen(kAttributePrefix));
}

MangledKind GetMangledKind(absl::string_view str) {
  if (absl::StartsWith(str, kDataTypePrefix)) {
    return MangledKind::kDataType;
  } else if (absl::StartsWith(str, kTensorShapePrefix)) {
    return MangledKind::kTensorShape;
  } else if (absl::StartsWith(str, kTensorPrefix)) {
    return MangledKind::kTensor;
  } else {
    return MangledKind::kUnknown;
  }
}

string MangleShape(const TensorShapeProto& shape) {
  return absl::StrCat(kTensorShapePrefix, PrintShortTextProto(shape));
}

Status DemangleShape(absl::string_view str, TensorShapeProto* proto) {
  return ParseTextProto(str, kTensorShapePrefix, proto);
}

string MangleTensor(const TensorProto& tensor) {
  return absl::StrCat(kTensorPrefix, PrintShortTextProto(tensor));
}

Status DemangleTensor(absl::string_view str, TensorProto* proto) {
  return ParseTextProto(str, kTensorPrefix, proto);
}

string MangleDataType(const DataType& dtype) {
  return absl::StrCat(kDataTypePrefix, DataType_Name(dtype));
}

Status DemangleDataType(absl::string_view str, DataType* proto) {
  absl::string_view pbtxt;
  TF_RETURN_IF_ERROR(ConsumePrefix(str, kDataTypePrefix, &pbtxt));
  if (!DataType_Parse(string(pbtxt), proto)) {
    return errors::FailedPrecondition(
        "Could not parse TFDataType mangled proto");
  }
  return OkStatus();
}

}  // namespace mangling_util
}  // namespace tensorflow
