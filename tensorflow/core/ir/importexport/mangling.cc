/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/ir/importexport/mangling.h"

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/ir/importexport/parse_text_proto.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/protobuf.h"

using tensorflow::DataType;
using tensorflow::Status;
using tensorflow::TensorProto;
using tensorflow::TensorShapeProto;
using tensorflow::errors::FailedPrecondition;

namespace mlir {
namespace tfg {
namespace mangling_util {
namespace {

const char kAttributePrefix[] = "tf.";
const char kDataTypePrefix[] = "tfdtype$";
const char kTensorShapePrefix[] = "tfshape$";
const char kTensorPrefix[] = "tftensor$";

}  // namespace

std::string MangleAttributeName(absl::string_view str) {
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

std::string MangleShape(const TensorShapeProto& shape) {
  return absl::StrCat(kTensorShapePrefix, shape.ShortDebugString());
}

Status DemangleShape(absl::string_view str, TensorShapeProto* proto) {
  return ParseTextProto(str, kTensorShapePrefix, proto);
}

std::string MangleTensor(const TensorProto& tensor) {
  return absl::StrCat(kTensorPrefix, tensor.ShortDebugString());
}

Status DemangleTensor(absl::string_view str, TensorProto* proto) {
  return ParseTextProto(str, kTensorPrefix, proto);
}

std::string MangleDataType(const DataType& dtype) {
  return absl::StrCat(kDataTypePrefix, DataType_Name(dtype));
}

Status DemangleDataType(absl::string_view str, DataType* proto) {
  absl::string_view pbtxt;
  TF_RETURN_IF_ERROR(ConsumePrefix(str, kDataTypePrefix, &pbtxt));
  // NOLINTNEXTLINE: redundant string conversion for divergence in OSS API.
  if (!DataType_Parse(std::string(pbtxt), proto)) {
    return FailedPrecondition("Could not parse TFDataType mangled proto");
  }
  return ::tensorflow::OkStatus();
}

}  // namespace mangling_util
}  // namespace tfg
}  // namespace mlir
