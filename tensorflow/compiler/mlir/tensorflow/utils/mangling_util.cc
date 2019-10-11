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

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
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

// Sets output to the given input with 'prefix' stripped, or return an error if
// the prefix did not exist.
Status ConsumePrefix(absl::string_view str, absl::string_view prefix,
                     absl::string_view* output) {
  if (absl::StartsWith(str, prefix)) {
    *output = str.substr(prefix.size());
    return Status::OK();
  }
  return errors::FailedPrecondition("Not a mangled string");
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
  return absl::StrCat(kTensorShapePrefix, shape.ShortDebugString());
}

Status DemangleShape(absl::string_view str, TensorShapeProto* proto) {
  absl::string_view pbtxt;
  TF_RETURN_IF_ERROR(ConsumePrefix(str, kTensorShapePrefix, &pbtxt));
  tensorflow::protobuf::io::ArrayInputStream input_stream(pbtxt.data(),
                                                          pbtxt.size());
  if (!tensorflow::protobuf::TextFormat::Parse(&input_stream, proto)) {
    return errors::FailedPrecondition(
        "Could not parse TFTensorShape mangled proto");
  }
  return Status::OK();
}

string MangleTensor(const TensorProto& tensor) {
  return absl::StrCat(kTensorPrefix, tensor.ShortDebugString());
}

Status DemangleTensor(absl::string_view str, TensorProto* proto) {
  absl::string_view pbtxt;
  TF_RETURN_IF_ERROR(ConsumePrefix(str, kTensorPrefix, &pbtxt));
  tensorflow::protobuf::io::ArrayInputStream input_stream(pbtxt.data(),
                                                          pbtxt.size());
  if (!tensorflow::protobuf::TextFormat::Parse(&input_stream, proto)) {
    return errors::FailedPrecondition("Could not parse TFTensor mangled proto");
  }
  return Status::OK();
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
  return Status::OK();
}

}  // namespace mangling_util
}  // namespace tensorflow
