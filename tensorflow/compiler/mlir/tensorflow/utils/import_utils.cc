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
#include "tensorflow/compiler/mlir/tensorflow/utils/import_utils.h"

#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/parse_text_proto.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {
namespace {
inline llvm::StringRef StringViewToRef(absl::string_view view) {
  return {view.data(), view.size()};
}
}  // namespace

Status LoadProtoFromBuffer(absl::string_view input, protobuf::Message* proto) {
  // Attempt to parse as text.
  if (ParseTextProto(input, "", proto).ok()) return OkStatus();

  // Else attempt to parse as binary.
  return LoadProtoFromBuffer(input, static_cast<protobuf::MessageLite*>(proto));
}

Status LoadProtoFromBuffer(absl::string_view input,
                           protobuf::MessageLite* proto) {
  // Attempt to parse as binary.
  protobuf::io::ArrayInputStream binary_stream(input.data(), input.size());
  if (proto->ParseFromZeroCopyStream(&binary_stream)) return OkStatus();

  LOG(ERROR) << "Error parsing Protobuf";
  return errors::InvalidArgument("Could not parse input proto");
}

template <class T>
Status LoadProtoFromFileImpl(absl::string_view input_filename, T* proto) {
  const auto file_or_err =
      llvm::MemoryBuffer::getFileOrSTDIN(StringViewToRef(input_filename));
  if (std::error_code error = file_or_err.getError()) {
    return errors::InvalidArgument(
        "Could not open input file ",
        string(input_filename.data(), input_filename.size()).c_str());
  }

  const auto& input_file = *file_or_err;
  absl::string_view content(input_file->getBufferStart(),
                            input_file->getBufferSize());
  return LoadProtoFromBuffer(content, proto);
}

Status LoadProtoFromFile(absl::string_view input_filename,
                         protobuf::Message* proto) {
  return LoadProtoFromFileImpl(input_filename, proto);
}

Status LoadProtoFromFile(absl::string_view input_filename,
                         protobuf::MessageLite* proto) {
  return LoadProtoFromFileImpl(input_filename, proto);
}

}  // namespace tensorflow
