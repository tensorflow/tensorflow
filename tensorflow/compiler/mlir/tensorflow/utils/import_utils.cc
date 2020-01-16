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
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/protobuf.h"

namespace {
// Error collector that simply ignores errors reported.
class NoOpErrorCollector : public tensorflow::protobuf::io::ErrorCollector {
 public:
  void AddError(int line, int column, const std::string& message) override {}
};

inline llvm::StringRef StringViewToRef(absl::string_view view) {
  return {view.data(), view.size()};
}
}  // namespace

namespace tensorflow {

Status LoadProtoFromBuffer(absl::string_view input,
                           tensorflow::protobuf::Message* proto) {
  tensorflow::protobuf::TextFormat::Parser parser;
  // Don't produce errors when attempting to parse text format as it would fail
  // when the input is actually a binary file.
  NoOpErrorCollector collector;
  parser.RecordErrorsTo(&collector);
  // Attempt to parse as text.
  tensorflow::protobuf::io::ArrayInputStream input_stream(input.data(),
                                                          input.size());
  if (parser.Parse(&input_stream, proto)) {
    return Status::OK();
  }
  // Else attempt to parse as binary.
  proto->Clear();
  tensorflow::protobuf::io::ArrayInputStream binary_stream(input.data(),
                                                           input.size());
  if (proto->ParseFromZeroCopyStream(&binary_stream)) {
    return Status::OK();
  }
  LOG(ERROR) << "Error parsing Protobuf";
  return errors::InvalidArgument("Could not parse input proto");
}

Status LoadProtoFromFile(absl::string_view input_filename,
                         tensorflow::protobuf::Message* proto) {
  auto file_or_err =
      llvm::MemoryBuffer::getFileOrSTDIN(StringViewToRef(input_filename));
  if (std::error_code error = file_or_err.getError())
    return errors::InvalidArgument("Could not open input file");

  auto& input_file = *file_or_err;
  absl::string_view content(input_file->getBufferStart(),
                            input_file->getBufferSize());

  return LoadProtoFromBuffer(content, proto);
}

}  // namespace tensorflow
