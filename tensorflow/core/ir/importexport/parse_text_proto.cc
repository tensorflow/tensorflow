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

#include "tensorflow/core/ir/importexport/parse_text_proto.h"

#include <string>

#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/protobuf.h"

using tensorflow::Status;
using tensorflow::errors::InvalidArgument;
using tensorflow::errors::NotFound;

namespace mlir {
namespace tfg {

namespace {
// Error collector that simply ignores errors reported.
class NoOpErrorCollector : public tensorflow::protobuf::io::ErrorCollector {
 public:
  void AddError(int line, int column, const std::string& message) override {}
};
}  // namespace

Status ConsumePrefix(absl::string_view str, absl::string_view prefix,
                     absl::string_view* output) {
  if (absl::StartsWith(str, prefix)) {
    *output = str.substr(prefix.size());
    return absl::OkStatus();
  }
  return NotFound("No prefix \"", prefix, "\" in \"", str, "\"");
}

Status ParseTextProto(absl::string_view text_proto,
                      absl::string_view prefix_to_strip,
                      tensorflow::protobuf::Message* parsed_proto) {
  tensorflow::protobuf::TextFormat::Parser parser;
  // Don't produce errors when attempting to parse text format as it would fail
  // when the input is actually a binary file.
  NoOpErrorCollector collector;
  parser.RecordErrorsTo(&collector);
  // Attempt to parse as text.
  absl::string_view text_proto_without_prefix = text_proto;
  if (!prefix_to_strip.empty()) {
    TF_RETURN_IF_ERROR(
        ConsumePrefix(text_proto, prefix_to_strip, &text_proto_without_prefix));
  }
  tensorflow::protobuf::io::ArrayInputStream input_stream(
      text_proto_without_prefix.data(), text_proto_without_prefix.size());
  if (parser.Parse(&input_stream, parsed_proto)) {
    return absl::OkStatus();
  }
  parsed_proto->Clear();
  return InvalidArgument("Could not parse text proto: ", text_proto);
}

}  // namespace tfg
}  // namespace mlir
