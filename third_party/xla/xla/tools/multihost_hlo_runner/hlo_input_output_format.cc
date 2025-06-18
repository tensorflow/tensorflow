/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/tools/multihost_hlo_runner/hlo_input_output_format.h"

#include <string>

#include "absl/strings/string_view.h"
#include "xla/tsl/util/fixed_option_set_flag.h"

namespace xla {

static const FixedOptionSetFlagParser<InputFormat>& GetInputFormatParser() {
  static const auto& parser = GetFixedOptionSetFlagParser<InputFormat>(
      {{"text", InputFormat::kText},
       {"proto_text", InputFormat::kProtoText},
       {"proto_binary", InputFormat::kProtoBinary},
       {"snapshot_proto_binary", InputFormat::kSnapshotProtoBinary},
       {"unoptimized_snapshot_proto_binary",
        InputFormat::kUnoptimizedSnapshotProtoBinary},
       {"unoptimized_snapshot_proto_text",
        InputFormat::kUnoptimizedSnapshotProtoText}});
  return parser;
}

static const FixedOptionSetFlagParser<OutputFormat>& GetOutputFormatParser() {
  static const auto& parser = GetFixedOptionSetFlagParser<OutputFormat>(
      {{"text", OutputFormat::kText},
       {"proto_binary", OutputFormat::kProtoBinary},
       {"proto_text", OutputFormat::kProtoText}});
  return parser;
}

bool AbslParseFlag(absl::string_view text, InputFormat* input_format,
                   std::string* error) {
  return GetInputFormatParser().Parse(text, input_format, error);
}

std::string AbslUnparseFlag(InputFormat input_format) {
  return GetInputFormatParser().Unparse(input_format);
}

bool AbslParseFlag(absl::string_view text, OutputFormat* output_format,
                   std::string* error) {
  return GetOutputFormatParser().Parse(text, output_format, error);
}

std::string AbslUnparseFlag(OutputFormat output_format) {
  return GetOutputFormatParser().Unparse(output_format);
}
}  // namespace xla
