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

#ifndef XLA_TOOLS_MULTIHOST_HLO_RUNNER_HLO_INPUT_OUTPUT_FORMAT_H_
#define XLA_TOOLS_MULTIHOST_HLO_RUNNER_HLO_INPUT_OUTPUT_FORMAT_H_

#include <cstdint>
#include <string>

#include "absl/strings/string_view.h"

namespace xla {

// Supported input formats for the input HLO module.
enum class InputFormat {
  kText,                 // Text format returned by HloModule::ToString().
  kProtoText,            // Protobuf text format of an xla::HloProto message.
  kProtoBinary,          // Protobuf binary format of an xla::HloProto message.
  kSnapshotProtoBinary,  // HloSnapshot protobuf binary format. Can be dumped by
                         // TensorFlow by setting the environment variable
                         // xla_dump_hlo_snapshots.
  kUnoptimizedSnapshotProtoBinary,  // HloUnoptimizedSnapshot protobuf binary
                                    // format. Can be dumped by
                                    // setting the flag
                                    // xla_dump_hlo_snapshots in conjunction
                                    // with xla_dump_as_text.
  kUnoptimizedSnapshotProtoText,    // HloUnoptimizedSnapshot protobuf text
                                    // format. Can be dumped by TensorFlow by
                                    // setting the flag xla_dump_hlo_snapshots
                                    // in conjunction with xla_dump_as_text.
};

bool AbslParseFlag(absl::string_view text, InputFormat* input_format,
                   std::string* error);
std::string AbslUnparseFlag(InputFormat input_format);

enum class OutputFormat : std::uint8_t {
  kText,         // Text format returned by Literal::ToString().
  kProtoBinary,  // Protobuf binary format of an xla::LiteralProto message.
  kProtoText,    // Protobuf text format of an xla::LiteralProto message.
};

bool AbslParseFlag(absl::string_view text, OutputFormat* output_format,
                   std::string* error);
std::string AbslUnparseFlag(OutputFormat output_format);

}  // namespace xla

#endif  // XLA_TOOLS_MULTIHOST_HLO_RUNNER_HLO_INPUT_OUTPUT_FORMAT_H_
