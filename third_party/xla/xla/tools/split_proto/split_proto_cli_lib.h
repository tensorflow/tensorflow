/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#ifndef XLA_TOOLS_SPLIT_PROTO_SPLIT_PROTO_CLI_LIB_H_
#define XLA_TOOLS_SPLIT_PROTO_SPLIT_PROTO_CLI_LIB_H_

#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"

namespace xla::split_proto_cli {

// Format of the input/output protocol buffers.
enum class ProtoFormat {
  kText,
  kBinary,
  // Attempt to auto-detect format based on the input.
  kAuto,
};

// Functions to parse and unparse ProtoFormat flag values for Abseil flags.
bool AbslParseFlag(absl::string_view text, ProtoFormat* format,
                   std::string* error);
std::string AbslUnparseFlag(ProtoFormat format);

// Options for packing a protocol buffer.
struct PackOptions {
  // Format of the input protocol buffer.
  ProtoFormat input_format = ProtoFormat::kAuto;
  // The fully qualified protobuf type name (e.g. "xla.gpu.GpuExecutableProto").
  std::string proto_type;
};

// Options for unpacking a protocol buffer.
struct UnpackOptions {
  // Format to write the output protocol buffer in.
  ProtoFormat output_format = ProtoFormat::kText;
};

// Reads a standard serialized protobuf message from `reader` according to
// `options.input_format` and `options.proto_type`, packs it using split
// representation, and writes the output record structure to `writer`.
absl::Status Pack(std::unique_ptr<riegeli::Reader> reader,
                  std::unique_ptr<riegeli::Writer> writer,
                  const PackOptions& options);

// Reads a split-represented protocol buffer from `reader`, reconstructs the
// original protobuf message, and writes the serialized result to `writer` in
// the format specified by `options.output_format`.
absl::Status Unpack(std::unique_ptr<riegeli::Reader> reader,
                    std::unique_ptr<riegeli::Writer> writer,
                    const UnpackOptions& options);

// Reads a standard serialized `DeserializedSplitExecutableAndOptions` protobuf
// from `reader`, and writes it as a Split Proto `ExecutableAndOptionsProto` to
// `writer`, i.e it converts it to the standard AOT binary format.
absl::Status PackAot(std::unique_ptr<riegeli::Reader> reader,
                     std::unique_ptr<riegeli::Writer> writer,
                     const PackOptions& options);

// Reads a split `ExecutableAndOptionsProto` from `reader` (which is the
// standard AOT binary format), and writes a standard serialized
// `DeserializedSplitExecutableAndOptions` protobuf to `writer` in the format
// specified by `options.output_format`.
absl::Status UnpackAot(std::unique_ptr<riegeli::Reader> reader,
                       std::unique_ptr<riegeli::Writer> writer,
                       const UnpackOptions& options);

// Reads a split proto serialized ExecutableAndOptions (i.e. an AOT binary) from
// `reader`, and prints useful info to stdout.
absl::Status AotInfo(std::unique_ptr<riegeli::Reader> reader);

}  // namespace xla::split_proto_cli

#endif  // XLA_TOOLS_SPLIT_PROTO_SPLIT_PROTO_CLI_LIB_H_
