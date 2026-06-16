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

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "riegeli/base/maker.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/std_io.h"
#include "riegeli/bytes/writer.h"
#include "xla/service/riegeli_file_reader_factory.h"
#include "xla/service/riegeli_file_writer_factory.h"
#include "xla/tools/split_proto/split_proto_cli_lib.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"

namespace xla::split_proto_cli {
namespace {

constexpr absl::string_view kUsageHeader = R"(
split-proto-cli is a command-line tool for converting XLA protocol buffers
between standard serialized formats (text/binary) and the Split Proto format
(Riegeli), which is used to serialized the AOT compiled executables.

Subcommands:
  unpack-aot:        Unpacks an AOT binary (Split Proto
                     ExecutableAndOptionsProto) into a single text/binary
                     proto file of type DeserializedSplitExecutableAndOptions.
  pack-aot:          Packs a DeserializedSplitExecutableAndOptions (obtained
                     from unpack-aot) back into a Split Proto
                     ExecutableAndOptionsProto, i.e. the reverse of
                     unpack-aot.
  unpack:            Reconstructs a standard protobuf from a split proto file.
                     The proto type is automatically inferred from the split
                     manifest.
  pack:              Converts a standard protobuf (text or binary) into a split
                     proto. Requires `--proto_type` to identify the message type
  diff:              Compares two split proto files and prints differences to
                     stdout.

Usage:
  # Convert an AOT binary into a textproto, so that you can inspect the
  # contents.
  split-proto-cli unpack-aot aot_binary.riegeli --output_file=output.textproto

  # Re-pack the textproto (which you might have edited) back into a binary.
  split-proto-cli pack-aot output.textproto --output_file=repacked.riegeli

  # Pack a standard proto (text or binary) into a split proto.
  split-proto-cli pack --proto_type=<type> aot_binary.riegeli

  # Unpack a split proto into a standard proto (text or binary).
  split-proto-cli unpack aot_binary.riegeli

  # Compare two split proto files and print differences.
  split-proto-cli diff <file1> <file2>

Input/Output:
  If the input file is omitted or '-', it reads from stdin.
  Output defaults to stdout, or can be specified via `--output_file`.

Supported Proto Types:
  - xla.gpu.GpuExecutableProto
  - xla.ExecutableAndOptionsProto
)";

absl::Status RunMain(int argc, char** argv) {
  std::string output_file;
  std::string output_format_str = "text";
  std::string input_format_str = "auto";
  std::string proto_type;

  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("output_file", &output_file,
                "The path to write the output to. If empty or '-', writes to "
                "stdout."),
      tsl::Flag("output_format", &output_format_str,
                "The output format: 'text' or 'binary'."),
      tsl::Flag("input_format", &input_format_str,
                "The input format: 'text', 'binary', or 'auto'."),
      tsl::Flag(
          "proto_type", &proto_type,
          "The fully qualified protobuf type name (required for packing, e.g. "
          "xla.gpu.GpuExecutableProto or xla.ExecutableAndOptionsProto)."),
  };

  std::string usage = std::string(kUsageHeader);
  absl::StrAppend(&usage, tsl::Flags::Usage(argv[0], flag_list));
  if (argc > 1 && absl::string_view(argv[1]) == "--help") {
    std::cerr << usage << "\n";
    return absl::OkStatus();
  }

  if (!tsl::Flags::Parse(&argc, argv, flag_list)) {
    std::cerr << "Failed to parse flags.\n\n" << usage << "\n";
    return absl::InvalidArgumentError("Failed to parse flags.");
  }

  tsl::port::InitMain(usage.c_str(), &argc, &argv);

  if (argc < 2) {
    std::cerr << kUsageHeader << "\n";
    return absl::InvalidArgumentError("Missing subcommand");
  }
  std::string subcommand = argv[1];

  std::unique_ptr<riegeli::Reader> reader;
  if (subcommand != "diff") {
    if (argc < 3 || std::string(argv[2]) == "-") {
      LOG(INFO) << "Reading input from stdin";
      reader = riegeli::Maker<riegeli::StdIn>();
    } else {
      LOG(INFO) << "Reading input from file: " << argv[2];
      reader = CreateRiegeliFileReader(argv[2]);
      TF_RETURN_WITH_CONTEXT_IF_ERROR(
          reader->status(),
          absl::StrCat("Failed to open input file: ", argv[2]));
    }
  } else if (argc != 4) {
    return absl::InvalidArgumentError(
        "Usage: split-proto-cli diff <file1> <file2>");
  }

  std::unique_ptr<riegeli::Writer> writer;
  if (output_file.empty() || output_file == "-") {
    LOG(INFO) << "Output will be written to stdout";
    writer = riegeli::Maker<riegeli::StdOut>();
  } else {
    LOG(INFO) << "Output will be written to file: " << output_file;
    writer = CreateRiegeliFileWriter(output_file);
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        writer->status(),
        absl::StrCat("Failed to open output file: ", output_file));
  }

  auto parse_format = [](absl::string_view str,
                         ProtoFormat* format) -> absl::Status {
    std::string error;
    if (!AbslParseFlag(str, format, &error)) {
      return absl::InvalidArgumentError(error);
    }
    return absl::OkStatus();
  };

  absl::Status status;
  if (subcommand == "pack") {
    PackOptions options;
    options.proto_type = proto_type;
    if (options.proto_type.empty()) {
      return absl::InvalidArgumentError(
          "Flag --proto_type is required for pack subcommand.");
    }

    RETURN_IF_ERROR(parse_format(input_format_str, &options.input_format));

    status = Pack(std::move(reader), std::move(writer), options);
  } else if (subcommand == "unpack") {
    UnpackOptions options;
    RETURN_IF_ERROR(parse_format(output_format_str, &options.output_format));

    status = Unpack(std::move(reader), std::move(writer), options);
  } else if (subcommand == "pack-aot") {
    PackOptions options;
    RETURN_IF_ERROR(parse_format(input_format_str, &options.input_format));

    status = PackAot(std::move(reader), std::move(writer), options);
  } else if (subcommand == "unpack-aot") {
    UnpackOptions options;
    RETURN_IF_ERROR(parse_format(output_format_str, &options.output_format));

    status = UnpackAot(std::move(reader), std::move(writer), options);
  } else if (subcommand == "diff") {
    LOG(INFO) << "Reading input from file: " << argv[2] << " and " << argv[3];
    status = Diff(CreateRiegeliFileReader(argv[2]),
                  CreateRiegeliFileReader(argv[3]), std::move(writer));
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unknown subcommand: ", subcommand));
  }

  return status;
}

}  // namespace
}  // namespace xla::split_proto_cli

int main(int argc, char** argv) {
  absl::Status status = xla::split_proto_cli::RunMain(argc, argv);
  if (!status.ok()) {
    std::cerr << "Error: " << status.ToString() << "\n";
    return 1;
  }
  return 0;
}
