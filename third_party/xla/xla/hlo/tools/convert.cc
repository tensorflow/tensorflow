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

#include <stdio.h>

#include <string>
#include <vector>

#include "xla/service/hlo_module_util.h"
#include "xla/service/hlo_proto_util.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/tsl/util/fixed_option_set_flag.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace {

constexpr char kUsage[] = R"(
  Reads an HLO module and outputs it in the requested format.
  Supported formats: text, proto_text, proto_binary.
)";

enum class Format {
  kText,
  kProtoBinary,
  kProtoText,
};

bool ParseFlag(absl::string_view text, Format* format, std::string* error) {
  return GetFixedOptionSetFlagParser<Format>(
             {{"text", Format::kText},
              {"proto_binary", Format::kProtoBinary},
              {"proto_text", Format::kProtoText}})
      .Parse(text, format, error);
}

absl::Status RealMain(absl::string_view input_file,
                      absl::string_view output_file, Format input_format,
                      Format output_format) {
  std::unique_ptr<HloModule> module;
  switch (input_format) {
    case Format::kText: {
      TF_ASSIGN_OR_RETURN(module, ReadModuleFromHloTextFile(input_file));
      break;
    }
    case Format::kProtoText: {
      TF_ASSIGN_OR_RETURN(module, ReadModuleFromTextProtoFile(input_file));
      break;
    }
    case Format::kProtoBinary: {
      TF_ASSIGN_OR_RETURN(module, ReadModuleFromBinaryProtoFile(input_file));
      break;
    }
  }
  std::string result;
  switch (output_format) {
    case Format::kText:
      result = module->ToString();
      break;
    case Format::kProtoText:
      if (!tsl::protobuf::TextFormat::PrintToString(MakeHloProto(*module),
                                                    &result)) {
        return absl::InternalError("Proto to text conversion failed.");
      }
      break;
    case Format::kProtoBinary:
      MakeHloProto(*module).AppendToString(&result);
  }
  if (output_file == "-") {
    std::cout << result;
    return absl::OkStatus();
  }
  return tsl::WriteStringToFile(tsl::Env::Default(), std::string(output_file),
                                result);
}

}  // namespace
}  // namespace xla

int main(int argc, char** argv) {
  std::string output_file = "-", input_format_str = "text",
              output_format_str = "text";
  const std::vector<tsl::Flag> flag_list = {
      tsl::Flag("output", &output_file, "Output file. '-' for stdout."),
      tsl::Flag("input_format", &input_format_str, "Input format."),
      tsl::Flag("output_format", &output_format_str, "Output format."),
  };
  const std::string kUsageAndFlags =
      absl::StrCat(xla::kUsage, "\n", tsl::Flags::Usage(argv[0], flag_list));
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(kUsageAndFlags.c_str(), &argc, &argv);
  CHECK(parse_ok && argc == 2) << "\n" << kUsageAndFlags;

  xla::Format input_format;
  xla::Format output_format;
  if (std::string error; !ParseFlag(input_format_str, &input_format, &error)) {
    LOG(ERROR) << "Failed parsing input format: " + error;
  }
  if (std::string error;
      !ParseFlag(output_format_str, &output_format, &error)) {
    LOG(ERROR) << "Failed parsing output format: " + error;
  }

  absl::Status result =
      xla::RealMain(argv[1], output_file, input_format, output_format);
  if (!result.ok()) {
    LOG(ERROR) << result.message();
  }
  return result.raw_code();
}
