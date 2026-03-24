// Copyright 2026 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// A tool for decomposing an HLO module into individual operations. See kUsage
// for details.

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_clone_context.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_proto_util.h"
#include "xla/status_macros.h"
#include "xla/tools/hlo_decomposer.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/protobuf.h"

namespace xla {

const char* const kUsage = R"(
This tool takes an HLO module as a serialized HloProto or HLO text and extracts
a single HLO instruction. This single instruction is encapsulated into its own
module which is written out to as a HloProto proto or HLO text.

This enables an HLO instruction to be run in isolation. The layouts of the
operations are preserved, and fusion instructions remain intact.

Supported input formats:
 - txt
 - pb (HLO proto)
 - pbtxt (textual HLO proto)
 - snapshot (HLO snapshot)
 - module.pbtxt (HLO module proto)

Supported output formats:
 - short_txt
 - long_txt
 - pb (HLO proto)
 - pbtxt (textual HLO proto)
 - url (graphviz URL)
 - html (dot wrapped in a self-rendering HTML file)
 - dot (dot graph format)

 Example usage:

   isolate_hlo --input=.../hlo_model.pb \
     --output=.../isolated_hlo.pb \
     --input_format=pb --output_format=pb \
     --instruction_name=%conv.2
 )";

static HloInstruction* FindInstruction(const HloModule* module,
                                       absl::string_view name) {
  for (const HloComputation* computation : module->computations()) {
    if (HloInstruction* instruction =
            hlo_query::FindInstruction(computation, name)) {
      return instruction;
    }
  }
  return nullptr;
}

absl::Status RealMain(const std::string& input, const std::string& output,
                      const std::string& instruction_name,
                      const std::string& input_format,
                      const std::string& output_format) {
  QCHECK(!input.empty()) << "Must pass --input flag.";
  QCHECK(!output.empty()) << "Must pass --output flag.";
  QCHECK(!instruction_name.empty()) << "Must pass --instruction_name flag.";

  std::string input_contents;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), input, &input_contents));

  std::unique_ptr<HloModule> module;
  if (input_format == "snapshot") {
    HloSnapshot snapshot;
    if (!snapshot.ParseFromString(input_contents)) {
      return absl::InvalidArgumentError(
          "Failed to parse input as HloSnapshot proto.");
    }
    TF_ASSIGN_OR_RETURN(module,
                        HloModule::CreateFromProto(snapshot.hlo().hlo_module(),
                                                   HloModuleConfig{}));
  } else {
    std::string load_format = input_format;
    if (load_format == "module.pbtxt") {
      load_format = "pbtxt";
    }
    TF_ASSIGN_OR_RETURN(module,
                        LoadModuleFromData(input_contents, load_format));
  }

  std::string instr_name = instruction_name;
  // Drop the "%" from the input to get the instruction name if present.
  if (!instr_name.empty() && instr_name.front() == '%') {
    instr_name = instr_name.substr(1);
  }

  HloInstruction* instruction = FindInstruction(module.get(), instr_name);
  if (!instruction) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Instruction '", instr_name, "' was not found in the hlo module"));
  }

  std::unique_ptr<HloModule> extracted =
      ExtractInstructionIntoNewModule(*instruction);

  std::string output_contents;

  if (output_format == "short_txt") {
    HloPrintOptions options = HloPrintOptions::ShortParsable();
    output_contents = extracted->ToString(options);
  } else if (output_format == "long_txt") {
    HloPrintOptions options;
    output_contents = extracted->ToString(options);
  } else if (output_format == "pb" || output_format == "pbtxt" ||
             output_format == "snapshot" || output_format == "module.pbtxt") {
    HloProto proto = MakeHloProto(*extracted);
    if (output_format == "pb") {
      TF_RET_CHECK(proto.SerializeToString(&output_contents))
          << "Failed to serialize HloProto to string.";
    } else if (output_format == "snapshot") {
      HloSnapshot snapshot;
      *snapshot.mutable_hlo()->mutable_hlo_module() = proto.hlo_module();
      TF_RET_CHECK(snapshot.SerializeToString(&output_contents))
          << "Failed to serialize HloSnapshot to string.";
    } else {
      if (!tsl::protobuf::TextFormat::PrintToString(proto, &output_contents)) {
        return absl::InternalError(absl::StrCat(
            "Failed to print HloProto to string for format: ", output_format));
      }
    }
  } else if (output_format == "url") {
    TF_ASSIGN_OR_RETURN(
        output_contents,
        xla::RenderGraph(*extracted->entry_computation(), /*label=*/"",
                         extracted->config().debug_options(),
                         RenderedGraphFormat::kUrl));
  } else if (output_format == "html") {
    TF_ASSIGN_OR_RETURN(
        output_contents,
        xla::RenderGraph(*extracted->entry_computation(), /*label=*/"",
                         extracted->config().debug_options(),
                         RenderedGraphFormat::kHtml));
  } else if (output_format == "dot") {
    TF_ASSIGN_OR_RETURN(
        output_contents,
        xla::RenderGraph(*extracted->entry_computation(), /*label=*/"",
                         extracted->config().debug_options(),
                         RenderedGraphFormat::kDot));
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unknown output format: ", output_format));
  }

  TF_RETURN_IF_ERROR(
      tsl::WriteStringToFile(tsl::Env::Default(), output, output_contents));

  return absl::OkStatus();
}

}  // namespace xla

int main(int argc, char** argv) {
  std::string input;
  std::string output;
  std::string instruction_name;
  std::string input_format = "pb";
  auto input_format_flag =
      tsl::Flag("input_format", &input_format,
                "The format of the input file. Valid values:\n"
                "  txt: HLO text format\n"
                "  pb: HLO protobuf binary format\n"
                "  pbtxt: HLO protobuf text format\n"
                "  snapshot: HLO snapshot pb\n"
                "  module.pbtxt: HLO module in protobuf text format\n");
  std::string output_format = "short_txt";
  auto output_format_flag =
      tsl::Flag("output_format", &output_format,
                "The format of the output. Valid values:\n"
                "  short_txt: HLO text format without layouts\n"
                "  long_txt: HLO text format with layouts\n"
                "  pb: HLO protobuf binary format\n"
                "  pbtxt: HLO protobuf text format\n"
                "  snapshot: HLO snapshot pb\n"
                "  module.pbtxt: HLO module in protobuf text format\n"
                "  url: A URL to the rendered graph\n"
                "  html: An HTML page with the rendered graph.\n"
                "  dot: DOT graph format");
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("input", &input,
                "XLA HLO module proto to extract isolated HLO instruction from "
                "(required)."),
      tsl::Flag("output", &output,
                "Name of file to write the isolated HLO module proto to "
                "(required)."),
      tsl::Flag("instruction_name", &instruction_name,
                "Name of HLO instruction to isolate in its own module "
                "(required)."),
      input_format_flag, output_format_flag};

  xla::AppendDebugOptionsFlags(&flag_list);
  const std::string usage =
      absl::StrCat(xla::kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));

  const bool parse_result = tsl::Flags::Parse(&argc, argv, flag_list);
  if (!parse_result) {
    std::cerr << usage << std::endl;
    return 1;
  }
  tsl::port::InitMain(usage.c_str(), &argc, &argv);

  absl::Status status = xla::RealMain(input, output, instruction_name,
                                      input_format, output_format);
  if (!status.ok()) {
    std::cerr << "Error: " << status << std::endl;
    return 1;
  }
  return 0;
}
