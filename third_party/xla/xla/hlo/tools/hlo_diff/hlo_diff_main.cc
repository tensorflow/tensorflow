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

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_summary.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_diff.h"
#include "xla/hlo/tools/hlo_diff/render/hlo_gumgraph_html_renderer.h"
#include "xla/hlo/tools/hlo_diff/render/hlo_gumgraph_text_renderer.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_module_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"

namespace xla {
namespace hlo_diff {
namespace {

const char* const kUsage = R"(
Given two HLO Modules, compares the graph structure of two HLO Modules and 
summarizes the differences in a human readable format. The tool focuses on 
computational differences ignoring irrelevant changes such as instruction 
names, parameter ordering etc, layouts (in some instances).

  Usage:
      bazel run hlo_diff -- \
        --{first_hlo_snapshot,first_hlo_proto,first_hlo_module_proto,first_hlo_text}=path/to/first/binary_proto
        --{second_hlo_snapshot,second_hlo_proto,second_hlo_module_proto,second_hlo_text}=path/to/second/binary_proto
        [--ignore_shape_during_instruction_matching]
        [--text_output=path/to/file/to/save/text]
        [--html_output=path/to/file/to/save/html]

first and second hlo file paths are required flags. Optionally the following
flags can be used:

If --ignore_shape_during_instruction_matching is specified, the tool ignores
array/tensor shapes when matching instructions allowing for more permissive
matches.
If --text_output is specified, the full diff result will be printed in text
format and saved to the specified file.
if --html_output is specified, the diff result will be rendered in HTML
format and saved to the specified path.
)";

// Command line opts to this tool. See the main() for descriptions of these
// fields.
struct Options {
  struct HloPath {
    std::string hlo_snapshot;
    std::string hlo_proto;
    std::string hlo_module_proto;
    std::string hlo_text;
  };

  struct RenderOptions {
    std::string text_output;
    std::string html_output;
  };

  HloPath first;
  HloPath second;
  DiffOptions diff_options;
  RenderOptions render_options;
};

absl::Status CheckGroupFlags(const Options::HloPath& hlo_path) {
  int nonempty_options_amount = 0;
  for (const auto& path : {hlo_path.hlo_snapshot, hlo_path.hlo_proto,
                           hlo_path.hlo_module_proto, hlo_path.hlo_text}) {
    if (!path.empty()) {
      ++nonempty_options_amount;
    }
  }
  return nonempty_options_amount == 1
             ? absl::OkStatus()
             : absl::FailedPreconditionError(
                   "Can only specify one and only one of path flags.");
}

// Builds a HloModule from the HloModuleProto.
absl::StatusOr<std::unique_ptr<HloModule>> BuildHloModule(
    const HloModuleProto& hlo_module_proto) {
  TF_ASSIGN_OR_RETURN(HloModuleConfig config,
                      HloModule::CreateModuleConfigFromProto(
                          hlo_module_proto, xla::GetDebugOptionsFromFlags()));
  return HloModule::CreateFromProto(hlo_module_proto, config);
}

absl::StatusOr<std::unique_ptr<HloModule>> LoadHLOModule(
    const Options::HloPath& hlo_path) {
  if (!hlo_path.hlo_snapshot.empty()) {
    HloSnapshot snapshot;
    TF_CHECK_OK(tsl::ReadBinaryProto(tsl::Env::Default(), hlo_path.hlo_snapshot,
                                     &snapshot))
        << "Can't open, read, or parse HloSnapshot proto at "
        << hlo_path.hlo_snapshot;
    return BuildHloModule(snapshot.hlo().hlo_module());
  }
  if (!hlo_path.hlo_proto.empty()) {
    return ReadModuleFromBinaryProtoFile(hlo_path.hlo_proto,
                                         xla::GetDebugOptionsFromFlags());
  }
  if (!hlo_path.hlo_module_proto.empty()) {
    return ReadModuleFromModuleBinaryProtofile(hlo_path.hlo_module_proto,
                                               xla::GetDebugOptionsFromFlags());
  }
  if (!hlo_path.hlo_text.empty()) {
    return ReadModuleFromHloTextFile(
        hlo_path.hlo_text, xla::GetDebugOptionsFromFlags(),
        xla::HloParserOptions().set_fill_shortform_constants_with_random_values(
            false));
  }

  return absl::InvalidArgumentError("No hlo_path specified.");
}

// Runs Gumgraph algorithm based diff and renders the diff results.
absl::Status RunGumgraphDiff(HloModule& first_module, HloModule& second_module,
                             const Options& opts) {
  TF_RETURN_IF_ERROR(first_module.RemoveUnusedComputations());
  TF_RETURN_IF_ERROR(second_module.RemoveUnusedComputations());

  TF_ASSIGN_OR_RETURN(
      auto hlo_gumgraph_diff,
      ComputeDiff(first_module, second_module, opts.diff_options));
  std::cout << "Diffing finished" << '\n';

  const DiffResult& diff = *hlo_gumgraph_diff.diff_result;
  const DiffSummary& diff_summary = *hlo_gumgraph_diff.diff_summary;
  LogDiffResult(diff);
  std::ostringstream text;
  RenderTextSummary(diff, text);
  std::cout << text.str() << '\n';

  const std::string& text_output = opts.render_options.text_output;
  if (!text_output.empty()) {
    std::ostringstream text;
    RenderText(diff, text);
    TF_RETURN_IF_ERROR(
        tsl::WriteStringToFile(tsl::Env::Default(), text_output, text.str()));
  }

  std::string html_output = opts.render_options.html_output;
  if (!html_output.empty()) {
    std::ostringstream html;
    RenderHtml(diff, diff_summary, html);
    TF_RETURN_IF_ERROR(
        tsl::WriteStringToFile(tsl::Env::Default(), html_output, html.str()));

    std::cout << "The diff summary is saved to: " << html_output << '\n';
  }

  return absl::OkStatus();
}

void RealMain(const Options& opts) {
  TF_CHECK_OK(CheckGroupFlags(opts.first))
      << "Can only specify one and ony one of --first_hlo_snapshot, "
         "--first_hlo_proto, --first_hlo_module_proto, --first_hlo_text";
  TF_CHECK_OK(CheckGroupFlags(opts.second))
      << "Can only specify one and ony one of --second_hlo_snapshot, "
         "--second_hlo_proto, --second_hlo_module_proto, --second_hlo_text";

  LOG(INFO) << "Loading first module";
  absl::StatusOr<std::unique_ptr<HloModule>> first_module =
      LoadHLOModule(opts.first);
  TF_CHECK_OK(first_module.status()) << "Failed to build first HLO module";
  LOG(INFO) << "Loaded first module";

  LOG(INFO) << "Loading second module";
  absl::StatusOr<std::unique_ptr<HloModule>> second_module =
      LoadHLOModule(opts.second);
  TF_CHECK_OK(second_module.status()) << "Failed to build second HLO module";
  LOG(INFO) << "Loaded second module";

  CHECK_OK(
      RunGumgraphDiff(*first_module.value(), *second_module.value(), opts));
}

}  // namespace
}  // namespace hlo_diff
}  // namespace xla

int main(int argc, char** argv) {
  xla::hlo_diff::Options opts;
  bool need_help = false;
  const std::vector<tsl::Flag> flag_list = {
      tsl::Flag("first_hlo_snapshot", &opts.first.hlo_snapshot,
                "first HloSnapshot proto to compare"),
      tsl::Flag("first_hlo_proto", &opts.first.hlo_proto,
                "first XLA hlo proto to compare"),
      tsl::Flag("first_hlo_module_proto", &opts.first.hlo_module_proto,
                "first XLA hlo module proto to compare"),
      tsl::Flag("first_hlo_text", &opts.first.hlo_text,
                "first XLA hlo text to compare"),
      tsl::Flag("second_hlo_snapshot", &opts.second.hlo_snapshot,
                "second HloSnapshot proto to compare"),
      tsl::Flag("second_hlo_proto", &opts.second.hlo_proto,
                "second XLA hlo proto to compare"),
      tsl::Flag("second_hlo_module_proto", &opts.second.hlo_module_proto,
                "second XLA hlo module proto to compare"),
      tsl::Flag("second_hlo_text", &opts.second.hlo_text,
                "second XLA hlo text to compare"),
      tsl::Flag("ignore_shape_during_instruction_matching",
                &opts.diff_options.fingerprint_options.ignore_shape,
                "Ignore array/tensor shapes when matching instructions"),
      tsl::Flag("text_output", &opts.render_options.text_output,
                "file to save diff blocks as text"),
      tsl::Flag("html_output", &opts.render_options.html_output,
                "file to save an overview of the diff result as html"),
      tsl::Flag("help", &need_help, "Prints this help message"),
  };

  std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(xla::hlo_diff::kUsage, &argc, &argv);
  LOG_IF(QFATAL, argc != 1 || !parse_ok || need_help) << usage;
  xla::hlo_diff::RealMain(opts);
  return 0;
}
