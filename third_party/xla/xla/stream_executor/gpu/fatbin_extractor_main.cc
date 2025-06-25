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

// Iterates over the given list of files and extracts the fatbin section.

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xla/stream_executor/gpu/elf_section_extraction.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/platform/init_main.h"

ABSL_FLAG(std::string, output_cc_file, "",
          "A filepath where the C++ implementation file will be written to");
ABSL_FLAG(std::string, output_header_file, "",
          "A filepath where the C++ header file will be written to");
ABSL_FLAG(std::string, section_name, ".nv_fatbin",
          "The ELF section to extract");
ABSL_FLAG(std::string, cpp_namespace, "stream_executor::cuda",
          "The C++ namespace where the symbol should be placed in");
ABSL_FLAG(std::string, cpp_identifier, "kKernel", "The name of the constant");

static constexpr absl::string_view kHeaderTemplate = R"cpp(
#pragma once

#include <cstdint>

#include "absl/types/span.h"

  namespace $0 {
  extern const absl::Span<const uint8_t> $1;
  }  // namespace $0
)cpp";

static constexpr absl::string_view kCcFilePreamble = R"cpp(
#include <cstdint>

#include "absl/types/span.h"

  namespace $0 {
  const uint8_t kData[] =
)cpp";

static constexpr absl::string_view kCcFilePostamble = R"cpp(
  // We have to remove the null terminator from the span.
  extern const absl::Span<const uint8_t> $1 =
      absl::Span<const uint8_t>(kData, sizeof(kData) - 1);
  }  // namespace $0
)cpp";

absl::Status ProcessFile(absl::string_view filename) {
  TF_RETURN_IF_ERROR(tsl::WriteStringToFile(
      tsl::Env::Default(), absl::GetFlag(FLAGS_output_header_file),
      absl::Substitute(kHeaderTemplate, absl::GetFlag(FLAGS_cpp_namespace),
                       absl::GetFlag(FLAGS_cpp_identifier))));

  tsl::Env* env = tsl::Env::Default();

  std::string file_contents;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(env, std::string(filename), &file_contents));
  TF_ASSIGN_OR_RETURN(std::string contents,
                      stream_executor::gpu::GetSectionContentsFromLibraryDump(
                          file_contents, absl::GetFlag(FLAGS_section_name)));

  std::unique_ptr<tsl::WritableFile> cc_file;
  TF_RETURN_IF_ERROR(tsl::Env::Default()->NewWritableFile(
      absl::GetFlag(FLAGS_output_cc_file), &cc_file));

  TF_RETURN_IF_ERROR(cc_file->Append(
      absl::Substitute(kCcFilePreamble, absl::GetFlag(FLAGS_cpp_namespace),
                       absl::GetFlag(FLAGS_cpp_identifier))));

  absl::string_view remaining_bytes = contents;

  while (!remaining_bytes.empty()) {
    constexpr size_t kMaxLengthPerLine =
        60;  // Since hex encoding is rather verbose we limit the length of each
             // line to 60 characters as that can still be easily viewed in an
             // editor.
    absl::string_view current_line =
        remaining_bytes.substr(0, kMaxLengthPerLine);
    remaining_bytes = remaining_bytes.substr(
        std::min(kMaxLengthPerLine, remaining_bytes.size()));
    bool is_last_line = remaining_bytes.empty();
    TF_RETURN_IF_ERROR(
        cc_file->Append(absl::StrCat("  \"", absl::CHexEscape(current_line),
                                     "\"", is_last_line ? ";\n" : "\n")));
  }

  TF_RETURN_IF_ERROR(cc_file->Append(
      absl::Substitute(kCcFilePostamble, absl::GetFlag(FLAGS_cpp_namespace),
                       absl::GetFlag(FLAGS_cpp_identifier))));

  return absl::OkStatus();
}

int main(int argc, char* argv[]) {
  tsl::port::InitMain(
      "Usage: fatbin_extractor_main --output_cc_file=out.cc "
      "--output_header_file=out.h --section_name=.nv_fatbin "
      "--cpp_namespace=stream_executor::cuda "
      "--cpp_identifier=kFatbinMyName <file>...",
      &argc, &argv);

  std::vector<char*> remaining_command_line_options =
      absl::ParseCommandLine(argc, argv);
  if (absl::GetFlag(FLAGS_output_cc_file).empty()) {
    LOG(FATAL) << "`--output_cc_file` needs to provided.";
  }

  if (absl::GetFlag(FLAGS_output_header_file).empty()) {
    LOG(FATAL) << "`--output_header_file` needs to provided.";
  }

  absl::Status last_result;

  auto positional_arguments =
      absl::Span<char* const>{remaining_command_line_options}.subspan(1);

  // We iterate over all the
  for (char* positional_argument : positional_arguments) {
    last_result = ProcessFile(positional_argument);
    if (last_result.ok()) {
      break;
    }
  }

  if (!last_result.ok()) {
    LOG(FATAL) << last_result;
  }

  return 0;
}
