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
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/init_main.h"

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

absl::Status ProcessFile(absl::string_view filename,
                         absl::string_view output_header_file,
                         absl::string_view cpp_namespace,
                         absl::string_view cpp_identifier,
                         absl::string_view output_cc_file,
                         absl::string_view section_name) {
  TF_RETURN_IF_ERROR(tsl::WriteStringToFile(
      tsl::Env::Default(), std::string(output_header_file),
      absl::Substitute(kHeaderTemplate, cpp_namespace, cpp_identifier)));

  tsl::Env* env = tsl::Env::Default();

  std::string file_contents;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(env, std::string(filename), &file_contents));
  TF_ASSIGN_OR_RETURN(std::string contents,
                      stream_executor::gpu::GetSectionContentsFromLibraryDump(
                          file_contents, section_name));

  std::unique_ptr<tsl::WritableFile> cc_file;
  TF_RETURN_IF_ERROR(tsl::Env::Default()->NewWritableFile(
      std::string(output_cc_file), &cc_file));

  TF_RETURN_IF_ERROR(cc_file->Append(
      absl::Substitute(kCcFilePreamble, cpp_namespace, cpp_identifier)));

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
      absl::Substitute(kCcFilePostamble, cpp_namespace, cpp_identifier)));

  return absl::OkStatus();
}

int main(int argc, char* argv[]) {
  std::string output_cc_file;
  std::string output_header_file;
  std::string section_name;
  std::string cpp_namespace;
  std::string cpp_identifier;
  std::vector<tsl::Flag> flags = {
      tsl::Flag{"output_cc_file", &output_cc_file,
                "A filepath where the C++ implementation file "
                "will be written to"},
      tsl::Flag{"output_header_file", &output_header_file,
                "A filepath where the C++ header file "
                "will be written to"},
      tsl::Flag{"section_name", &section_name, "The ELF section to extract"},
      tsl::Flag{"cpp_namespace", &cpp_namespace,
                "The C++ namespace where the symbol should be placed in"},
      tsl::Flag{"cpp_identifier", &cpp_identifier, "The name of the constant"}};

  bool parse_ok = tsl::Flags::Parse(&argc, argv, flags);

  constexpr absl::string_view kUsage =
      "This tool lets you extract an arbitrary section from an ELF file and "
      "generate a C++ compilation unit where the section is embedded as a "
      "constant.";

  std::string usage =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[0], flags));

  tsl::port::InitMain(usage.c_str(), &argc, &argv);

  if (!parse_ok) {
    LOG(QFATAL) << usage;
  }

  if (output_cc_file.empty()) {
    LOG(QFATAL) << "`--output_cc_file` needs to provided.";
  }

  if (output_header_file.empty()) {
    LOG(QFATAL) << "`--output_header_file` needs to provided.";
  }

  absl::Status last_result;

  // subspan(1) removes the first argument (the binary name) from the span.
  auto positional_arguments = absl::MakeConstSpan(argv, argc).subspan(1);

  // We iterate over all the positional arguments (the files to extract) and
  // process them one by one until we find the first file that is successfully
  // processed. This is a rather peculiar behaviour but it is exactly what we
  // need to integrate nicely with Bazel.
  for (char* positional_argument : positional_arguments) {
    last_result =
        ProcessFile(positional_argument, output_header_file, cpp_namespace,
                    cpp_identifier, output_cc_file, section_name);
    if (last_result.ok()) {
      break;
    }
  }

  if (!last_result.ok()) {
    LOG(QFATAL) << last_result;
  }

  return 0;
}
