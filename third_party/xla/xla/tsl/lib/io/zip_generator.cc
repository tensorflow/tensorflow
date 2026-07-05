/* Copyright 2026 The OpenXLA Authors.

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

// This is a test-only utility for validating that a zip file created with
// ZipWriter is valid.

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/tsl/lib/io/zip_writer.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "tsl/platform/init_main.h"

struct FileEntry {
  std::string name;
  std::string content;
};

absl::StatusOr<std::string> GenerateZip(absl::string_view output_path,
                                        const std::vector<FileEntry>& entries) {
  std::unique_ptr<tsl::WritableFile> file;
  RETURN_IF_ERROR(
      tsl::Env::Default()->NewWritableFile(std::string(output_path), &file));

  ASSIGN_OR_RETURN(tsl::io::ZipWriter writer,
                   tsl::io::ZipWriter::Create(std::move(file)));

  for (const auto& entry : entries) {
    RETURN_IF_ERROR(writer.AddFile(entry.name, entry.content));
  }

  RETURN_IF_ERROR(std::move(writer).Finish());
  return std::string(output_path);
}

int main(int argc, char** argv) {
  tsl::port::InitMain(argv[0], &argc, &argv);

  if (argc < 2) {
    std::cerr << "Usage: zip_generator <output_zip_path> [file1_name "
                 "content1]...\n";
    return 1;
  }
  std::string output_path = argv[1];

  std::vector<FileEntry> entries;
  entries.reserve((argc - 2) / 2);
  for (int i = 2; i < argc; i += 2) {
    if (i + 1 >= argc) {
      std::cerr << "Missing content for file: " << argv[i] << "\n";
      return 1;
    }
    entries.push_back({argv[i], argv[i + 1]});
  }

  auto result = GenerateZip(output_path, entries);
  if (!result.ok()) {
    std::cerr << "Failed to generate ZIP: " << result.status().ToString()
              << "\n";
    return 1;
  }

  std::cout << "Successfully generated ZIP at " << *result << "\n";
  return 0;
}
