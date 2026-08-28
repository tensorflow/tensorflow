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

#include <iostream>
#include <string>

#include "absl/status/status.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/SHA256.h"
#include "xla/tsl/platform/env.h"
#include "tsl/platform/init_main.h"

int main(int argc, char** argv) {
  tsl::port::InitMain(argv[0], &argc, &argv);
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>\n";
    return 1;
  }

  std::string file_content;
  absl::Status s =
      tsl::ReadFileToString(tsl::Env::Default(), argv[1], &file_content);
  if (!s.ok()) {
    std::cerr << "Failed to read input file " << argv[1] << ": " << s.message()
              << "\n";
    return 1;
  }

  llvm::SHA256 sha256;
  sha256.update(file_content);

  std::string hex_hash = llvm::toHex(sha256.final(), /*LowerCase=*/true);

  s = tsl::WriteStringToFile(tsl::Env::Default(), argv[2], hex_hash);
  if (!s.ok()) {
    std::cerr << "Failed to write output file " << argv[2] << ": "
              << s.message() << "\n";
    return 1;
  }

  return 0;
}
