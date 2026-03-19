/* Copyright 2024 The OpenXLA Authors.

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

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

int main(int argc, char** argv) {
  if (argc == 1) {
    return 0;
  }
  if (argc > 2) {
    return -1;
  }

  const auto process_was_called_as = [&](absl::string_view binary_name) {
    return argv[0] == binary_name ||
           absl::EndsWith(argv[0], absl::StrCat("/", binary_name));
  };

  if (process_was_called_as("ptxas") &&
      argv[1] == absl::string_view{"--version"}) {
    std::cout << "ptxas dummy V111.2.3\n";
    return 0;
  }

  if (process_was_called_as("nvlink") &&
      argv[1] == absl::string_view{"--version"}) {
    std::cout << "nvlink dummy V444.5.6\n";
    return 0;
  }

  if (process_was_called_as("fatbinary") &&
      argv[1] == absl::string_view{"--version"}) {
    std::cout << "fatbinary dummy V777.8.9\n";
    return 0;
  }

  if (process_was_called_as("nvdisasm") &&
      argv[1] == absl::string_view{"--version"}) {
    std::cout << "nvdisasm dummy V999.1.2\n";
    return 0;
  }

  return -2;
}
