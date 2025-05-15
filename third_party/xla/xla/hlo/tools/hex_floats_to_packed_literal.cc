/* Copyright 2017 The OpenXLA Authors.

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

#include <memory>
#include <string>
#include <vector>

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/io/buffered_inputstream.h"
#include "xla/tsl/lib/io/random_inputstream.h"
#include "xla/tsl/util/command_line_flags.h"
#include "tsl/platform/env.h"
#include "tsl/platform/file_system.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"

using std::string;

int main(int argc, char** argv) {
  // Flags
  std::string input_file = "";
  std::string output_file = "";
  const std::vector<tsl::Flag> flag_list = {
      tsl::Flag("input_file", &input_file, "file to convert"),
      tsl::Flag("output_file", &output_file, "converted file"),
  };
  std::string usage = tsl::Flags::Usage(argv[0], flag_list);
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(argv[0], &argc, &argv);
  if (argc != 1 || !parse_ok) {
    LOG(QFATAL) << usage;
  }

  if (input_file.empty()) {
    LOG(QFATAL) << "--input_file is required";
  }
  if (output_file.empty()) {
    LOG(QFATAL) << "--output_file is required";
  }

  std::unique_ptr<tsl::RandomAccessFile> file;
  TF_CHECK_OK(tsl::Env::Default()->NewRandomAccessFile(input_file, &file));

  std::vector<float> floats;
  std::string line;
  tsl::io::RandomAccessInputStream stream(file.get());
  tsl::io::BufferedInputStream buf(&stream, 1048576);
  while (buf.ReadLine(&line).ok()) {
    float value;
    QCHECK(sscanf(line.c_str(), "%f", &value) != 1)
        << "invalid float value: " << line;
    floats.push_back(value);
  }

  absl::string_view content(absl::bit_cast<const char*>(floats.data()),
                            floats.size() * sizeof(float));
  TF_CHECK_OK(
      tsl::WriteStringToFile(tsl::Env::Default(), output_file, content));
  return 0;
}
