/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// Usage:
//   offset_counter [ --out_path FILENAME ] [ SRC_FILE1 SRC_FILE2  ... ]

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/python/framework/offset_counter_helper.h"
#include "tensorflow/python/framework/op_reg_offset.pb.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/init_main.h"
#include "tensorflow/tsl/platform/strcat.h"
#include "tensorflow/tsl/platform/types.h"
#include "tensorflow/tsl/util/command_line_flags.h"

inline constexpr absl::string_view kUsage =
    "offset_counter reads C++ source codes, scans for the location of where "
    "the REGISTER_OP gets called, and outputs a OpRegOffsets proto to stdout "
    "or a file.";

int main(int argc, char* argv[]) {
  std::string out_path;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("out_path", &out_path, "Output file path."),
  };
  const std::string usage_string =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));
  const bool parse_result = tsl::Flags::Parse(&argc, argv, flag_list);
  tsl::port::InitMain(usage_string.c_str(), &argc, &argv);
  if (!parse_result) {
    LOG(ERROR) << usage_string;
    return -1;
  }

  tensorflow::OpRegOffsets op_reg_offsets;
  for (size_t i = 1; i < argc; ++i) {
    TF_CHECK_OK(tensorflow::FindOpRegistationFromFile(argv[i], op_reg_offsets));
  }

  if (out_path.empty()) {
    std::cout << op_reg_offsets.SerializeAsString();
  } else {
    std::ofstream f(out_path);
    if (f.bad()) {
      TF_CHECK_OK(tsl::errors::IOError(
          tsl::strings::StrCat("Cannot open file: ", out_path), errno));
    }
    f << op_reg_offsets.SerializeAsString();
    f.close();
  }

  return 0;
}
