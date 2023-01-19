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

#include "tensorflow/python/framework/offset_counter_helper.h"

#include <cstdint>
#include <fstream>
#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/regexp.h"
#include "tensorflow/tsl/platform/strcat.h"

namespace tensorflow {

tsl::Status FindOpRegistationFromFile(absl::string_view filename,
                                      OpRegOffsets& op_reg_offsets) {
  static constexpr LazyRE2 reg_pattern = {
      R"regex((REGISTER_OP)\("([\w>]+)"\))regex"};
  std::ifstream f(std::string{filename});
  if (f.bad()) {
    return tsl::errors::IOError(
        tsl::strings::StrCat("Cannot open file: ", filename), errno);
  }
  std::string line;
  absl::string_view reg_keyword, op_name;
  uint32_t offsets = 0;
  while (std::getline(f, line)) {
    if (RE2::PartialMatch(line, *reg_pattern, &reg_keyword, &op_name)) {
      // Set the [start, end] to the beginning char of REGISTER_OP
      // See phase 1 in go/pywald-tf-ops-xref for more details.
      uint32_t offset_start = offsets + (line.data() - reg_keyword.data());
      uint32_t offset_end = offset_start;
      auto op_reg_offset = op_reg_offsets.add_offsets();
      op_reg_offset->set_name(std::string{op_name});
      op_reg_offset->set_filepath(std::string{filename});
      op_reg_offset->set_start(offset_start);
      op_reg_offset->set_end(offset_end);
    }
    offsets += line.size() + 1;
  }
  f.close();
  return tsl::OkStatus();
}

}  // namespace tensorflow
