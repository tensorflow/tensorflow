/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/acceleration_test_util_internal.h"

#include <ctype.h>

#include <algorithm>
#include <functional>
#include <iterator>
#include <sstream>
#include <string>

namespace tflite {

void ReadAccelerationConfig(
    const char* config,
    const std::function<void(std::string, std::string, bool)>& consumer) {
  if (config) {
    std::istringstream istream{config};

    std::string curr_config_line;
    while (std::getline(istream, curr_config_line)) {
      // trim whitespaces
      curr_config_line.erase(
          curr_config_line.begin(),
          std::find_if_not(curr_config_line.begin(), curr_config_line.end(),
                           [](int ch) { return std::isspace(ch); }));
      // skipping comments and empty lines.
      if (curr_config_line.empty() || curr_config_line.at(0) == '#') {
        continue;
      }

      // split in test id regexp and rest of the config.
      auto first_sep_pos =
          std::find(curr_config_line.begin(), curr_config_line.end(), ',');

      bool is_denylist = false;
      std::string key = curr_config_line;
      std::string value{};
      if (first_sep_pos != curr_config_line.end()) {
        key = std::string(curr_config_line.begin(), first_sep_pos);
        value = std::string(first_sep_pos + 1, curr_config_line.end());
      }

      // Regexps starting with '-'' are denylist ones.
      if (key[0] == '-') {
        key = key.substr(1);
        is_denylist = true;
      }

      consumer(key, value, is_denylist);
    }
  }
}

}  // namespace tflite
