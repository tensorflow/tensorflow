/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/run_handler_thread_pool/run_handler_util.h"

#include <cstdlib>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/ascii.h"
#include "absl/strings/numbers.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/str_util.h"

namespace tfrt {
namespace tf {

double ParamFromEnvWithDefault(const char* var_name, double default_value) {
  const char* val = std::getenv(var_name);
  double num;
  return (val && absl::SimpleAtod(val, &num)) ? num : default_value;
}

std::vector<double> ParamFromEnvWithDefault(const char* var_name,
                                            std::vector<double> default_value) {
  const char* val = std::getenv(var_name);
  if (!val) {
    return default_value;
  }
  std::vector<std::string> splits = tensorflow::str_util::Split(val, ",");
  std::vector<double> result;
  result.reserve(splits.size());
  for (auto& split : splits) {
    double num;
    if (absl::SimpleAtod(split, &num)) {
      result.push_back(num);
    } else {
      LOG(ERROR) << "Wrong format for " << var_name << ". Use default value.";
      return default_value;
    }
  }
  return result;
}

std::vector<int> ParamFromEnvWithDefault(const char* var_name,
                                         std::vector<int> default_value) {
  const char* val = std::getenv(var_name);
  if (!val) {
    return default_value;
  }
  std::vector<std::string> splits = tensorflow::str_util::Split(val, ",");
  std::vector<int> result;
  result.reserve(splits.size());
  for (auto& split : splits) {
    int num;
    if (absl::SimpleAtoi(split, &num)) {
      result.push_back(num);
    } else {
      LOG(ERROR) << "Wrong format for " << var_name << ". Use default value.";
      return default_value;
    }
  }
  return result;
}

bool ParamFromEnvBoolWithDefault(const char* var_name, bool default_value) {
  const char* val = std::getenv(var_name);
  return (val) ? absl::AsciiStrToLower(val) == "true" : default_value;
}

}  // namespace tf
}  // namespace tfrt
