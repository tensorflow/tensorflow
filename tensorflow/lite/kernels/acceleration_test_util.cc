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
#include "tensorflow/lite/kernels/acceleration_test_util.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstring>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>

#include "absl/types/optional.h"
#include "tensorflow/lite/kernels/acceleration_test_util_internal.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {

std::string GetCurrentTestId() {
  const ::testing::TestInfo* const test_info =
      ::testing::UnitTest::GetInstance()->current_test_info();

  std::stringstream test_id_stream;

  test_id_stream << test_info->test_suite_name() << "/" << test_info->name();

  return test_id_stream.str();
}

}  // namespace tflite
