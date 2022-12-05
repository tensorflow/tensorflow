/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdint>
#include <string>

#include "fuzztest/fuzztest.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/security/fuzzing/cc/fuzz_helpers.h"

// This is a fuzzer for `tensorflow::StatusGroup`. Since `Status` is used almost
// everywhere, we need to ensure that the common functionality is safe. We don't
// expect many crashes from this fuzzer

namespace {

void FuzzTest(uint32_t code, bool is_derived) {
  const std::string error_message = "ERROR";
  tensorflow::StatusGroup sg;
  tensorflow::error::Code error_code = helper::BuildRandomErrorCode(code);
  tensorflow::Status s = tensorflow::Status(error_code, error_message);

  if (is_derived) {
    tensorflow::Status derived_s = tensorflow::StatusGroup::MakeDerived(s);
    sg.Update(derived_s);
  } else {
    sg.Update(s);
  }

  // Ignore warnings that these values are unused
  sg.as_summary_status().IgnoreError();
  sg.as_concatenated_status().IgnoreError();
  sg.AttachLogMessages();
}
FUZZ_TEST(CC_FUZZING, FuzzTest);

}  // namespace
