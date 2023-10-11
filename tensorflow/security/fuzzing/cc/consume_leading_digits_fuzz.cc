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
#include <cstdlib>
#include <string>
#include <string_view>

#include "fuzztest/fuzztest.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/stringpiece.h"

// This is a fuzzer for tensorflow::str_util::ConsumeLeadingDigits

namespace {

void FuzzTest(std::string data) {
  tensorflow::StringPiece sp(data);
  tensorflow::uint64 val;

  const bool leading_digits =
      tensorflow::str_util::ConsumeLeadingDigits(&sp, &val);
  const char lead_char_consume_digits = *(sp.data());
  if (leading_digits) {
    if (lead_char_consume_digits >= '0') {
      assert(lead_char_consume_digits > '9');
    }
    assert(val >= 0);
  }
}
FUZZ_TEST(CC_FUZZING, FuzzTest)
    .WithDomains(
        fuzztest::Arbitrary<std::string>().WithMaxSize(25));
}  // namespace
