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

#include "fuzztest/fuzztest.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/stringpiece.h"

// This is a fuzzer for tensorflow::str_util::StringReplace

namespace {

void FuzzTest(bool all_flag, std::string s, std::string oldsub,
              std::string newsub) {
  tensorflow::StringPiece sp(s);
  tensorflow::StringPiece oldsubp(oldsub);
  tensorflow::StringPiece newsubp(newsub);

  std::string subbed =
      tensorflow::str_util::StringReplace(sp, oldsubp, newsubp, all_flag);
}
FUZZ_TEST(CC_FUZZING, FuzzTest)
    .WithDomains(/*all_flag=*/fuzztest::Arbitrary<bool>(),
                 /*s=*/fuzztest::Arbitrary<std::string>().WithSize(10),
                 /*oldsub=*/fuzztest::Arbitrary<std::string>().WithSize(5),
                 /*newsub=*/fuzztest::Arbitrary<std::string>());

}  // namespace
