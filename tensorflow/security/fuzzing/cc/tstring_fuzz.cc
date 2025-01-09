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
#include <cassert>
#include <string>
#include <vector>

#include "fuzztest/fuzztest.h"
#include "tensorflow/core/platform/tstring.h"

// This is a fuzzer for tensorflow::tstring

namespace {

void FuzzTest(const std::vector<std::string>& ss) {
  tensorflow::tstring base = ss[0];

  for (int i = 1; i < ss.size(); ++i) {
    base.append(ss[i]);
    assert(base.size() <= base.capacity());
  }
}
FUZZ_TEST(CC_FUZZING, FuzzTest)
    .WithDomains(
        fuzztest::VectorOf(fuzztest::Arbitrary<std::string>().WithMaxSize(10))
            .WithMinSize(1));

}  // namespace
