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
#include <vector>

#include "fuzztest/fuzztest.h"
#include "tensorflow/core/platform/stringprintf.h"

// This is a fuzzer for tensorflow::strings::Printf

namespace {

void FuzzTest(const std::vector<std::string> ss) {
  const std::string all = ss[0] + ss[1] + ss[2];

  int n[4] = {-1, -1, -1, -1};
  const std::string ret =
      tensorflow::strings::Printf("%n%s%n%s%n%s%n", &n[0], ss[0].c_str(), &n[1],
                                  ss[1].c_str(), &n[2], ss[2].c_str(), &n[3]);

  int size_so_far = 0;
  for (int i = 0; i < 3; i++) {
    assert(n[i] >= 0);
    assert(n[i] <= size_so_far);
    size_so_far += ss[i].size();
  }

  assert(n[3] >= 0);
  assert(n[3] <= size_so_far);
  assert(ret.size() == n[3]);
}
FUZZ_TEST(CC_FUZZING, FuzzTest)
  .WithDomains(fuzztest::Arbitrary<std::vector<std::string>>().WithSize(3));

}  // namespace
