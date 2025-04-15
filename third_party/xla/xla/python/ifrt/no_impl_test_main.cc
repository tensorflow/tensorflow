/* Copyright 2022 The OpenXLA Authors.

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

#include <gtest/gtest.h>

int main(int argc, char** argv) {
  // Skip all tests. This is to verify that implementation tests build
  // successfully without registering an IFRT client factory.
  //
  // Actual implementation tests may link with the standard `gtest_main` to run
  // all tests or define a custom `main` function to filter out some tests.
  const char* kFilter = "-*";
#ifdef GTEST_FLAG_SET
  GTEST_FLAG_SET(filter, kFilter);
#else
  testing::GTEST_FLAG(filter) = kFilter;
#endif

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
