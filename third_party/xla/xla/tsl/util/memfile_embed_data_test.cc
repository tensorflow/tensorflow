/* Copyright 2026 The OpenXLA Authors.

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

#include <string>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"

namespace tsl {
namespace {

TEST(MemfileEmbedData, DataLinesUp) {
  std::string memfile_contents;
  ASSERT_OK(ReadFileToString(Env::Default(),
                             "embed://test_memfile/memfile_test.txt",
                             &memfile_contents));

  std::string resource_contents;
  std::string resource_path =
      io::JoinPath(testing::XlaSrcRoot(), "tsl", "util", "memfile_test.txt");
  ASSERT_OK(
      ReadFileToString(Env::Default(), resource_path, &resource_contents));
  EXPECT_EQ(memfile_contents, resource_contents);
}

}  // namespace
}  // namespace tsl
