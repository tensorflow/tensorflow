/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#include "absl/strings/string_view.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/file_toc.h"
#include "xla/tsl/util/filewrapper_testdata.h"
#include "tsl/platform/path.h"

using ::testing::EndsWith;

TEST(FilewrapperTest, CompareData) {
  constexpr absl::string_view filename = "util/filewrapper_testdata.txt";

  ASSERT_EQ(filewrapper_testdata_size(), 1);
  const FileToc* file_toc = filewrapper_testdata_create();
  EXPECT_THAT(file_toc[0].name, EndsWith(filename));
  EXPECT_EQ(nullptr, file_toc[1].name);

  std::string true_contents;
  TF_ASSERT_OK(tsl::ReadFileToString(
      tsl::Env::Default(),
      tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "tsl", filename),
      &true_contents));

  ASSERT_EQ(true_contents.size(), file_toc[0].size);
  EXPECT_EQ(true_contents,
            absl::string_view(file_toc[0].data, file_toc[0].size));
}

TEST(FilewrapperTest, CheckSourceContents) {
  std::string cc_contents;
  TF_ASSERT_OK(tsl::ReadFileToString(
      tsl::Env::Default(),
      tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "tsl", "util",
                        "filewrapper_testdata.cc"),
      &cc_contents));

  // Our source files should contain no embedded NULs. Other control characters
  // are OK in string literals, with certain restrictions, but we rely on the
  // compiler to flag those for us.
  EXPECT_EQ(std::string::npos, cc_contents.find('\0'));

  std::string header_contents;
  TF_ASSERT_OK(
      tsl::ReadFileToString(tsl::Env::Default(),
                            tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "tsl",
                                              "util", "filewrapper_testdata.h"),
                            &header_contents));

  EXPECT_EQ(std::string::npos, header_contents.find('\0'));
}
