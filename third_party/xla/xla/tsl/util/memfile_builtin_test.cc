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

#include "xla/tsl/util/memfile_builtin.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/util/file_toc.h"

namespace tsl::memfile {
namespace {

TEST(RegisterBuiltInFile, Works) {
  std::string data1 = "foo";
  FileToc toc1;
  toc1.name = "one";
  toc1.data = data1.data();
  toc1.size = data1.size();

  std::string data2 = "foo";
  FileToc toc2;
  toc2.name = "two";
  toc2.data = data2.data();
  toc2.size = data2.size();

  FileToc sentinel;
  sentinel.name = nullptr;
  std::vector<FileToc> toc = {
      toc1,
      toc2,
      sentinel,
  };

  ASSERT_TRUE(GlobalRegisterFiles("dir", toc.data()));

  std::string contents1;
  ASSERT_OK(
      ReadFileToString(tsl::Env::Default(), "embed://dir/one", &contents1));
  EXPECT_EQ(contents1, data1);

  std::string contents2;
  ASSERT_OK(
      ReadFileToString(tsl::Env::Default(), "embed://dir/two", &contents2));
  EXPECT_EQ(contents2, data2);
}

}  // namespace
}  // namespace tsl::memfile
