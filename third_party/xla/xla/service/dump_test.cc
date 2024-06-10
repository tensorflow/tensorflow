/* Copyright 2024 The OpenXLA Authors.

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
#include "xla/service/dump.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/xla.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

using ::testing::IsEmpty;

TEST(DumpTest, NoDumpingWhenNotEnabled) {
  std::string filename =
      tsl::io::JoinPath(tsl::testing::TmpDir(), "disable_override");
  std::string contents = "hello";
  DebugOptions options;
  options.set_xla_enable_dumping(false);
  options.set_xla_dump_to(filename);
  DumpToFileInDir(options, "disable_override", contents);

  std::vector<std::string> matches;
  ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(filename, &matches));
  EXPECT_THAT(matches, IsEmpty());
}

TEST(DumpTest, NoDumpingWhenDisabled) {
  std::string filename =
      tsl::io::JoinPath(tsl::testing::TmpDir(), "disable_override");
  std::string contents = "hello";
  DebugOptions options;
  options.set_xla_disable_dumping(true);
  options.set_xla_dump_to(filename);
  DumpToFileInDir(options, "disable_override", contents);

  std::vector<std::string> matches;
  ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(filename, &matches));
  EXPECT_THAT(matches, IsEmpty());
}

TEST(DumpTest, DisablingTakesPrecedence) {
  std::string filename =
      tsl::io::JoinPath(tsl::testing::TmpDir(), "disable_override");
  std::string contents = "hello";
  DebugOptions options;
  options.set_xla_enable_dumping(false);
  options.set_xla_disable_dumping(true);
  options.set_xla_dump_to(filename);
  DumpToFileInDir(options, "disable_override", contents);

  std::vector<std::string> matches;
  ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(filename, &matches));
  EXPECT_THAT(matches, IsEmpty());
}

TEST(DumpTest, DumpingWorks) {
  std::string filename =
      tsl::io::JoinPath(tsl::testing::TmpDir(), "enable_dumping");
  std::string contents = "hello";
  DebugOptions options;
  options.set_xla_dump_to(tsl::testing::TmpDir());
  options.set_xla_enable_dumping(true);
  DumpToFileInDir(options, "enable_dumping", contents);

  std::string real_contents;
  ASSERT_OK(
      tsl::ReadFileToString(tsl::Env::Default(), filename, &real_contents));
  EXPECT_EQ(contents, real_contents);
}

}  // namespace
}  // namespace xla
