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

#include "xla/service/riegeli_dump_writer.h"

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/testing/temporary_directory.h"
#include "xla/xla.pb.h"
#include "tsl/platform/path.h"

namespace xla {
namespace {

using ::testing::HasSubstr;
using ::testing::SizeIs;

TEST(RiegeliDumpWriterTest, CreateRiegeliDumpWriter) {
  TF_ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory dump_folder,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  DebugOptions debug_options = DefaultDebugOptionsIgnoringFlags();
  debug_options.set_xla_dump_to(dump_folder.path());
  debug_options.set_xla_enable_dumping(true);

  std::string filename = "test_dump";
  TF_ASSERT_OK_AND_ASSIGN(auto writer,
                          CreateRiegeliDumpWriter(debug_options, filename));

  writer->Write("hello world");
  EXPECT_TRUE(writer->Close());

  std::string file_path = tsl::io::JoinPath(dump_folder.path(), filename);
  std::string content;
  TF_ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), file_path, &content));
  EXPECT_THAT(content, HasSubstr("hello world"));
}

TEST(RiegeliDumpWriterTest, CreateRiegeliDumpWriterWithModule) {
  TF_ASSERT_OK_AND_ASSIGN(
      tsl::testing::TemporaryDirectory dump_folder,
      tsl::testing::TemporaryDirectory::CreateForCurrentTestcase());
  const HloModule hlo_module("test_module", HloModuleConfig());
  DebugOptions debug_options = DefaultDebugOptionsIgnoringFlags();
  debug_options.set_xla_dump_to(dump_folder.path());
  debug_options.set_xla_enable_dumping(true);

  std::string filename = "module_dump";
  TF_ASSERT_OK_AND_ASSIGN(
      auto writer,
      CreateRiegeliDumpWriter(debug_options, filename, &hlo_module));

  writer->Write("hello world");
  EXPECT_TRUE(writer->Close());

  std::vector<std::string> matches;
  TF_ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(
      tsl::io::JoinPath(dump_folder.path(), "*module_dump*"), &matches));
  EXPECT_THAT(matches, SizeIs(1));
  EXPECT_THAT(matches[0], HasSubstr("test_module"));

  std::string content;
  TF_ASSERT_OK(
      tsl::ReadFileToString(tsl::Env::Default(), matches[0], &content));
  EXPECT_THAT(content, HasSubstr("hello world"));
}

}  // namespace
}  // namespace xla
