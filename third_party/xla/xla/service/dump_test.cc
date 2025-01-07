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

#include <memory>
#include <string>
#include <vector>

#include "absl/strings/match.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

using ::testing::IsEmpty;

TEST(DumpHloIfEnabled, LargeConstantElided) {
  HloModuleConfig config;
  DebugOptions options = config.debug_options();
  auto env = tsl::Env::Default();
  std::string dump_dir;
  EXPECT_TRUE(env->LocalTempFilename(&dump_dir));
  options.set_xla_dump_to(dump_dir);
  options.set_xla_dump_hlo_as_text(true);
  options.set_xla_dump_large_constants(false);
  config.set_debug_options(options);
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = s32[11] parameter(0)
      c = s32[11] constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
      ROOT x = s32[11] multiply(p0, c)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m,
                          ParseAndReturnUnverifiedModule(kModuleStr, config));
  std::string dump_name = "dump";
  auto paths = DumpHloModuleIfEnabled(*m, dump_name);
  EXPECT_EQ(paths.size(), 1);
  std::string data;
  EXPECT_TRUE(ReadFileToString(env, paths[0], &data).ok());
  EXPECT_TRUE(absl::StrContains(data, "{...}"));
}

TEST(DumpHloIfEnabled, LargeConstantPrinted) {
  HloModuleConfig config;
  DebugOptions options = config.debug_options();
  auto env = tsl::Env::Default();
  std::string dump_dir;
  EXPECT_TRUE(env->LocalTempFilename(&dump_dir));
  options.set_xla_dump_to(dump_dir);
  options.set_xla_dump_hlo_as_text(true);
  options.set_xla_dump_large_constants(true);
  config.set_debug_options(options);
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = s32[11] parameter(0)
      c = s32[11] constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
      ROOT x = s32[11] multiply(p0, c)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m,
                          ParseAndReturnUnverifiedModule(kModuleStr, config));
  std::string dump_name = "dump";
  auto paths = DumpHloModuleIfEnabled(*m, dump_name);
  EXPECT_EQ(paths.size(), 1);
  std::string data;
  EXPECT_TRUE(ReadFileToString(env, paths[0], &data).ok());
  EXPECT_TRUE(!absl::StrContains(data, "{...}"));
}

TEST(DumpTest, NoDumpingToFileWhenNotEnabled) {
  std::string filename =
      tsl::io::JoinPath(tsl::testing::TmpDir(), "disable_override");
  std::string contents = "hello";
  DebugOptions options;
  options.set_xla_enable_dumping(false);
  options.set_xla_dump_to(filename);
  DumpToFileInDir(options, "disable_override", contents);

  std::vector<std::string> matches;
  TF_ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(filename, &matches));
  EXPECT_THAT(matches, IsEmpty());
}

TEST(DumpTest, DumpingToFileWorksWhenEnabled) {
  std::string filename =
      tsl::io::JoinPath(tsl::testing::TmpDir(), "enable_dumping");
  std::string contents = "hello";
  DebugOptions options;
  options.set_xla_dump_to(tsl::testing::TmpDir());
  options.set_xla_enable_dumping(true);
  DumpToFileInDir(options, "enable_dumping", contents);

  std::string real_contents;
  TF_ASSERT_OK(
      tsl::ReadFileToString(tsl::Env::Default(), filename, &real_contents));
  EXPECT_EQ(contents, real_contents);
}

TEST(DumpTest, DumpProtobufToFileWhenEnabled) {
  HloModuleProto module;
  module.set_name("hello");
  std::string filename =
      tsl::io::JoinPath(tsl::testing::TmpDir(), "enable_proto_dumping.txt");

  DebugOptions options;
  options.set_xla_dump_to(tsl::testing::TmpDir());
  options.set_xla_enable_dumping(true);
  DumpProtobufToFile(module, options, "enable_proto_dumping");

  HloModuleProto mod;
  TF_ASSERT_OK(tsl::ReadTextProto(tsl::Env::Default(), filename, &mod));
  EXPECT_EQ(mod.name(), module.name());
}

TEST(DumpTest, DumpProtobufToFileWhenDisabled) {
  HloModuleProto module;
  module.set_name("hello");
  std::string filename =
      tsl::io::JoinPath(tsl::testing::TmpDir(), "disable_proto_dumping.txt");

  DebugOptions options;
  options.set_xla_dump_to(tsl::testing::TmpDir());
  options.set_xla_enable_dumping(false);
  DumpProtobufToFile(module, options, "disable_proto_dumping");

  std::vector<std::string> matches;
  TF_ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(filename, &matches));
  EXPECT_THAT(matches, IsEmpty());
}

TEST(DumpTest, DumpFdoProfileToFileWhenEnabled) {
  std::string fdo_profile = "fdo_profile";
  HloModuleConfig config;
  config.set_fdo_profile(fdo_profile);
  DebugOptions options = config.debug_options();
  auto env = tsl::Env::Default();
  std::string dump_dir;
  ASSERT_TRUE(env->LocalTempFilename(&dump_dir));
  options.set_xla_dump_to(dump_dir);
  options.set_xla_gpu_experimental_dump_fdo_profiles(true);
  config.set_debug_options(options);
  const char* kModuleStr = R"(
    HloModule m
    test {
      p0 = s32[11] parameter(0)
      c = s32[11] constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
      ROOT x = s32[11] multiply(p0, c)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto m,
                          ParseAndReturnUnverifiedModule(kModuleStr, config));
  std::string dump_name = "dump";
  auto paths = DumpHloModuleIfEnabled(*m, dump_name);
  EXPECT_EQ(paths.size(), 2);

  std::string data;
  EXPECT_TRUE(ReadFileToString(env, paths[1], &data).ok());
  EXPECT_TRUE(absl::StrContains(data, fdo_profile));
}

TEST(DumpTest, DumpHloUnoptimizedSnapshot) {
  HloUnoptimizedSnapshot hlo_snapshot;
  HloModuleProto module;
  module.set_name("hello");
  *hlo_snapshot.mutable_hlo_module() = module;
  *hlo_snapshot.add_partitions() = HloInputs();

  HloModuleConfig config;
  DebugOptions options = config.debug_options();

  options.set_xla_dump_to(tsl::testing::TmpDir());
  options.set_xla_dump_hlo_as_text(true);
  options.set_xla_gpu_dump_hlo_unoptimized_snapshots(true);
  config.set_debug_options(options);

  DumpHloUnoptimizedSnapshotIfEnabled(hlo_snapshot, options);

  std::vector<std::string> matches;
  std::string pattern_filename =
      tsl::io::JoinPath(tsl::testing::TmpDir(), "*hlo_unoptimized_snapshot*");
  TF_ASSERT_OK(
      tsl::Env::Default()->GetMatchingPaths(pattern_filename, &matches));
  EXPECT_THAT(matches, Not(IsEmpty()));

  HloUnoptimizedSnapshot hlo_snapshot_loaded;
  TF_ASSERT_OK(tsl::ReadTextProto(tsl::Env::Default(), matches.front(),
                                  &hlo_snapshot_loaded));
  EXPECT_EQ(hlo_snapshot_loaded.hlo_module().name(), module.name());
}
}  // namespace
}  // namespace xla
