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

#include <sys/types.h>

#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/runtime/large_hlo_snapshot_serialization/serialization.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"
#include "tsl/platform/path.h"
#include "tsl/platform/platform.h"
#include "tsl/platform/protobuf.h"

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

TEST(DumpHloIfEnabled, DumpsBuildClNumber) {
  // BuildData isn't available in OSS.
  if (tsl::kIsOpenSource) {
    GTEST_SKIP() << "BuildData isn't available in OSS.";
  }

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

  EXPECT_TRUE(absl::StrContains(paths[0], ".cl_"));
}

TEST(DumpTest, DumpHloUnoptimizedSnapshotProtoBinary) {
  HloUnoptimizedSnapshot hlo_snapshot;
  HloModuleProto module;
  module.set_name("hello");
  *hlo_snapshot.mutable_hlo_module() = module;
  *hlo_snapshot.add_partitions() = HloInputs();

  HloModuleConfig config;
  DebugOptions options = config.debug_options();

  auto env = tsl::Env::Default();
  std::string dump_dir;
  EXPECT_TRUE(env->LocalTempFilename(&dump_dir));
  options.set_xla_dump_to(dump_dir);
  options.set_xla_dump_hlo_as_proto(true);
  options.set_xla_gpu_dump_hlo_unoptimized_snapshots(true);
  config.set_debug_options(options);

  DumpHloUnoptimizedSnapshotIfEnabled(hlo_snapshot, options);

  std::vector<std::string> matches;
  std::string pattern_filename =
      tsl::io::JoinPath(dump_dir, "*hlo_unoptimized_snapshot*");
  TF_ASSERT_OK(
      tsl::Env::Default()->GetMatchingPaths(pattern_filename, &matches));
  EXPECT_THAT(matches, Not(IsEmpty()));

  std::string file_contents;
  TF_ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), matches.front(),
                                     &file_contents));
  tsl::protobuf::io::ArrayInputStream input_stream(file_contents.data(),
                                                   file_contents.size());
  TF_ASSERT_OK_AND_ASSIGN(HloUnoptimizedSnapshot hlo_snapshot_loaded,
                          DeserializeHloUnoptimizedSnapshot(&input_stream));
  EXPECT_EQ(hlo_snapshot_loaded.hlo_module().name(), module.name());
}

TEST(DumpTest, GetNonDefaultDebugOptions) {
  DebugOptions options;
  DebugOptions default_options = DefaultDebugOptionsIgnoringFlags();
  std::string dump_folder = tsl::testing::TmpDir();

  // String field
  options.set_xla_dump_to(dump_folder);
  // Int32 field
  options.set_xla_gpu_dot_merger_threshold_mb(
      default_options.xla_gpu_dot_merger_threshold_mb() + 100);
  // Int64 field
  options.set_xla_gpu_experimental_collective_cse_distance_threshold(
      default_options.xla_gpu_experimental_collective_cse_distance_threshold() +
      100);
  // Bool field
  options.set_xla_gpu_enable_nccl_user_buffers(
      !default_options.xla_gpu_enable_nccl_user_buffers());
  options.set_xla_enable_dumping(true);
  options.set_xla_gpu_enable_shared_constants(false);
  // Enum field
  options.clear_xla_gpu_enable_command_buffer();
  options.add_xla_gpu_enable_command_buffer(DebugOptions::CUBLAS);
  options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
  // Message field
  int gpus_per_node;
  EXPECT_TRUE(absl::SimpleAtoi(
      default_options.xla_gpu_analytical_latency_estimator_options().at(
          "gpus_per_node"),
      &gpus_per_node));
  int chunk_size_bytes;
  EXPECT_TRUE(absl::SimpleAtoi(
      default_options.xla_gpu_analytical_latency_estimator_options().at(
          "chunk_size_bytes"),
      &chunk_size_bytes));
  options.mutable_xla_gpu_analytical_latency_estimator_options()->insert(
      {"gpus_per_node", std::to_string(gpus_per_node + 1)});
  options.mutable_xla_gpu_analytical_latency_estimator_options()->insert(
      {"chunk_size_bytes", std::to_string(chunk_size_bytes)});

  auto non_default_options = GetNonDefaultDebugOptions(options);
  EXPECT_THAT(non_default_options,
              testing::HasSubstr("xla_dump_to: \"" + dump_folder + "\""));
  EXPECT_THAT(
      non_default_options,
      testing::HasSubstr(
          "xla_gpu_dot_merger_threshold_mb: " +
          std::to_string(default_options.xla_gpu_dot_merger_threshold_mb() +
                         100)));
  EXPECT_THAT(
      non_default_options,
      testing::HasSubstr(
          "xla_gpu_experimental_collective_cse_distance_threshold: " +
          std::to_string(
              default_options
                  .xla_gpu_experimental_collective_cse_distance_threshold() +
              100)));
  EXPECT_THAT(non_default_options,
              testing::HasSubstr("xla_gpu_enable_nccl_user_buffers: true"));
  EXPECT_THAT(non_default_options,
              testing::HasSubstr("xla_gpu_enable_shared_constants: false"));
  EXPECT_THAT(non_default_options,
              testing::HasSubstr("xla_gpu_enable_command_buffer: CUBLAS"));
  EXPECT_THAT(non_default_options,
              testing::HasSubstr("xla_gpu_enable_command_buffer: FUSION"));
  EXPECT_THAT(
      non_default_options,
      testing::HasSubstr("xla_gpu_analytical_latency_estimator_options: {\n"
                         "  key: \"gpus_per_node\"\n"
                         "  value: \"" +
                         std::to_string(gpus_per_node + 1) +
                         "\"\n"
                         "}"));
  EXPECT_THAT(
      non_default_options,
      testing::HasSubstr("xla_gpu_analytical_latency_estimator_options: {\n"
                         "  key: \"chunk_size_bytes\"\n"
                         "  value: \"" +
                         std::to_string(chunk_size_bytes) +
                         "\"\n"
                         "}"));
  tsl::protobuf::TextFormat::Parser parser;
  DebugOptions parsed_options = DefaultDebugOptionsIgnoringFlags();
  parser.ParseFromString(non_default_options, &parsed_options);
  EXPECT_EQ(parsed_options.xla_dump_to(), dump_folder);
  EXPECT_EQ(parsed_options.xla_gpu_dot_merger_threshold_mb(),
            default_options.xla_gpu_dot_merger_threshold_mb() + 100);
  EXPECT_EQ(
      parsed_options.xla_gpu_experimental_collective_cse_distance_threshold(),
      default_options.xla_gpu_experimental_collective_cse_distance_threshold() +
          100);
  EXPECT_EQ(parsed_options.xla_gpu_enable_nccl_user_buffers(),
            !default_options.xla_gpu_enable_nccl_user_buffers());
  EXPECT_EQ(parsed_options.xla_gpu_enable_command_buffer_size(), 2);
  EXPECT_EQ(parsed_options.xla_gpu_enable_command_buffer(0),
            DebugOptions::CUBLAS);
  EXPECT_EQ(parsed_options.xla_gpu_enable_command_buffer(1),
            DebugOptions::FUSION);
  EXPECT_EQ(parsed_options.xla_gpu_analytical_latency_estimator_options().at(
                "gpus_per_node"),
            std::to_string(gpus_per_node + 1));
  EXPECT_EQ(parsed_options.xla_gpu_analytical_latency_estimator_options().at(
                "chunk_size_bytes"),
            std::to_string(chunk_size_bytes));

  HloModuleConfig config;
  config.set_debug_options(options);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnUnverifiedModule(R"(
    HloModule test
    ENTRY test {
      p0 = s32[11] parameter(0)
      c = s32[11] constant({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
      ROOT x = s32[11] multiply(p0, c)
    }
  )",
                                                         config));
  DumpNonDefaultDebugOptions(*m, kNonDefaultDebugOptionsDumpSuffix);
  std::string real_contents;
  TF_ASSERT_OK(tsl::ReadFileToString(
      tsl::Env::Default(),
      tsl::io::JoinPath(dump_folder,
                        FilenameFor(*m, "", kNonDefaultDebugOptionsDumpSuffix)),
      &real_contents));
  EXPECT_THAT(real_contents, testing::Eq(non_default_options));
}

}  // namespace
}  // namespace xla
