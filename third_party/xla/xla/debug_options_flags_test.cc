/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/debug_options_flags.h"

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_set.h"
#include "google/protobuf/descriptor.h"
#include "xla/parse_flags_from_env.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/protobuf.h"

using ::testing::ElementsAre;
using ::testing::IsEmpty;

namespace xla {
namespace {

TEST(DebugOptions, GetDebugOptionsFromProtoAndFlags_WithEmptyProto) {
  int* pargc;
  std::vector<char*>* pargv;
  ResetFlagsFromEnvForTesting("XLA_FLAGS", &pargc, &pargv);
  tsl::setenv("XLA_FLAGS", "--xla_cpu_enable_fast_math=true", 1);

  DebugOptions empty_options;
  DebugOptions options = GetDebugOptionsFromProtoAndFlags(&empty_options);
  EXPECT_TRUE(options.xla_cpu_enable_fast_math());
}

TEST(DebugOptions, GetDebugOptionsFromProtoAndFlags_WithExistingProto) {
  int* pargc;
  std::vector<char*>* pargv;
  ResetFlagsFromEnvForTesting("XLA_FLAGS", &pargc, &pargv);

  DebugOptions proto_options;
  proto_options.set_xla_cpu_enable_fast_math(false);
  proto_options.set_xla_backend_optimization_level(1);

  tsl::setenv("XLA_FLAGS", "--xla_cpu_enable_fast_math=true", 1);
  DebugOptions options = GetDebugOptionsFromProtoAndFlags(&proto_options);

  EXPECT_TRUE(options.xla_cpu_enable_fast_math());
  EXPECT_EQ(options.xla_backend_optimization_level(), 1);
}

TEST(DebugOptions, GetDebugOptionsFromProtoAndFlags_PtxCompilerExtraFlags) {
  int* pargc;
  std::vector<char*>* pargv;
  ResetFlagsFromEnvForTesting("XLA_FLAGS", &pargc, &pargv);

  // Needed to avoid using globally-set and cached flags set by other tests.
  DebugOptions empty_options;
  tsl::setenv("XLA_FLAGS",
              "--xla_gpu_ptx_compiler_extra_flags='--maxntid=8,8,8 "
              "--register-usage-level=10'",
              1);

  DebugOptions options = GetDebugOptionsFromProtoAndFlags(&empty_options);
  EXPECT_THAT(options.xla_gpu_ptx_compiler_extra_flags(),
              ElementsAre("--maxntid=8,8,8", "--register-usage-level=10"));
}

TEST(DebugOptions, CommandBufferUpdateModeDefaultsToAlwaysUpdate) {
  EXPECT_EQ(
      DefaultDebugOptionsIgnoringFlags().xla_gpu_command_buffer_update_mode(),
      DebugOptions::ALWAYS_UPDATE);
}

TEST(DebugOptions, SchedulerMemoryFencingDefaultsToDisabled) {
  EXPECT_EQ(
      DefaultDebugOptionsIgnoringFlags()
          .xla_gpu_experimental_scheduler_memory_fencing_threshold_bytes(),
      -1);
}

TEST(DebugOptions, CommandBufferUpdateModesParseFromFlags) {
  for (const auto& [name, expected] : std::vector<
           std::pair<const char*, DebugOptions::CommandBufferUpdateMode>>{
           {"ALWAYS_UPDATE", DebugOptions::ALWAYS_UPDATE},
           {"SKIP_TEMP", DebugOptions::SKIP_TEMP},
           {"SKIP_PROFILED", DebugOptions::SKIP_PROFILED}}) {
    int* pargc;
    std::vector<char*>* pargv;
    ResetFlagsFromEnvForTesting("XLA_FLAGS", &pargc, &pargv);
    std::string flag = "--xla_gpu_command_buffer_update_mode=";
    flag += name;
    tsl::setenv("XLA_FLAGS", flag.c_str(), 1);

    DebugOptions proto_options;
    DebugOptions options = GetDebugOptionsFromProtoAndFlags(&proto_options);

    EXPECT_EQ(options.xla_gpu_command_buffer_update_mode(), expected);
  }
}

TEST(DebugOptions, RemovedCommandBufferUpdateModesRejectedByTextProto) {
  for (const char* name : {"DYNAMIC_ALLOCATE", "VMM_PERSISTENT_TEMP",
                           "NEVER_UPDATE", "CAPTURE_CMD_NEVER_UPDATE"}) {
    DebugOptions options;
    std::string text = "xla_gpu_command_buffer_update_mode: ";
    text += name;
    EXPECT_FALSE(tsl::protobuf::TextFormat::ParseFromString(text, &options));
  }
}

TEST(DebugOptionsDeathTest, RemovedCommandBufferUpdateModesRejectedByFlags) {
  for (const char* name : {"DYNAMIC_ALLOCATE", "VMM_PERSISTENT_TEMP",
                           "NEVER_UPDATE", "CAPTURE_CMD_NEVER_UPDATE"}) {
    int* pargc;
    std::vector<char*>* pargv;
    ResetFlagsFromEnvForTesting("XLA_FLAGS", &pargc, &pargv);
    std::string flag = "--xla_gpu_command_buffer_update_mode=";
    flag += name;
    tsl::setenv("XLA_FLAGS", flag.c_str(), 1);

    DebugOptions proto_options;
    EXPECT_DEATH((void)GetDebugOptionsFromProtoAndFlags(&proto_options),
                 "Flag parsing failed")
        << name;
    tsl::unsetenv("XLA_FLAGS");
  }
}

TEST(DebugOptions, AllFieldsHavePresence) {
  absl::flat_hash_set<std::string> fields_missing_presence;

  const tsl::protobuf::Descriptor* debug_options = DebugOptions::descriptor();
  for (int i = 0; i < debug_options->field_count(); ++i) {
    const tsl::protobuf::FieldDescriptor* field = debug_options->field(i);
    // Repeated fields don't technically have presence (no has_foo) but
    // foo().empty() is just as good.
    if (!field->is_repeated() && !field->has_presence()) {
      fields_missing_presence.insert(std::string(field->name()));
    }
  }

  EXPECT_THAT(fields_missing_presence, IsEmpty())
      << "All scalar fields in DebugOptions must have presence defined by "
         "being labeled `optional`.";
}

TEST(DebugOptions, EnableNcclSymmetricBuffersForCollectives) {
  int* pargc;
  std::vector<char*>* pargv;
  ResetFlagsFromEnvForTesting("XLA_FLAGS", &pargc, &pargv);

  DebugOptions empty_options;
  tsl::setenv("XLA_FLAGS",
              "--xla_enable_nccl_symmetric_buffers_for_collectives="
              "AllReduce:1024:f32,AllGather:2048:S32,ReduceScatter,all",
              1);

  DebugOptions options = GetDebugOptionsFromProtoAndFlags(&empty_options);
  ASSERT_EQ(options.xla_enable_nccl_symmetric_buffers_for_collectives_size(),
            4);

  {
    const auto& filter =
        options.xla_enable_nccl_symmetric_buffers_for_collectives(0);
    EXPECT_EQ(filter.collective(), DebugOptions::ALLREDUCE);
    EXPECT_EQ(filter.max_size_bytes(), 1024);
    EXPECT_EQ(filter.op_type(), xla::F32);
  }
  {
    const auto& filter =
        options.xla_enable_nccl_symmetric_buffers_for_collectives(1);
    EXPECT_EQ(filter.collective(), DebugOptions::ALLGATHER);
    EXPECT_EQ(filter.max_size_bytes(), 2048);
    EXPECT_EQ(filter.op_type(), xla::S32);
  }
  {
    const auto& filter =
        options.xla_enable_nccl_symmetric_buffers_for_collectives(2);
    EXPECT_EQ(filter.collective(), DebugOptions::REDUCESCATTER);
    EXPECT_FALSE(filter.has_max_size_bytes());
    EXPECT_FALSE(filter.has_op_type());
  }
  {
    const auto& filter =
        options.xla_enable_nccl_symmetric_buffers_for_collectives(3);
    EXPECT_EQ(filter.collective(), DebugOptions::ALLCOLLECTIVES);
    EXPECT_FALSE(filter.has_max_size_bytes());
    EXPECT_FALSE(filter.has_op_type());
  }
}

// Helper that parses a single flag value using MakeDebugOptionsFlags and
// returns the resulting DebugOptions.  The flag is passed as "--name=value".
DebugOptions ParseCollectiveKernelsFlag(const std::string& value) {
  DebugOptions opts;
  std::vector<tsl::Flag> flags;
  MakeDebugOptionsFlags(&flags, &opts);
  std::vector<std::string> flag_args = {
      "--xla_gpu_experimental_use_collective_kernels=" + value};
  EXPECT_TRUE(tsl::Flags::Parse(flag_args, flags));
  return opts;
}

// ---- xla_gpu_experimental_use_collective_kernels user-friendly name tests ---

TEST(DebugOptions, CollectiveKernelsFlagHyphenatedAllReduce) {
  DebugOptions opts = ParseCollectiveKernelsFlag("ALL_REDUCE");
  EXPECT_THAT(opts.xla_gpu_experimental_use_collective_kernels(),
              ElementsAre(DebugOptions::COLLECTIVE_KERNEL_ALL_REDUCE));
}

TEST(DebugOptions, CollectiveKernelsFlagUnderscoreAllReduce) {
  DebugOptions opts = ParseCollectiveKernelsFlag("all_reduce");
  EXPECT_THAT(opts.xla_gpu_experimental_use_collective_kernels(),
              ElementsAre(DebugOptions::COLLECTIVE_KERNEL_ALL_REDUCE));
}

TEST(DebugOptions, CollectiveKernelsFlagHyphenatedAllGather) {
  DebugOptions opts = ParseCollectiveKernelsFlag("ALL_GATHER");
  EXPECT_THAT(opts.xla_gpu_experimental_use_collective_kernels(),
              ElementsAre(DebugOptions::COLLECTIVE_KERNEL_ALL_GATHER));
}

TEST(DebugOptions, CollectiveKernelsFlagUnderscoreAllGather) {
  DebugOptions opts = ParseCollectiveKernelsFlag("all_gather");
  EXPECT_THAT(opts.xla_gpu_experimental_use_collective_kernels(),
              ElementsAre(DebugOptions::COLLECTIVE_KERNEL_ALL_GATHER));
}

TEST(DebugOptions, CollectiveKernelsFlagBothHyphenated) {
  DebugOptions opts = ParseCollectiveKernelsFlag("ALL_REDUCE,ALL_GATHER");
  EXPECT_THAT(opts.xla_gpu_experimental_use_collective_kernels(),
              ElementsAre(DebugOptions::COLLECTIVE_KERNEL_ALL_REDUCE,
                          DebugOptions::COLLECTIVE_KERNEL_ALL_GATHER));
}

TEST(DebugOptions, CollectiveKernelsFlagFullEnumName) {
  DebugOptions opts =
      ParseCollectiveKernelsFlag("COLLECTIVE_KERNEL_ALL_REDUCE");
  EXPECT_THAT(opts.xla_gpu_experimental_use_collective_kernels(),
              ElementsAre(DebugOptions::COLLECTIVE_KERNEL_ALL_REDUCE));
}

TEST(DebugOptions, CollectiveKernelsFlagShortEnumNameUppercase) {
  DebugOptions opts = ParseCollectiveKernelsFlag("ALL_REDUCE");
  EXPECT_THAT(opts.xla_gpu_experimental_use_collective_kernels(),
              ElementsAre(DebugOptions::COLLECTIVE_KERNEL_ALL_REDUCE));
}

TEST(DebugOptions, CollectiveKernelsFlagEmptyDisablesAll) {
  DebugOptions opts = ParseCollectiveKernelsFlag("");
  EXPECT_THAT(opts.xla_gpu_experimental_use_collective_kernels(), IsEmpty());
}

TEST(DebugOptions, CollectiveKernelsFlagIncrementalAdd) {
  // Start with all-reduce (from default), then add all-gather via +.
  DebugOptions opts;
  std::vector<tsl::Flag> flags;
  MakeDebugOptionsFlags(&flags, &opts);
  // First parse: set to all-reduce only.
  {
    std::vector<std::string> flag_args = {
        "--xla_gpu_experimental_use_collective_kernels=ALL_REDUCE"};
    EXPECT_TRUE(tsl::Flags::Parse(flag_args, flags));
  }
  EXPECT_THAT(opts.xla_gpu_experimental_use_collective_kernels(),
              ElementsAre(DebugOptions::COLLECTIVE_KERNEL_ALL_REDUCE));
  // Second parse: incrementally add all-gather.
  {
    std::vector<std::string> flag_args = {
        "--xla_gpu_experimental_use_collective_kernels=+ALL_GATHER"};
    EXPECT_TRUE(tsl::Flags::Parse(flag_args, flags));
  }
  EXPECT_THAT(opts.xla_gpu_experimental_use_collective_kernels(),
              ElementsAre(DebugOptions::COLLECTIVE_KERNEL_ALL_REDUCE,
                          DebugOptions::COLLECTIVE_KERNEL_ALL_GATHER));
}

TEST(DebugOptions, CollectiveKernelsFlagIncrementalRemove) {
  // Start with both, then remove all-reduce via -.
  DebugOptions opts;
  std::vector<tsl::Flag> flags;
  MakeDebugOptionsFlags(&flags, &opts);
  // First parse: set to both.
  {
    std::vector<std::string> flag_args = {
        "--xla_gpu_experimental_use_collective_kernels=ALL_REDUCE,ALL_GATHER"};
    EXPECT_TRUE(tsl::Flags::Parse(flag_args, flags));
  }
  EXPECT_THAT(opts.xla_gpu_experimental_use_collective_kernels(),
              ElementsAre(DebugOptions::COLLECTIVE_KERNEL_ALL_REDUCE,
                          DebugOptions::COLLECTIVE_KERNEL_ALL_GATHER));
  // Second parse: incrementally remove all-reduce.
  {
    std::vector<std::string> flag_args = {
        "--xla_gpu_experimental_use_collective_kernels=-ALL_REDUCE"};
    EXPECT_TRUE(tsl::Flags::Parse(flag_args, flags));
  }
  EXPECT_THAT(opts.xla_gpu_experimental_use_collective_kernels(),
              ElementsAre(DebugOptions::COLLECTIVE_KERNEL_ALL_GATHER));
}

TEST(DebugOptions, CollectiveKernelsFlagNoDuplicates) {
  // Parsing the same value twice should not produce duplicates.
  DebugOptions opts;
  std::vector<tsl::Flag> flags;
  MakeDebugOptionsFlags(&flags, &opts);
  for (int i = 0; i < 2; ++i) {
    std::vector<std::string> flag_args = {
        "--xla_gpu_experimental_use_collective_kernels=+ALL_REDUCE"};
    EXPECT_TRUE(tsl::Flags::Parse(flag_args, flags));
  }
  EXPECT_THAT(opts.xla_gpu_experimental_use_collective_kernels(),
              ElementsAre(DebugOptions::COLLECTIVE_KERNEL_ALL_REDUCE));
}

// -------------------------------------------------------------------------
// xla_gpu_hlo_custom_call_allowlist flag/proto tests
// (feature: XLA:GPU custom-call AOT allowlist enforcement).
// -------------------------------------------------------------------------

// Helper that parses a single --xla_gpu_hlo_custom_call_allowlist value using
// MakeDebugOptionsFlags and returns the resulting DebugOptions.
DebugOptions ParseAotAllowlistFlag(const std::string& value) {
  DebugOptions opts;
  std::vector<tsl::Flag> flags;
  MakeDebugOptionsFlags(&flags, &opts);
  std::vector<std::string> flag_args = {"--xla_gpu_hlo_custom_call_allowlist=" +
                                        value};
  EXPECT_TRUE(tsl::Flags::Parse(flag_args, flags));
  return opts;
}

// UT-7: comma-separated values are split into individual allowlist entries.
TEST(DebugOptions, HloCustomCallAllowlistFlagCommaSplit) {
  DebugOptions opts = ParseAotAllowlistFlag("target_a,target_b,target_c");
  EXPECT_THAT(opts.xla_gpu_hlo_custom_call_allowlist(),
              ElementsAre("target_a", "target_b", "target_c"));
}

// UT-7: a single value produces a single allowlist entry.
TEST(DebugOptions, HloCustomCallAllowlistFlagSingleValue) {
  DebugOptions opts = ParseAotAllowlistFlag("only_target");
  EXPECT_THAT(opts.xla_gpu_hlo_custom_call_allowlist(),
              ElementsAre("only_target"));
}

// UT-7: the empty flag value leaves the allowlist empty (feature disabled).
TEST(DebugOptions, HloCustomCallAllowlistFlagEmptyIsDisabled) {
  DebugOptions opts = ParseAotAllowlistFlag("");
  EXPECT_THAT(opts.xla_gpu_hlo_custom_call_allowlist(), IsEmpty());
}

// UT-7: empty tokens (from absl::SkipEmpty) are dropped during parsing.
TEST(DebugOptions, HloCustomCallAllowlistFlagSkipsEmptyTokens) {
  DebugOptions opts = ParseAotAllowlistFlag("a,,b,");
  EXPECT_THAT(opts.xla_gpu_hlo_custom_call_allowlist(), ElementsAre("a", "b"));
}

// UT-7: surrounding whitespace around each token is trimmed, and
// whitespace-only tokens are dropped, so "a, b , ,c" yields exactly {a,b,c}.
TEST(DebugOptions, HloCustomCallAllowlistFlagTrimsWhitespace) {
  DebugOptions opts = ParseAotAllowlistFlag("a, b , ,c");
  EXPECT_THAT(opts.xla_gpu_hlo_custom_call_allowlist(),
              ElementsAre("a", "b", "c"));
}

// UT-7: repeated flag occurrences accumulate (setter appends, never clears).
TEST(DebugOptions, HloCustomCallAllowlistFlagRepeatedAccumulates) {
  DebugOptions opts;
  std::vector<tsl::Flag> flags;
  MakeDebugOptionsFlags(&flags, &opts);
  std::vector<std::string> first = {"--xla_gpu_hlo_custom_call_allowlist=a,b"};
  std::vector<std::string> second = {"--xla_gpu_hlo_custom_call_allowlist=c"};
  EXPECT_TRUE(tsl::Flags::Parse(first, flags));
  EXPECT_TRUE(tsl::Flags::Parse(second, flags));
  EXPECT_THAT(opts.xla_gpu_hlo_custom_call_allowlist(),
              ElementsAre("a", "b", "c"));
}

// UT-8: the repeated proto field (id 537) survives a serialize/parse
// round-trip, confirming the field id does not collide and set/get works as
// expected.
TEST(DebugOptions, HloCustomCallAllowlistProtoRoundTrip) {
  DebugOptions opts;
  opts.add_xla_gpu_hlo_custom_call_allowlist("foo");
  opts.add_xla_gpu_hlo_custom_call_allowlist("bar");

  std::string wire;
  ASSERT_TRUE(opts.SerializeToString(&wire));

  DebugOptions parsed;
  ASSERT_TRUE(parsed.ParseFromString(wire));
  EXPECT_THAT(parsed.xla_gpu_hlo_custom_call_allowlist(),
              ElementsAre("foo", "bar"));
}

// -------------------------------------------------------------------------
// xla_disable_hlo_passes / xla_enable_hlo_passes_only flag setter tests
// (feature: richer entry syntax with occurrence / scope / pass_id).
// -------------------------------------------------------------------------

// Helper that parses a single --xla_disable_hlo_passes value and returns the
// parse result together with the resulting DebugOptions.
std::pair<bool, DebugOptions> ParseDisableHloPassesFlag(
    const std::string& value) {
  DebugOptions opts;
  std::vector<tsl::Flag> flags;
  MakeDebugOptionsFlags(&flags, &opts);
  std::vector<std::string> flag_args = {"--xla_disable_hlo_passes=" + value};
  bool ok = tsl::Flags::Parse(flag_args, flags);
  return {ok, opts};
}

std::pair<bool, DebugOptions> ParseEnableHloPassesOnlyFlag(
    const std::string& value) {
  DebugOptions opts;
  std::vector<tsl::Flag> flags;
  MakeDebugOptionsFlags(&flags, &opts);
  std::vector<std::string> flag_args = {"--xla_enable_hlo_passes_only=" +
                                        value};
  bool ok = tsl::Flags::Parse(flag_args, flags);
  return {ok, opts};
}

// All five supported entry formats parse successfully.
TEST(DebugOptions, DisableHloPassesRichSyntaxIsAccepted) {
  auto [ok, opts] = ParseDisableHloPassesFlag(
      "algsimp,algsimp:2,simplification/algsimp,simplification/algsimp:2,@42");
  EXPECT_TRUE(ok);
  EXPECT_THAT(opts.xla_disable_hlo_passes(),
              ElementsAre("algsimp", "algsimp:2", "simplification/algsimp",
                          "simplification/algsimp:2", "@42"));
}

TEST(DebugOptions, EnableHloPassesOnlyRichSyntaxIsAccepted) {
  auto [ok, opts] = ParseEnableHloPassesOnlyFlag("simplification/algsimp:2,@7");
  EXPECT_TRUE(ok);
  EXPECT_THAT(opts.xla_enable_hlo_passes_only(),
              ElementsAre("simplification/algsimp:2", "@7"));
}

// Malformed entries are rejected at flag-parse time.
TEST(DebugOptions, DisableHloPassesRejectsMalformedEntries) {
  EXPECT_FALSE(ParseDisableHloPassesFlag("@notanumber").first);
  EXPECT_FALSE(ParseDisableHloPassesFlag("algsimp:notanumber").first);
  EXPECT_FALSE(ParseEnableHloPassesOnlyFlag("@").first);
}

TEST(DebugOptions, DeduplicateBackendConfigsMinSizeDefaultIsMaxInt) {
  DebugOptions opts = DefaultDebugOptionsIgnoringFlags();
  EXPECT_EQ(opts.xla_deduplicate_backend_configs_min_size(),
            std::numeric_limits<int64_t>::max());
}

TEST(DebugOptions, DeduplicateBackendConfigsMinSizeFlagsParsing) {
  DebugOptions opts;
  std::vector<tsl::Flag> flags;
  MakeDebugOptionsFlags(&flags, &opts);
  std::vector<std::string> flag_args = {
      "--xla_deduplicate_backend_configs_min_size=128"};
  EXPECT_TRUE(tsl::Flags::Parse(flag_args, flags));
  EXPECT_EQ(opts.xla_deduplicate_backend_configs_min_size(), 128);
}

}  // namespace
}  // namespace xla
