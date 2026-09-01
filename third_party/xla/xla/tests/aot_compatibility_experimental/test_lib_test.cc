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

#include "xla/tests/aot_compatibility_experimental/test_lib.h"

#include <stdlib.h>

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/tests/aot_interception_pjrt_client.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/util/split_proto/human_readable_aot_executable.pb.h"
#include "tsl/platform/path.h"

namespace xla {
namespace aot_compatibility_experimental {
namespace {

using ::absl_testing::StatusIs;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::tsl::proto_testing::ParseTextProtoOrDie;

TEST(TestLibTest, GetAotTestParamsForBackwardsCompatibility_With4Versions) {
  unsetenv("XLA_AOT_TEST_ALL_VERSIONS");
  ASSERT_OK_AND_ASSIGN(
      std::vector<AotTestParam> params,
      GetAotTestParamsForBackwardsCompatibility("test_dummy_test"));
  EXPECT_THAT(params,
              ElementsAre(AotTestParam{AOTTestMode::kBackwardsCompatibility, 1,
                                       "test_dummy_test"},
                          AotTestParam{AOTTestMode::kBackwardsCompatibility, 3,
                                       "test_dummy_test"}));
}

TEST(TestLibTest, GetAotTestParamsForBackwardsCompatibility_AllVersions) {
  setenv("XLA_AOT_TEST_ALL_VERSIONS", "1", 1);
  ASSERT_OK_AND_ASSIGN(
      std::vector<AotTestParam> params,
      GetAotTestParamsForBackwardsCompatibility("test_dummy_test"));
  EXPECT_THAT(params,
              ElementsAre(AotTestParam{AOTTestMode::kBackwardsCompatibility, 1,
                                       "test_dummy_test"},
                          AotTestParam{AOTTestMode::kBackwardsCompatibility, 2,
                                       "test_dummy_test"},
                          AotTestParam{AOTTestMode::kBackwardsCompatibility, 3,
                                       "test_dummy_test"},
                          AotTestParam{AOTTestMode::kBackwardsCompatibility, 4,
                                       "test_dummy_test"}));
  unsetenv("XLA_AOT_TEST_ALL_VERSIONS");
}

TEST(TestLibTest, GetAotTestParamsForGoldenFileVerification_With4Versions) {
  ASSERT_OK_AND_ASSIGN(
      std::vector<AotTestParam> params,
      GetAotTestParamsForGoldenFileVerification("test_dummy_test"));
  EXPECT_THAT(params, ElementsAre(AotTestParam{AOTTestMode::kGoldenVerification,
                                               4, "test_dummy_test"}));
}

TEST(TestLibTest,
     AOTInterceptionPjrtClientPackArtifactForInnerClient_Succeeds) {
  std::string artifact_path = tsl::io::JoinPath(
      GetExecutablesDirectory("test_dummy_test"), "v1", "exec.pbtxt");
  AOTInterceptionPjrtClient client(
      nullptr, AOTTestMode::kBackwardsCompatibility, artifact_path);
  ASSERT_OK_AND_ASSIGN(std::string serialized,
                       client.PackArtifactForInnerClient());
  EXPECT_FALSE(serialized.empty());
}

TEST(TestLibTest, CompareGPUExecutables_NormalizesXlaDumpToAndReturnsOk) {
  auto golden = ParseTextProtoOrDie<HumanReadableAotExecutable>(R"pb(
    gpu_executable {
      binary: "golden_binary"
      hlo_module_with_config {
        config { debug_options { xla_dump_to: "/tmp/golden_dir" } }
      }
      asm_text: "test_asm"
    }
  )pb");
  auto fresh = ParseTextProtoOrDie<HumanReadableAotExecutable>(R"pb(
    gpu_executable {
      binary: "fresh_binary_that_should_be_ignored"
      hlo_module_with_config {
        config { debug_options { xla_dump_to: "/tmp/fresh_dir" } }
      }
      asm_text: "completely_different_asm"
    }
  )pb");
  EXPECT_OK(AOTInterceptionPjrtClient::CompareGPUExecutables(fresh, golden));
}

// TODO(b/528258781): Debug options are currently cleared wholesale before the
// structural comparison, so differing compiler flags are NOT detected. Once we
// decide which flags must be preserved, this test should assert that meaningful
// flag changes are detected again.
TEST(TestLibTest, CompareGPUExecutables_IgnoresDebugOptionsForNow) {
  auto golden = ParseTextProtoOrDie<HumanReadableAotExecutable>(R"pb(
    gpu_executable {
      binary: "same_binary"
      hlo_module_with_config {
        config { debug_options { xla_gpu_enable_fast_min_max: true } }
      }
    }
  )pb");
  auto fresh = ParseTextProtoOrDie<HumanReadableAotExecutable>(R"pb(
    gpu_executable {
      binary: "same_binary"
      hlo_module_with_config {
        config { debug_options { xla_gpu_enable_fast_min_max: false } }
      }
    }
  )pb");
  EXPECT_OK(AOTInterceptionPjrtClient::CompareGPUExecutables(fresh, golden));
}

// The compile-options copy of debug_options must also be cleared before the
// structural comparison; a flag difference there must not cause a spurious
// mismatch. See b/528258781.
TEST(TestLibTest,
     CompareGPUExecutables_IgnoresCompileOptionsDebugOptionsForNow) {
  auto golden = ParseTextProtoOrDie<HumanReadableAotExecutable>(R"pb(
    gpu_executable { binary: "same_binary" }
    executable_and_options {
      compile_options {
        executable_build_options {
          debug_options { xla_gpu_enable_fast_min_max: true }
        }
      }
    }
  )pb");
  auto fresh = ParseTextProtoOrDie<HumanReadableAotExecutable>(R"pb(
    gpu_executable { binary: "same_binary" }
    executable_and_options {
      compile_options {
        executable_build_options {
          debug_options { xla_gpu_enable_fast_min_max: false }
        }
      }
    }
  )pb");
  EXPECT_OK(AOTInterceptionPjrtClient::CompareGPUExecutables(fresh, golden));
}

// A different host-specific debug option (CUDA install path) must be normalized
// away and must not cause a spurious mismatch.
TEST(TestLibTest, CompareGPUExecutables_NormalizesHostPathFieldsAndReturnsOk) {
  auto golden = ParseTextProtoOrDie<HumanReadableAotExecutable>(R"pb(
    gpu_executable {
      binary: "same_binary"
      hlo_module_with_config {
        config { debug_options { xla_gpu_cuda_data_dir: "/host_a/cuda" } }
      }
    }
  )pb");
  auto fresh = ParseTextProtoOrDie<HumanReadableAotExecutable>(R"pb(
    gpu_executable {
      binary: "same_binary"
      hlo_module_with_config {
        config { debug_options { xla_gpu_cuda_data_dir: "/host_b/cuda" } }
      }
    }
  )pb");
  EXPECT_OK(AOTInterceptionPjrtClient::CompareGPUExecutables(fresh, golden));
}

TEST(TestLibTest, CompareGPUExecutables_FailsOnGenuineDifferences) {
  auto golden = ParseTextProtoOrDie<HumanReadableAotExecutable>(R"pb(
    gpu_executable { binary: "same_binary" module_name: "test_module" }
  )pb");
  auto fresh = ParseTextProtoOrDie<HumanReadableAotExecutable>(R"pb(
    gpu_executable { binary: "same_binary" module_name: "different_module" }
  )pb");

  EXPECT_THAT(
      AOTInterceptionPjrtClient::CompareGPUExecutables(fresh, golden),
      StatusIs(absl::StatusCode::kInternal,
               AllOf(HasSubstr("Golden Proto structural comparison failed"),
                     HasSubstr("module_name"), HasSubstr("test_module"),
                     HasSubstr("different_module"))));
}

// The CPU comparator must ignore both copies of debug_options as well as the
// compiled machine-code fields (object_files, target_machine_options,
// data_layout). See b/528258781.
TEST(TestLibTest, CompareGoldenCPUExecutable_IgnoresDebugOptionsAndReturnsOk) {
  HumanReadableAotExecutable golden;
  golden.mutable_cpu_executable()->set_entry_function_name("main");
  golden.mutable_cpu_executable()
      ->mutable_hlo_module()
      ->mutable_config()
      ->mutable_debug_options()
      ->set_xla_dump_to("/tmp/golden_dir");
  golden.mutable_cpu_executable()->set_data_layout("golden_layout");

  HumanReadableAotExecutable fresh;
  fresh.mutable_cpu_executable()->set_entry_function_name("main");
  fresh.mutable_cpu_executable()
      ->mutable_hlo_module()
      ->mutable_config()
      ->mutable_debug_options()
      ->set_xla_dump_to("/tmp/fresh_dir");
  fresh.mutable_cpu_executable()->set_data_layout("fresh_layout_ignored");

  EXPECT_OK(
      AOTInterceptionPjrtClient::CompareGoldenCPUExecutable(fresh, golden));
}

TEST(TestLibTest, CompareGoldenCPUExecutable_FailsOnGenuineDifferences) {
  HumanReadableAotExecutable golden;
  golden.mutable_cpu_executable()->set_entry_function_name("main");

  HumanReadableAotExecutable fresh;
  fresh.mutable_cpu_executable()->set_entry_function_name("different_entry");

  EXPECT_THAT(
      AOTInterceptionPjrtClient::CompareGoldenCPUExecutable(fresh, golden),
      StatusIs(absl::StatusCode::kInternal,
               AllOf(HasSubstr("Golden Proto structural comparison failed"),
                     HasSubstr("entry_function_name"),
                     HasSubstr("different_entry"))));
}

// CPU artifacts are plain (non-split) protos: an ExecutableAndOptionsProto
// whose serialized_executable holds a serialized cpu::CompilationResultProto.
// The CPU deserializer must unpack the inner proto and clear
// serialized_executable.
TEST(TestLibTest, DeserializeToHumanReadable_CpuUnpacksPlainProto) {
  cpu::CompilationResultProto cpu_proto;
  cpu_proto.set_entry_function_name("main");

  ExecutableAndOptionsProto outer;
  outer.set_serialized_executable(cpu_proto.SerializeAsString());
  std::string serialized;
  ASSERT_TRUE(outer.SerializeToString(&serialized));

  ASSERT_OK_AND_ASSIGN(HumanReadableAotExecutable unpacked,
                       AOTInterceptionPjrtClient::DeserializeToHumanReadable(
                           serialized, AOTTestPlatform::kCpu));
  EXPECT_TRUE(unpacked.has_cpu_executable());
  EXPECT_EQ(unpacked.cpu_executable().entry_function_name(), "main");
  EXPECT_TRUE(
      unpacked.executable_and_options().serialized_executable().empty());
}

}  // namespace
}  // namespace aot_compatibility_experimental
}  // namespace xla
