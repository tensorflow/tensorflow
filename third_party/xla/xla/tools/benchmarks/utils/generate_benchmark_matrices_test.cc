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
#include "xla/tools/benchmarks/utils/generate_benchmark_matrices.h"

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "json/json.h"
#include "xla/tools/benchmarks/proto/benchmark_config.pb.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"

namespace xla {
namespace tools {
namespace benchmarks {
namespace {

using testing::ElementsAre;
using testing::HasSubstr;
using testing::Not;
using testing::SizeIs;
using tsl::testing::IsOkAndHolds;
using tsl::testing::StatusIs;

// Helper function to create a temporary registry file.
std::string CreateTempRegistryFile(const std::string& content,
                                   const std::string& filename_prefix =
                                       "registry_test_") {  // Underscore added
  std::string temp_dir = tsl::testing::TmpDir();
  // Ensure .textproto extension for clarity, though LoadBenchmarkSuiteFromFile
  // parses as TextProto
  std::string filepath = tsl::io::JoinPath(
      temp_dir, absl::StrCat(filename_prefix, tsl::Env::Default()->NowMicros(),
                             ".textproto"));

  std::ofstream file_stream(filepath);
  if (!file_stream.is_open()) {
    ADD_FAILURE() << "Failed to open temporary file for writing: " << filepath;
    return "";
  }
  file_stream << content;
  file_stream.close();
  if (!file_stream) {  // Check if stream operations were successful
    ADD_FAILURE() << "Failed to write to or close temporary file: " << filepath;
    // Attempt to remove partially written or problematic file
    tsl::Env::Default()->DeleteFile(filepath).IgnoreError();
    return "";
  }
  EXPECT_TRUE(tsl::Env::Default()->FileExists(filepath).ok())
      << "File not found after creation: " << filepath;
  return filepath;
}

// Matchers for JsonCpp values
MATCHER_P(JsonStringEq, expected_str, "") {
  return arg.isString() && arg.asString() == expected_str;
}
MATCHER_P(JsonBoolEq, expected_bool, "") {
  return arg.isBool() && arg.asBool() == expected_bool;
}
MATCHER_P(JsonIntEq, expected_int, "") {
  return arg.isIntegral() && arg.asInt() == expected_int;
}
MATCHER_P(JsonArrayContainsString, expected_str, "") {
  if (!arg.isArray()) {
    return false;
  }
  for (const auto& item : arg) {
    if (item.isString() && item.asString() == expected_str) {
      return true;
    }
  }
  return false;
}
MATCHER(IsJsonArray, "") { return arg.isArray(); }
MATCHER(IsJsonObject, "") { return arg.isObject(); }

class GenerateBenchmarkMatricesTest : public testing::Test {
 protected:
  // Test registry with the new structure
  std::string CreateTestRegistryContentNew() {
    return R"(
benchmarks: [
  {
    name: "gemma_test"
    owner: "owner1@"
    input_artifact: {
      input_format: STABLEHLO_MLIR
      artifact_gcs_bucket_path: "gs://bucket/gemma.mlir"
    }
    model_source_info: ["Gemma Model"]
    github_labels: ["label_gemma"]
    hardware_execution_configs: [
      { # HEC 0 for gemma_test: GPU_L4, PRESUBMIT & SCHEDULED
        hardware_category: GPU_L4
        topology: {num_hosts:1, num_devices_per_host:1, multi_host: false, multi_device: false}
        target_metrics: [GPU_DEVICE_TIME, PEAK_GPU_MEMORY]
        workflow_type: [PRESUBMIT, SCHEDULED]
        runtime_flags: ["--repeat_l4=5"]
        xla_compilation_flags: ["--l4_specific_flag"] # Overrides common
      },
      { # HEC 1 for gemma_test: CPU_X86, PRESUBMIT only
        hardware_category: CPU_X86
        topology: {num_hosts:1, num_devices_per_host:2, multi_host: false, multi_device: true} # 1h2d
        target_metrics: [CPU_TIME]
        workflow_type: [PRESUBMIT]
        runtime_flags: ["--repeat_cpu=3"]
        # Uses common xla_compilation_flags from parent
      }
    ]
  },
  {
    name: "fusion_test"
    owner: "owner2@"
    input_artifact: {
      input_format: HLO_TEXT
      artifact_path: "hlo/fusion.hlo"
    }
    model_source_info: ["Fusion Model"]
    hardware_execution_configs: [
      { # HEC 0 for fusion_test: GPU_B200, POSTSUBMIT & MANUAL
        hardware_category: GPU_B200
        topology: {num_hosts:1, num_devices_per_host:1} # Defaults multi_host/device to false
        target_metrics: [GPU_DEVICE_TIME]
        workflow_type: [POSTSUBMIT, MANUAL]
        xla_compilation_flags: ["--b200_flag=true"]
      }
    ]
  },
  { # Benchmark with an unmappable hardware config (for testing error/skip)
    name: "unmapped_hw_test"
    owner: "test@"
    input_artifact: { input_format: HLO_TEXT, artifact_path: "test.hlo"}
    hardware_execution_configs: [
      {
        hardware_category: CPU_ARM64 # Assume this is not in our test runner map
        topology: {num_hosts:1, num_devices_per_host:1}
        workflow_type: [PRESUBMIT]
      }
    ]
  }
]
)";
  }
  void SetEnvVar(const std::string& name, const std::string& value) {
    setenv(name.c_str(), value.c_str(), 1);
    env_vars_to_clear_.push_back(name);
  }
  void TearDown() override {
    for (const auto& var_name : env_vars_to_clear_) {
      unsetenv(var_name.c_str());
    }
  }

 private:
  std::vector<std::string> env_vars_to_clear_;
};

// --- LoadBenchmarkSuiteFromFile Tests ---
TEST_F(GenerateBenchmarkMatricesTest, LoadBenchmarkSuiteFromFileSuccess) {
  std::string filepath = CreateTempRegistryFile(CreateTestRegistryContentNew());
  ASSERT_FALSE(filepath.empty());
  TF_ASSERT_OK_AND_ASSIGN(BenchmarkSuite suite,
                          LoadBenchmarkSuiteFromFile(filepath));

  ASSERT_THAT(suite.benchmarks(), SizeIs(3));
  const auto& gemma_def = suite.benchmarks(0);
  EXPECT_EQ(gemma_def.name(), "gemma_test");
  ASSERT_THAT(gemma_def.hardware_execution_configs(), SizeIs(2));
  EXPECT_EQ(gemma_def.hardware_execution_configs(0).hardware_category(),
            HardwareCategory::GPU_L4);
  EXPECT_EQ(gemma_def.hardware_execution_configs(1).hardware_category(),
            HardwareCategory::CPU_X86);
  EXPECT_THAT(gemma_def.hardware_execution_configs(0).xla_compilation_flags(),
              ElementsAre("--l4_specific_flag"));

  const auto& fusion_def = suite.benchmarks(1);
  EXPECT_EQ(fusion_def.name(), "fusion_test");
  ASSERT_THAT(fusion_def.hardware_execution_configs(), SizeIs(1));
  EXPECT_EQ(fusion_def.hardware_execution_configs(0).hardware_category(),
            HardwareCategory::GPU_B200);
}

TEST_F(GenerateBenchmarkMatricesTest,
       LoadBenchmarkSuiteFromFileFailsOnMissingBenchmarkName) {
  std::string content = R"(
    benchmarks: [ {
      # name: "missing_name"
      owner: "test@",
      input_artifact: {input_format: HLO_TEXT, artifact_path: "p.hlo"},
      hardware_execution_configs: [
        { hardware_category: GPU_L4, topology: {num_hosts:1, num_devices_per_host:1}, workflow_type: [PRESUBMIT]}
      ]
    } ]
  )";
  std::string filepath = CreateTempRegistryFile(content);
  ASSERT_FALSE(filepath.empty());
  EXPECT_THAT(
      LoadBenchmarkSuiteFromFile(filepath),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("empty 'name'")));
}

TEST_F(GenerateBenchmarkMatricesTest,
       LoadBenchmarkSuiteFromFileFailsOnNoHardwareExecutionConfigs) {
  std::string content = R"(
    benchmarks: [ {
      name: "no_hecs_benchmark",
      owner: "test@",
      input_artifact: {input_format: HLO_TEXT, artifact_path: "p.hlo"},
      hardware_execution_configs: [] # Empty list
    } ]
  )";
  std::string filepath = CreateTempRegistryFile(content);
  ASSERT_FALSE(filepath.empty());
  EXPECT_THAT(LoadBenchmarkSuiteFromFile(filepath),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("has no 'hardware_execution_configs'")));
}

// --- BuildGitHubActionsMatrix Tests ---
TEST_F(GenerateBenchmarkMatricesTest,
       BuildMatrix_Presubmit_GeneratesTwoEntriesForGemma) {
  std::string filepath = CreateTempRegistryFile(CreateTestRegistryContentNew());
  ASSERT_FALSE(filepath.empty());
  TF_ASSERT_OK_AND_ASSIGN(BenchmarkSuite suite,
                          LoadBenchmarkSuiteFromFile(filepath));

  TF_ASSERT_OK_AND_ASSIGN(
      Json::Value matrix,
      BuildGitHubActionsMatrix(suite, WorkflowType::PRESUBMIT));
  ASSERT_THAT(matrix, IsJsonArray());

  // Entry 0: gemma_test on GPU_L4 for PRESUBMIT
  const auto& entry0 = matrix[0];
  EXPECT_THAT(entry0["benchmark_name"], JsonStringEq("gemma_test"));
  EXPECT_THAT(entry0["hardware_category"], JsonStringEq("GPU_L4"));
  EXPECT_THAT(entry0["config_id"],
              JsonStringEq("gemma_test_l4_1h1d_presubmit"));
  EXPECT_THAT(entry0["runner_label"], JsonStringEq("linux-x86-g2-16-l4-1gpu"));
  EXPECT_THAT(entry0["container_image"],
              JsonStringEq("us-central1-docker.pkg.dev/tensorflow-sigs/"
                           "tensorflow/ml-build-cuda12.8-cudnn9.8:latest"));
  EXPECT_THAT(entry0["runtime_flags"],
              JsonArrayContainsString("--repeat_l4=5"));
  EXPECT_THAT(entry0["xla_compilation_flags"],
              JsonArrayContainsString("--l4_specific_flag"));
  EXPECT_THAT(entry0["xla_compilation_flags"],
              Not(JsonArrayContainsString("--common_gemma_flag")));
  EXPECT_THAT(entry0["workflow_type"], JsonStringEq("PRESUBMIT"));

  // Entry 1: gemma_test on CPU_X86 for PRESUBMIT
  const auto& entry1 = matrix[1];
  EXPECT_THAT(entry1["benchmark_name"], JsonStringEq("gemma_test"));
  EXPECT_THAT(entry1["hardware_category"], JsonStringEq("CPU_X86"));
  EXPECT_THAT(entry1["config_id"],
              JsonStringEq("gemma_test_x86_1h2d_presubmit"));
  EXPECT_THAT(entry1["runner_label"], JsonStringEq("linux-x86-n2-128"));
  EXPECT_THAT(entry1["container_image"],
              JsonStringEq("us-central1-docker.pkg.dev/tensorflow-sigs/"
                           "tensorflow/ml-build:latest"));
  EXPECT_THAT(entry1["runtime_flags"],
              JsonArrayContainsString("--repeat_cpu=3"));
  EXPECT_THAT(entry1["xla_compilation_flags"],
              Not(JsonArrayContainsString("--l4_specific_flag")));
  ASSERT_THAT(entry1["topology"], IsJsonObject());
  EXPECT_THAT(entry1["topology"]["num_devices_per_host"], JsonIntEq(2));
  EXPECT_THAT(entry1["topology"]["multi_device"], JsonBoolEq(true));
}

TEST_F(GenerateBenchmarkMatricesTest,
       BuildMatrix_Postsubmit_GeneratesOneEntryForFusion) {
  std::string filepath = CreateTempRegistryFile(CreateTestRegistryContentNew());
  ASSERT_FALSE(filepath.empty());
  TF_ASSERT_OK_AND_ASSIGN(BenchmarkSuite suite,
                          LoadBenchmarkSuiteFromFile(filepath));

  TF_ASSERT_OK_AND_ASSIGN(
      Json::Value matrix,
      BuildGitHubActionsMatrix(suite, WorkflowType::POSTSUBMIT));
  ASSERT_THAT(matrix, IsJsonArray());
  // fusion_test/GPU_B200 is for POSTSUBMIT

  const auto& entry0 = matrix[0];
  EXPECT_THAT(entry0["benchmark_name"], JsonStringEq("fusion_test"));
  EXPECT_THAT(entry0["hardware_category"], JsonStringEq("GPU_B200"));
  EXPECT_THAT(entry0["config_id"],
              JsonStringEq("fusion_test_b200_1h1d_postsubmit"));
  EXPECT_THAT(entry0["xla_compilation_flags"],
              JsonArrayContainsString("--b200_flag=true"));
  EXPECT_THAT(entry0["workflow_type"], JsonStringEq("POSTSUBMIT"));
}

TEST_F(GenerateBenchmarkMatricesTest,
       BuildMatrix_Scheduled_GeneratesOneEntryForGemmaL4) {
  std::string filepath = CreateTempRegistryFile(CreateTestRegistryContentNew());
  ASSERT_FALSE(filepath.empty());
  TF_ASSERT_OK_AND_ASSIGN(BenchmarkSuite suite,
                          LoadBenchmarkSuiteFromFile(filepath));

  TF_ASSERT_OK_AND_ASSIGN(
      Json::Value matrix,
      BuildGitHubActionsMatrix(suite, WorkflowType::SCHEDULED));
  ASSERT_THAT(matrix, IsJsonArray());
  // gemma_test/GPU_L4 is for SCHEDULED
  const auto& entry0 = matrix[0];
  EXPECT_THAT(entry0["benchmark_name"], JsonStringEq("gemma_test"));
  EXPECT_THAT(entry0["hardware_category"], JsonStringEq("GPU_L4"));
  EXPECT_THAT(entry0["config_id"],
              JsonStringEq("gemma_test_l4_1h1d_scheduled"));
  EXPECT_THAT(entry0["workflow_type"], JsonStringEq("SCHEDULED"));
}

TEST_F(GenerateBenchmarkMatricesTest, BuildMatrixFailsOnUnmappableHardware) {
  std::string specific_content = R"(
    benchmarks: [ {
        name: "unmapped_hw_test", owner: "test@",
        input_artifact: { input_format: HLO_TEXT, artifact_path: "test.hlo"},
        hardware_execution_configs: [ {
            hardware_category: GPU_NON_EXISTENT # This category is not in maps
            topology: {num_hosts:1, num_devices_per_host:1},
            workflow_type: [PRESUBMIT]
        }]
    }]
  )";
  std::string filepath = CreateTempRegistryFile(specific_content);
  EXPECT_THAT(LoadBenchmarkSuiteFromFile(filepath),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Error parsing TextProto registry file:")));
}

// --- FindRegistryFile Tests ---
TEST_F(GenerateBenchmarkMatricesTest, FindRegistryFileReturnsAbsolutePath) {
  std::string temp_dir = tsl::testing::TmpDir();
  std::string tmp_file = tsl::io::JoinPath(temp_dir, "find_abs_test.txt");
  std::ofstream file(tmp_file);
  ASSERT_TRUE(file.is_open()) << tmp_file;
  file << "test";
  file.close();
  ASSERT_TRUE(file) << "Failed to write to temp file: " << tmp_file;
  ASSERT_TRUE(tsl::Env::Default()->FileExists(tmp_file).ok());

  char* resolved_tmp_cstr = realpath(tmp_file.c_str(), nullptr);
  ASSERT_NE(resolved_tmp_cstr, nullptr) << "realpath failed for " << tmp_file;
  std::string expected_absolute_path(resolved_tmp_cstr);
  free(resolved_tmp_cstr);

  EXPECT_THAT(FindRegistryFile(tmp_file), IsOkAndHolds(expected_absolute_path));
}

TEST_F(GenerateBenchmarkMatricesTest,
       FindRegistryFileAbsolutePathDoesNotExist) {
  std::string non_existent_absolute_path =
      "/absolute/path/that/definitely/does/not/exist.textproto";
  EXPECT_THAT(FindRegistryFile(non_existent_absolute_path),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(GenerateBenchmarkMatricesTest,
       FindRegistryFileRelativePathExistsInWorkspace) {
  std::string temp_dir = tsl::testing::TmpDir();
  SetEnvVar("BUILD_WORKSPACE_DIRECTORY", temp_dir);

  std::string relative_path = "relative_in_workspace.textproto";
  std::string full_path_in_ws = tsl::io::JoinPath(temp_dir, relative_path);
  std::ofstream file(full_path_in_ws);
  ASSERT_TRUE(file.is_open());
  file << "test content";
  file.close();
  ASSERT_TRUE(file);
  ASSERT_TRUE(tsl::Env::Default()->FileExists(full_path_in_ws).ok());

  char* resolved_full_cstr = realpath(full_path_in_ws.c_str(), nullptr);
  ASSERT_NE(resolved_full_cstr, nullptr);
  std::string expected_absolute_path(resolved_full_cstr);
  free(resolved_full_cstr);

  EXPECT_THAT(FindRegistryFile(relative_path),
              IsOkAndHolds(expected_absolute_path));
}

TEST_F(GenerateBenchmarkMatricesTest,
       FindRegistryFileRelativePathDoesNotExist) {
  std::string non_existent_relative = "i_dont_exist_anywhere.textproto";
  unsetenv("BUILD_WORKSPACE_DIRECTORY");

  EXPECT_THAT(FindRegistryFile(non_existent_relative),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("NOT_FOUND: Registry file")));
}

TEST_F(GenerateBenchmarkMatricesTest, FindRegistryFileIsEmpty) {
  EXPECT_THAT(FindRegistryFile(""),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Registry file path cannot be empty")));
}

}  // namespace
}  // namespace benchmarks
}  // namespace tools
}  // namespace xla
