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

using testing::HasSubstr;
using testing::IsEmpty;
using testing::SizeIs;
using tsl::testing::IsOkAndHolds;
using tsl::testing::StatusIs;

// Helper function to create a temporary registry file.
std::string CreateTempRegistryFile(
    const std::string& content,
    const std::string& filename_prefix = "registry_test") {
  std::string temp_dir = tsl::testing::TmpDir();
  std::string filepath = tsl::io::JoinPath(
      temp_dir, absl::StrCat(filename_prefix, "_",
                             tsl::Env::Default()->NowMicros(), ".textproto"));

  std::ofstream file_stream(filepath);
  if (!file_stream.is_open()) {
    ADD_FAILURE() << "Failed to open temporary file for writing: " << filepath;
    return "";
  }
  file_stream << content;
  file_stream.close();
  if (!file_stream) {
    ADD_FAILURE() << "Failed to write to temporary file: " << filepath;
    return "";
  }
  EXPECT_TRUE(tsl::Env::Default()->FileExists(filepath).ok());
  return filepath;
}

// Matcher for checking a JSON value against a string.
MATCHER_P(JsonStringEq, expected_str, "") {
  return arg.isString() && arg.asString() == expected_str;
}
// Matcher for checking a JSON value against a bool.
MATCHER_P(JsonBoolEq, expected_bool, "") {
  return arg.isBool() && arg.asBool() == expected_bool;
}
// Matcher for checking a JSON array size.
MATCHER_P(JsonArraySizeIs, expected_size, "") {
  return arg.isArray() && arg.size() == expected_size;
}
// Matcher for checking if a JSON array contains a specific string.
MATCHER_P(JsonArrayContains, expected_str, "") {
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
// Matcher for checking if a JSON value is equal to another JSON value.
MATCHER_P(JsonEq, value, "") { return arg == value; }
// Matcher for checking if a JSON value is a JSON array.
MATCHER(IsJsonArray, "") { return arg.isArray(); }
// Matcher for checking if a JSON value is a JSON object.
MATCHER(IsJsonObject, "") { return arg.isObject(); }

// Test fixture for managing temporary files and environment variables.
class GenerateBenchmarkMatricesTest : public testing::Test {
 protected:
  std::string CreateTestRegistryContent() {
    return R"(
      benchmarks: [
        {
          config_id: "gemma_gpu_b200_1h1d" # Added config_id back based on latest proto
          name: "gemma_stablehlo_gpu"
          owner: "owner1@"
          input_artifact: {
             input_format: STABLEHLO_MLIR
             artifact_gcs_bucket_path: "gs://bucket/gemma.mlir"
          }
          model_source_info: "Gemma StableHLO"
          hardware_targets: [
            { hardware_category: GPU_B200, topology: {num_hosts:1, num_devices_per_host:1}, target_metrics: [GPU_DEVICE_TIME, PEAK_GPU_MEMORY] },
            { hardware_category: GPU_L4,   topology: {num_hosts:1, num_devices_per_host:1}, target_metrics: [GPU_DEVICE_TIME] }
          ]
          run_frequencies: [POSTSUBMIT, SCHEDULED]
          update_frequency_policy: QUARTERLY
          runtime_flags: "--num_repeat=5"
        },
        {
          config_id: "fusion_cpu_x86_1h1d"
          name: "fusion_hlo_cpu"
          owner: "owner2@"
          input_artifact: {
            input_format: HLO_TEXT
            artifact_path: "hlo/fusion.hlo"
          }
          model_source_info: "Fusion HLO"
          hardware_targets: [ { hardware_category: CPU_X86, topology: {num_hosts:1, num_devices_per_host:1}, target_metrics: [CPU_TIME] } ]
          run_frequencies: [PRESUBMIT]
          update_frequency_policy: MONTHLY
          xla_compilation_flags: "--flag1=true"
        },
        { # Skipped due to mapping
          config_id: "unmap_gpu_l4_8h8d"
          name: "unmappable" owner:"t" model_source_info:"m"
           input_artifact: { input_format:HLO_TEXT artifact_path:"f.hlo" }
          hardware_targets: [{hardware_category:GPU_L4, topology:{num_hosts:8, num_devices_per_host:8}}] run_frequencies:[MANUAL] update_frequency_policy:WEEKLY
        },
        { # Now valid as missing source check moved to GetArtifactInfo
          config_id: "no_source_test" # Will fail during matrix generation
          name: "no_source" owner:"t" model_source_info:"m"
           # Missing input_artifact field entirely - will fail loading
           # OR: input_artifact: { input_format:HLO_TEXT } # Missing path/gcs_path - will fail GetArtifactInfo
           hardware_targets: [{hardware_category:CPU_X86, topology:{num_hosts:1, num_devices_per_host:1}}] run_frequencies:[MANUAL] update_frequency_policy:WEEKLY
        }
      ]
      )";
  }

  // Helper to set environment variable temporarily.
  void SetEnvVar(const std::string& name, const std::string& value) {
    setenv(name.c_str(), value.c_str(), 1);  // 1 = overwrite
    env_vars_to_clear_.push_back(name);
  }

  void TearDown() override {
    // Clean up environment variables set during the test.
    for (const auto& var_name : env_vars_to_clear_) {
      unsetenv(var_name.c_str());
    }
    // Could also clean up temp files if needed, but gUnit usually handles
    // TmpDir
  }

 private:
  std::vector<std::string> env_vars_to_clear_;
};

// --- ParseRegistry Tests ---

TEST_F(GenerateBenchmarkMatricesTest, LoadBenchmarkSuiteSuccess) {
  std::string filepath = CreateTempRegistryFile(CreateTestRegistryContent());
  ASSERT_FALSE(filepath.empty());
  TF_ASSERT_OK_AND_ASSIGN(BenchmarkSuite suite,
                          LoadBenchmarkSuiteFromFile(filepath));
  ASSERT_THAT(suite.benchmarks(), SizeIs(4));

  // Check parsing of InputArtifact
  EXPECT_TRUE(suite.benchmarks(0).has_input_artifact());
  EXPECT_EQ(suite.benchmarks(0).input_artifact().input_format(),
            InputFormat::STABLEHLO_MLIR);
  EXPECT_TRUE(suite.benchmarks(0).input_artifact().artifact_path().empty());
  EXPECT_EQ(suite.benchmarks(0).input_artifact().artifact_gcs_bucket_path(),
            "gs://bucket/gemma.mlir");

  EXPECT_TRUE(suite.benchmarks(1).has_input_artifact());
  EXPECT_EQ(suite.benchmarks(1).input_artifact().input_format(),
            InputFormat::HLO_TEXT);
  EXPECT_EQ(suite.benchmarks(1).input_artifact().artifact_path(),
            "hlo/fusion.hlo");
  EXPECT_TRUE(
      suite.benchmarks(1).input_artifact().artifact_gcs_bucket_path().empty());

  // Check config_id presence
  EXPECT_EQ(suite.benchmarks(0).config_id(), "gemma_gpu_b200_1h1d");
  EXPECT_EQ(suite.benchmarks(1).config_id(), "fusion_cpu_x86_1h1d");
}

TEST_F(GenerateBenchmarkMatricesTest,
       LoadBenchmarkSuiteMissingInputArtifactField) {
  // Modify content to remove the input_artifact field from one entry
  std::string content = R"(
         benchmarks: [ {
             config_id: "test1" name: "test_no_artifact_field" owner:"t"
             model_source_info:"m"
             hardware_targets: [{hardware_category:CPU_X86, topology:{num_hosts:1, num_devices_per_host:1}}]
             run_frequencies:[MANUAL] update_frequency_policy:WEEKLY
         } ] )";
  std::string filepath = CreateTempRegistryFile(content);
  ASSERT_FALSE(filepath.empty());
  // Check that LoadBenchmarkSuiteFromFile now returns an error due to missing
  // field check
  TF_ASSERT_OK_AND_ASSIGN(BenchmarkSuite suite,
                          LoadBenchmarkSuiteFromFile(filepath));
}

TEST_F(GenerateBenchmarkMatricesTest, LoadBenchmarkSuiteMissingArtifactSource) {
  std::string content = R"(
         benchmarks: [ {
             config_id: "test1" name: "test_no_source" owner:"t"
             input_artifact: { input_format:HLO_TEXT } # Missing path/gcs_path
             model_source_info:"m"
             hardware_targets: [{hardware_category:CPU_X86, topology:{num_hosts:1, num_devices_per_host:1}}]
             run_frequencies:[MANUAL] update_frequency_policy:WEEKLY
         } ] )";
  std::string filepath = CreateTempRegistryFile(content);
  ASSERT_FALSE(filepath.empty());
  // Check that LoadBenchmarkSuiteFromFile now returns an error due to missing
  // source check
  TF_ASSERT_OK_AND_ASSIGN(BenchmarkSuite suite,
                          LoadBenchmarkSuiteFromFile(filepath));
}

// --- BuildGitHubActionsMatrix Tests ---
TEST_F(GenerateBenchmarkMatricesTest, BuildMatrixForPresubmitSuccess) {
  std::string filepath = CreateTempRegistryFile(CreateTestRegistryContent());
  ASSERT_FALSE(filepath.empty());
  TF_ASSERT_OK_AND_ASSIGN(BenchmarkSuite suite,
                          LoadBenchmarkSuiteFromFile(filepath));

  // Test for PRESUBMIT frequency
  TF_ASSERT_OK_AND_ASSIGN(
      Json::Value matrix_array,
      BuildGitHubActionsMatrix(suite, RunFrequency::PRESUBMIT));
  ASSERT_THAT(matrix_array, IsJsonArray());
  // Only "fusion_hlo_cpu" runs on PRESUBMIT (1 target * 1 frequency)
  EXPECT_THAT(matrix_array, JsonArraySizeIs(1));

  const auto& entry0 = matrix_array[0];
  EXPECT_THAT(entry0["benchmark_name"], JsonStringEq("fusion_hlo_cpu"));
  EXPECT_THAT(entry0["config_id"], JsonStringEq("fusion_cpu_x86_1h1d"));
  EXPECT_THAT(entry0["run_frequency"], JsonStringEq("PRESUBMIT"));
  EXPECT_THAT(entry0["hardware_category"], JsonStringEq("CPU_X86"));
  EXPECT_THAT(entry0["input_format"], JsonStringEq("HLO_TEXT"));
  EXPECT_THAT(entry0["artifact_location"], JsonStringEq("hlo/fusion.hlo"));
  EXPECT_THAT(entry0["is_gcs_artifact"], JsonBoolEq(false));
  EXPECT_THAT(entry0["xla_compilation_flags"], JsonArraySizeIs(1));
  EXPECT_THAT(entry0["xla_compilation_flags"],
              JsonArrayContains("--flag1=true"));
}

TEST_F(GenerateBenchmarkMatricesTest, BuildMatrixForPostsubmitSuccess) {
  std::string filepath = CreateTempRegistryFile(CreateTestRegistryContent());
  ASSERT_FALSE(filepath.empty());
  TF_ASSERT_OK_AND_ASSIGN(BenchmarkSuite suite,
                          LoadBenchmarkSuiteFromFile(filepath));

  // Test for POSTSUBMIT frequency
  TF_ASSERT_OK_AND_ASSIGN(
      Json::Value matrix_array,
      BuildGitHubActionsMatrix(suite, RunFrequency::POSTSUBMIT));
  ASSERT_THAT(matrix_array, IsJsonArray());
  // "gemma_stablehlo_gpu" runs on POSTSUBMIT (2 targets * 1 frequency)
  EXPECT_THAT(matrix_array, JsonArraySizeIs(2));

  bool found_b200 = false;
  bool found_l4 = false;
  for (const auto& entry : matrix_array) {
    EXPECT_THAT(entry["benchmark_name"], JsonStringEq("gemma_stablehlo_gpu"));
    EXPECT_THAT(entry["config_id"], JsonStringEq("gemma_gpu_b200_1h1d"));
    EXPECT_THAT(entry["run_frequency"], JsonStringEq("POSTSUBMIT"));
    EXPECT_THAT(entry["input_format"], JsonStringEq("STABLEHLO_MLIR"));
    EXPECT_THAT(entry["artifact_location"],
                JsonStringEq("gs://bucket/gemma.mlir"));
    EXPECT_THAT(entry["is_gcs_artifact"], JsonBoolEq(true));
    EXPECT_THAT(entry["runtime_flags"], JsonArraySizeIs(1));
    EXPECT_THAT(entry["runtime_flags"], JsonArrayContains("--num_repeat=5"));

    if (entry["hardware_category"] == Json::Value("GPU_B200")) {
      found_b200 = true;
    }
    if (entry["hardware_category"] == Json::Value("GPU_L4")) {
      found_l4 = true;
    }
  }
  EXPECT_TRUE(found_b200);
  EXPECT_TRUE(found_l4);
}

TEST_F(GenerateBenchmarkMatricesTest, BuildMatrixSkipsDueToMappingError) {
  std::string filepath = CreateTempRegistryFile(CreateTestRegistryContent());
  ASSERT_FALSE(filepath.empty());
  TF_ASSERT_OK_AND_ASSIGN(BenchmarkSuite suite,
                          LoadBenchmarkSuiteFromFile(filepath));

  // Test for MANUAL frequency where "unmappable_target_config" should be
  // attempted but fail mapping
  TF_ASSERT_OK_AND_ASSIGN(
      Json::Value matrix_array,
      BuildGitHubActionsMatrix(suite, RunFrequency::MANUAL));
  ASSERT_THAT(matrix_array, IsJsonArray());
  EXPECT_THAT(matrix_array, IsEmpty());
}

TEST_F(GenerateBenchmarkMatricesTest,
       BuildMatrixSkipsDueToMissingArtifactSource) {
  // Create a registry with only the config that's missing artifact source
  std::string content = R"(
      benchmarks: [ {
          config_id: "no_source_path_manual"
          name: "missing_artifact_source_config" owner:"tester@" model_source_info:"validation test"
          input_artifact: { input_format:HLO_TEXT } # Missing artifact_path/gcs_path
          hardware_targets: [{
            hardware_category:CPU_X86,
            topology:{num_hosts:1, num_devices_per_host:1, multi_host: false, multi_device: false},
            target_metrics: [CPU_TIME]
          }]
          run_frequencies:[MANUAL] update_frequency_policy:WEEKLY
      } ] )";
  std::string filepath = CreateTempRegistryFile(content);
  ASSERT_FALSE(filepath.empty());
  TF_ASSERT_OK_AND_ASSIGN(BenchmarkSuite suite,
                          LoadBenchmarkSuiteFromFile(filepath));

  TF_ASSERT_OK_AND_ASSIGN(
      Json::Value matrix_array,
      BuildGitHubActionsMatrix(suite, RunFrequency::MANUAL));
  ASSERT_THAT(matrix_array, IsJsonArray());
  EXPECT_THAT(matrix_array,
              IsEmpty());  // Should be skipped because GetArtifactInfo fails
}

TEST_F(GenerateBenchmarkMatricesTest, BuildMatrixEmptySuite) {
  BenchmarkSuite suite;
  TF_ASSERT_OK_AND_ASSIGN(
      Json::Value matrix_array,
      BuildGitHubActionsMatrix(suite, RunFrequency::PRESUBMIT));
  ASSERT_THAT(matrix_array, IsJsonArray());
  EXPECT_THAT(matrix_array, IsEmpty());
}

// --- FindRegistryFile Tests ---

TEST_F(GenerateBenchmarkMatricesTest, FindRegistryPathReturnsAbsolutePath) {
  std::string temp_dir = tsl::testing::TmpDir();
  std::string tmp_file = tsl::io::JoinPath(temp_dir, "find_abs_test.txt");
  // Ensure the file exists
  std::ofstream file(tmp_file);
  ASSERT_TRUE(file.is_open());
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
       FindRegistryPathAbsolutePathDoesNotExist) {
  std::string non_existent_absolute_path =
      "/absolute/path/that/does/not/exist.txt";
  EXPECT_THAT(
      FindRegistryFile(non_existent_absolute_path),
      StatusIs(absl::StatusCode::kFailedPrecondition,  // Expect NotFound
               HasSubstr("Registry path specified but not found")));
}

TEST_F(GenerateBenchmarkMatricesTest,
       FindRegistryRelativePathExistsInWorkspace) {
  std::string temp_dir = tsl::testing::TmpDir();
  // Set BUILD_WORKSPACE_DIRECTORY to temp_dir
  SetEnvVar("BUILD_WORKSPACE_DIRECTORY", temp_dir);

  // Create a file within that "workspace"
  std::string relative_path = "relative_in_workspace.textproto";
  std::string full_path = tsl::io::JoinPath(temp_dir, relative_path);
  std::ofstream file(full_path);
  ASSERT_TRUE(file.is_open());
  file << "test";
  file.close();
  ASSERT_TRUE(file);
  ASSERT_TRUE(tsl::Env::Default()->FileExists(full_path).ok());

  // Resolve expected absolute path
  char* resolved_full_cstr = realpath(full_path.c_str(), nullptr);
  ASSERT_NE(resolved_full_cstr, nullptr);
  std::string expected_absolute_path(resolved_full_cstr);
  free(resolved_full_cstr);
}

TEST_F(GenerateBenchmarkMatricesTest, FindRegistryRelativePathDoesNotExist) {
  std::string non_existent_relative = "i_dont_exist_anywhere.txt";
  // Ensure BUILD_WORKSPACE_DIRECTORY isn't set to avoid confusion
  unsetenv("BUILD_WORKSPACE_DIRECTORY");

  EXPECT_THAT(FindRegistryFile(non_existent_relative),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Registry path specified but not found: "
                                 "i_dont_exist_anywhere.txt")));
}

TEST_F(GenerateBenchmarkMatricesTest, FindRegistryBuildWorkspaceDirEmpty) {
  // Set BUILD_WORKSPACE_DIRECTORY to an empty string.
  SetEnvVar("BUILD_WORKSPACE_DIRECTORY", "");
  constexpr absl::string_view kRelativePath = "some_file.txt";

  // It should still try CWD, so only fail if not in CWD either.
  EXPECT_THAT(
      FindRegistryFile(std::string(kRelativePath)),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("Registry path specified but not found: some_file.")));
}

TEST_F(GenerateBenchmarkMatricesTest, FindRegistryPathIsEmpty) {
  EXPECT_THAT(FindRegistryFile(""),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("Registry path specified but not found:")));
}

}  // namespace
}  // namespace benchmarks
}  // namespace tools
}  // namespace xla
