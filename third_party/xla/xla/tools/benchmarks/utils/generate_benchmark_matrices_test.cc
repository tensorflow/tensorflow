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

// Test fixture for managing temporary files and environment variables.
class GenerateBenchmarkMatricesTest : public testing::Test {
 protected:
  // Creates the specific registry file content provided.
  std::string CreateSpecificRegistryContent() {
    // Directly use the provided registry content
    return R"(
        configs: [
        {
        name: "gemma3_1b_flax_call"
        description: "Benchmarks Gemma3 1b in Flax using B200 GPUs."
        owner: "juliagmt-google@"
        hlo_gcs_bucket_path: "https://storage.googleapis.com/xla-benchmarking-temp/gemma3_1b_flax_call.hlo"
        model_source_info: ["Gemma3 1B"]
        hardware_category: GPU_B200
        topology: { num_hosts: 1, num_devices_per_host: 1 }
        run_frequencies: [POSTSUBMIT]
        update_frequency_policy: QUARTERLY
        runtime_flags: ["--num_repeat=5"]
        github_labels: ["blocking_presubmit_test"] # Github label for presubmit triggering
        },
        {
        name: "gemma2_2b_keras_jax"
        description: "Gemma2 2B in Keras on x86 CPU."
        owner: "company-A@"
        hlo_path: "benchmarks/hlo/gemma2_2b_keras_jax.hlo"
        model_source_info: ["Gemma2 2B"]
        hardware_category: CPU_X86
        topology: { num_hosts: 1, num_devices_per_host: 1 }
        target_metrics: [CPU_TIME, PEAK_CPU_MEMORY] # Note: PEAK_CPU_MEMORY isn't in the default maps
        run_frequencies: [PRESUBMIT, POSTSUBMIT]
        update_frequency_policy: QUARTERLY
        runtime_flags: ["--num_repeat=5"]
        github_labels: ["blocking_presubmit_test"] # Github label for presubmit triggering
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
  }

 private:
  std::vector<std::string> env_vars_to_clear_;
};

// --- ParseRegistry Tests ---

TEST_F(GenerateBenchmarkMatricesTest, ParseRegistrySpecificContentSuccess) {
  std::string content = CreateSpecificRegistryContent();
  std::string filepath = CreateTempRegistryFile(content, "specific_registry");
  ASSERT_FALSE(filepath.empty());

  TF_ASSERT_OK_AND_ASSIGN(xla::BenchmarkSuite suite, ParseRegistry(filepath));
  // Verify basic structure parsed correctly.
  EXPECT_EQ(suite.configs_size(), 2);
  EXPECT_EQ(suite.configs(0).name(), "gemma3_1b_flax_call");
  EXPECT_EQ(suite.configs(0).hardware_category(), HardwareCategory::GPU_B200);
  EXPECT_EQ(suite.configs(0).run_frequencies_size(), 1);
  EXPECT_EQ(suite.configs(0).run_frequencies(0), RunFrequency::POSTSUBMIT);
  EXPECT_EQ(suite.configs(0).hlo_gcs_bucket_path(),
            "https://storage.googleapis.com/xla-benchmarking-temp/"
            "gemma3_1b_flax_call.hlo");
  EXPECT_TRUE(
      suite.configs(0).hlo_path().empty());  // Check it wasn't misparsed

  EXPECT_EQ(suite.configs(1).name(), "gemma2_2b_keras_jax");
  EXPECT_EQ(suite.configs(1).hardware_category(), HardwareCategory::CPU_X86);
  EXPECT_EQ(suite.configs(1).run_frequencies_size(), 2);
  EXPECT_EQ(suite.configs(1).run_frequencies(0), RunFrequency::PRESUBMIT);
  EXPECT_EQ(suite.configs(1).run_frequencies(1), RunFrequency::POSTSUBMIT);
  EXPECT_EQ(suite.configs(1).hlo_path(),
            "benchmarks/hlo/gemma2_2b_keras_jax.hlo");
  EXPECT_TRUE(suite.configs(1).hlo_gcs_bucket_path().empty());
  // Check explicitly defined target metrics were parsed.
  ASSERT_EQ(suite.configs(1).target_metrics_size(), 2);
  EXPECT_EQ(suite.configs(1).target_metrics(0), TargetMetric::CPU_TIME);
  EXPECT_EQ(suite.configs(1).target_metrics(1), TargetMetric::PEAK_CPU_MEMORY);
}

// Other ParseRegistry tests (FileNotFound, InvalidFormat) remain valid.

TEST_F(GenerateBenchmarkMatricesTest, ParseRegistryFileNotFound) {
  std::string non_existent_path = "/path/does/not/exist/registry.textproto";
  EXPECT_THAT(ParseRegistry(non_existent_path),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("Registry file not found at")));
}

TEST_F(GenerateBenchmarkMatricesTest, ParseRegistryInvalidFormat) {
  std::string content =
      "configs: [ { name: \"Incomplete";  // Invalid proto text.
  std::string filepath = CreateTempRegistryFile(content);
  ASSERT_FALSE(filepath.empty());

  EXPECT_THAT(ParseRegistry(filepath),
              StatusIs(absl::StatusCode::kInternal,  // Or kInvalidArgument
                       HasSubstr("Error parsing TextProto registry file")));
}

TEST_F(GenerateBenchmarkMatricesTest, GenerateMatrixReturnsUnimplemented) {
  BenchmarkSuite suite;
  EXPECT_THAT(GenerateMatrix(suite),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(GenerateBenchmarkMatricesTest, ResolveRegistryPathReturnsAbsolutePath) {
  // Create a temporary file for testing
  std::string tmp_file =
      tsl::io::JoinPath(tsl::testing::TmpDir(), "test_file.txt");
  std::ofstream file(tmp_file);
  ASSERT_TRUE(file.is_open());
  file.close();

  EXPECT_THAT(ResolveRegistryPath(tmp_file), IsOkAndHolds(tmp_file));
}

TEST_F(GenerateBenchmarkMatricesTest,
       ResolveRegistryPathAbsolutePathDoesNotExist) {
  std::string tmp_file =
      tsl::io::JoinPath(tsl::testing::TmpDir(), "non_existent_file.txt");
  EXPECT_THAT(
      ResolveRegistryPath(tmp_file),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("Absolute registry path specified but not found")));
}

TEST_F(GenerateBenchmarkMatricesTest, RelativePathExistsInWorkspace) {
  // Temporarily set BUILD_WORKSPACE_DIRECTORY if not already set
  char* original_workspace_dir = std::getenv("BUILD_WORKSPACE_DIRECTORY");
  if (original_workspace_dir == nullptr) {
    setenv("BUILD_WORKSPACE_DIRECTORY", tsl::testing::TmpDir().c_str(), 1);
  }

  // Create a temporary file in the expected relative path
  const char* build_workspace_dir_cstr =
      std::getenv("BUILD_WORKSPACE_DIRECTORY");
  ASSERT_NE(build_workspace_dir_cstr, nullptr);

  std::string relative_path = "test_relative_file.txt";
  std::string tmp_file =
      tsl::io::JoinPath(build_workspace_dir_cstr, relative_path);
  std::ofstream file(tmp_file);
  ASSERT_TRUE(file.is_open());
  file.close();

  EXPECT_THAT(ResolveRegistryPath(relative_path), IsOkAndHolds(tmp_file));

  // Restore the original value
  if (original_workspace_dir == nullptr) {
    unsetenv("BUILD_WORKSPACE_DIRECTORY");
  }
}

TEST_F(GenerateBenchmarkMatricesTest, RelativePathDoesNotExist) {
  std::string relative_path = "non_existent_relative_file.txt";
  EXPECT_THAT(
      ResolveRegistryPath(relative_path),
      StatusIs(
          absl::StatusCode::kFailedPrecondition,
          HasSubstr("not found. Tried absolute and relative to workspace")));
}

TEST_F(GenerateBenchmarkMatricesTest,
       ResolveRegistryPathBuildWorkspaceDirEmptyReturnsError) {
  // Set BUILD_WORKSPACE_DIRECTORY to an empty string.
  SetEnvVar("BUILD_WORKSPACE_DIRECTORY", "");
  constexpr absl::string_view kRelativePath = "test_relative_file.txt";

  EXPECT_THAT(
      ResolveRegistryPath(std::string(kRelativePath)),
      StatusIs(
          absl::StatusCode::kFailedPrecondition,
          HasSubstr("not found. Tried absolute and relative to workspace")));
}

}  // namespace
}  // namespace benchmarks
}  // namespace tools
}  // namespace xla
