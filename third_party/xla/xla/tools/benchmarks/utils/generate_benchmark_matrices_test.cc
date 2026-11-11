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

#include <fstream>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "json/json.h"
#include "xla/tools/benchmarks/proto/benchmark_config.pb.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"

namespace xla {
namespace tools {
namespace benchmarks {
namespace {

using testing::HasSubstr;
using testing::SizeIs;
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

// --- ParseRegistry Tests (Using Specific Content) ---

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

// --- ResolveRegistryPath Tests ---

TEST_F(GenerateBenchmarkMatricesTest, ResolveAbsolutePathSuccess) {
  std::string content = "configs { name: \"Test\" }";
  std::string filepath = CreateTempRegistryFile(content);
  ASSERT_FALSE(filepath.empty());
  EXPECT_TRUE(tsl::io::IsAbsolutePath(filepath));

  TF_ASSERT_OK_AND_ASSIGN(std::string resolved_path,
                          ResolveRegistryPath(filepath));
  EXPECT_EQ(resolved_path, filepath);
}

TEST_F(GenerateBenchmarkMatricesTest, ResolveAbsolutePathNotFound) {
  std::string temp_file_path = CreateTempRegistryFile("delete me");
  ASSERT_TRUE(tsl::Env::Default()->DeleteFile(temp_file_path).ok());
  ASSERT_FALSE(tsl::Env::Default()->FileExists(temp_file_path).ok());

  EXPECT_THAT(
      ResolveRegistryPath(temp_file_path),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("Absolute registry path specified but not found")));
}

TEST_F(GenerateBenchmarkMatricesTest, ResolveRelativePathWorkspaceSuccess) {
  std::string workspace_dir = tsl::testing::TmpDir();
  std::string relative_path = "my_subdir/registry.textproto";
  std::string full_path = tsl::io::JoinPath(workspace_dir, relative_path);

  ASSERT_TRUE(
      tsl::Env::Default()
          ->RecursivelyCreateDir(tsl::io::JoinPath(workspace_dir, "my_subdir"))
          .ok());
  std::string content = "configs { name: \"WorkspaceTest\" }";
  {
    std::ofstream file_stream(full_path);
    ASSERT_TRUE(file_stream.is_open());
    file_stream << content;
  }
  ASSERT_TRUE(tsl::Env::Default()->FileExists(full_path).ok());

  SetEnvVar("BUILD_WORKSPACE_DIRECTORY", workspace_dir);

  TF_ASSERT_OK_AND_ASSIGN(std::string resolved_path,
                          ResolveRegistryPath(relative_path));
  EXPECT_EQ(tsl::io::CleanPath(resolved_path), tsl::io::CleanPath(full_path));
}

TEST_F(GenerateBenchmarkMatricesTest, ResolveRelativePathWorkspaceNotFound) {
  std::string workspace_dir = tsl::testing::TmpDir();
  std::string relative_path = "non_existent_registry.textproto";
  SetEnvVar("BUILD_WORKSPACE_DIRECTORY", workspace_dir);

  EXPECT_THAT(
      ResolveRegistryPath(relative_path),
      StatusIs(
          absl::StatusCode::kFailedPrecondition,
          HasSubstr(
              "Registry file 'non_existent_registry.textproto' not found")));
}

TEST_F(GenerateBenchmarkMatricesTest, ResolveRelativePathNoWorkspaceNotFound) {
  std::string relative_path = "another_non_existent_registry.textproto";
  unsetenv("BUILD_WORKSPACE_DIRECTORY");

  EXPECT_THAT(
      ResolveRegistryPath(relative_path),
      StatusIs(
          absl::StatusCode::kFailedPrecondition,
          HasSubstr("Registry file 'another_non_existent_registry.textproto' "
                    "not found")));
}

// --- GenerateMatrix Tests (Using Specific Content) ---

TEST_F(GenerateBenchmarkMatricesTest, GenerateMatrixEmptySuite) {
  xla::BenchmarkSuite suite;  // Empty suite
  Json::Value matrix = GenerateMatrix(suite);

  ASSERT_TRUE(matrix.isObject());
  ASSERT_TRUE(matrix.isMember("include"));
  ASSERT_TRUE(matrix["include"].isArray());
  EXPECT_THAT(matrix["include"], SizeIs(0));
}

TEST_F(GenerateBenchmarkMatricesTest, GenerateMatrixFromRegistry) {
  xla::BenchmarkSuite suite;
  // Use the new function to get content matching the new proto.
  ASSERT_TRUE(tsl::protobuf::TextFormat::ParseFromString(
      CreateSpecificRegistryContent(), &suite));
  ASSERT_EQ(suite.configs_size(), 2);

  Json::Value matrix = GenerateMatrix(suite);

  ASSERT_TRUE(matrix.isObject());
  ASSERT_TRUE(matrix.isMember("include"));
  ASSERT_TRUE(matrix["include"].isArray());
  // Expected entries: 1 (Gemma3 POSTSUBMIT) + 2 (Gemma2 PRESUBMIT,
  // POSTSUBMIT) = 3.
  ASSERT_THAT(matrix["include"], SizeIs(3));

  // --- Check Gemma3 Entry (POSTSUBMIT) ---
  const Json::Value& gemma3_entry = matrix["include"][0];
  ASSERT_TRUE(gemma3_entry.isObject());

  // ** Check that config_id from proto is used. **
  ASSERT_TRUE(gemma3_entry["config_id"].isString());
  EXPECT_EQ(gemma3_entry["config_id"].asString(),
            "gemma3_1b_flax_call_b200_1h_1d");

  ASSERT_TRUE(gemma3_entry["benchmark_name"].isString());
  EXPECT_EQ(gemma3_entry["benchmark_name"].asString(), "gemma3_1b_flax_call");
  ASSERT_TRUE(gemma3_entry["run_frequency"].isString());
  EXPECT_EQ(gemma3_entry["run_frequency"].asString(), "POSTSUBMIT");
  ASSERT_TRUE(gemma3_entry["runner_label"].isString());
  EXPECT_EQ(gemma3_entry["runner_label"].asString(),
            "linux-x86-a4-224-b200-1gpu");
  ASSERT_TRUE(gemma3_entry["container_image"].isString());
  EXPECT_EQ(gemma3_entry["container_image"].asString(),
            "us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/"
            "ml-build-cuda12.8-cudnn9.8:latest");
  ASSERT_TRUE(gemma3_entry["hlo_location"].isString());
  EXPECT_EQ(gemma3_entry["hlo_location"].asString(),
            "https://storage.googleapis.com/xla-benchmarking-temp/"
            "gemma3_1b_flax_call.hlo");
  ASSERT_TRUE(gemma3_entry["is_gcs_hlo"].isBool());
  EXPECT_TRUE(gemma3_entry["is_gcs_hlo"].asBool());
  ASSERT_TRUE(gemma3_entry["required_hardware_category"].isString());
  EXPECT_EQ(gemma3_entry["required_hardware_category"].asString(), "GPU_B200");

  ASSERT_TRUE(gemma3_entry["target_metrics"].isArray());
  EXPECT_THAT(gemma3_entry["target_metrics"],
              SizeIs(2));  // Default B200 metrics.
  ASSERT_TRUE(gemma3_entry["target_metrics"][0].isString());
  EXPECT_EQ(gemma3_entry["target_metrics"][0].asString(), "GPU_DEVICE_TIME");
  ASSERT_TRUE(gemma3_entry["target_metrics"][1].isString());
  EXPECT_EQ(gemma3_entry["target_metrics"][1].asString(),
            "GPU_DEVICE_MEMCPY_TIME");

  ASSERT_TRUE(gemma3_entry["runtime_flags"].isArray());
  EXPECT_THAT(gemma3_entry["runtime_flags"], SizeIs(1));
  ASSERT_TRUE(gemma3_entry["runtime_flags"][0].isString());
  EXPECT_EQ(gemma3_entry["runtime_flags"][0].asString(), "--num_repeat=5");

  ASSERT_TRUE(gemma3_entry["github_labels"].isArray());
  EXPECT_THAT(gemma3_entry["github_labels"], SizeIs(1));
  ASSERT_TRUE(gemma3_entry["github_labels"][0].isString());
  EXPECT_EQ(gemma3_entry["github_labels"][0].asString(),
            "blocking_presubmit_test");

  // --- Check Gemma2 Entry 1 (PRESUBMIT) ---
  const Json::Value& gemma2_pre_entry = matrix["include"][1];
  EXPECT_TRUE(gemma2_pre_entry.isObject());
  EXPECT_EQ(gemma2_pre_entry["config_id"].asString(),
            "gemma2_2b_keras_jax_x86_1h_1d");
  EXPECT_EQ(gemma2_pre_entry["benchmark_name"].asString(),
            "gemma2_2b_keras_jax");
  EXPECT_EQ(gemma2_pre_entry["run_frequency"].asString(), "PRESUBMIT");
  EXPECT_EQ(gemma2_pre_entry["runner_label"].asString(), "linux-x86-n2-128");
  EXPECT_EQ(
      gemma2_pre_entry["container_image"].asString(),
      "us-central1-docker.pkg.dev/tensorflow-sigs/tensorflow/ml-build:latest");
  EXPECT_EQ(gemma2_pre_entry["hlo_location"].asString(),
            "benchmarks/hlo/gemma2_2b_keras_jax.hlo");
  EXPECT_FALSE(gemma2_pre_entry["is_gcs_hlo"].asBool());
  EXPECT_EQ(gemma2_pre_entry["required_hardware_category"].asString(),
            "CPU_X86");
  ASSERT_TRUE(gemma2_pre_entry["target_metrics"].isArray());
  // To test the proto values:
  EXPECT_THAT(gemma2_pre_entry["target_metrics"], SizeIs(2));

  EXPECT_THAT(gemma2_pre_entry["runtime_flags"], SizeIs(1));
  EXPECT_EQ(gemma2_pre_entry["runtime_flags"][0].asString(), "--num_repeat=5");

  EXPECT_THAT(gemma2_pre_entry["github_labels"], SizeIs(1));
  EXPECT_EQ(gemma2_pre_entry["github_labels"][0].asString(),
            "blocking_presubmit_test");

  // --- Check Gemma2 Entry 2 (POSTSUBMIT) ---
  const Json::Value& gemma2_post_entry = matrix["include"][2];
  EXPECT_TRUE(gemma2_post_entry.isObject());
  EXPECT_EQ(gemma2_post_entry["config_id"].asString(),
            "gemma2_2b_keras_jax_x86_1h_1d");  // Same ID.
  EXPECT_EQ(gemma2_post_entry["run_frequency"].asString(), "POSTSUBMIT");
  // Other fields should be the same as gemma2_pre_entry, except frequency.
  EXPECT_EQ(gemma2_post_entry["hlo_location"].asString(),
            "benchmarks/hlo/gemma2_2b_keras_jax.hlo");
  EXPECT_EQ(gemma2_post_entry["required_hardware_category"].asString(),
            "CPU_X86");
}

}  // namespace
}  // namespace benchmarks
}  // namespace tools
}  // namespace xla
