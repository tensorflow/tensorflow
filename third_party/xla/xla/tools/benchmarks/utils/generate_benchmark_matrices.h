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

#ifndef XLA_TOOLS_BENCHMARKS_UTILS_GENERATE_BENCHMARK_MATRICES_H_
#define XLA_TOOLS_BENCHMARKS_UTILS_GENERATE_BENCHMARK_MATRICES_H_

#include <string>

#include "absl/status/statusor.h"
#include "json/json.h"
#include "xla/tools/benchmarks/proto/benchmark_config.pb.h"

namespace xla {
namespace tools {
namespace benchmarks {

using BenchmarkSuite = xla::BenchmarkSuite;

// Parses a TextProto registry file into a BenchmarkSuite proto.
// Returns an error if parsing fails (file not found, parse error).
absl::StatusOr<BenchmarkSuite> LoadBenchmarkSuiteFromFile(
    const std::string& registry_path);

// Generates the benchmark matrix JSON object based on the run frequency
// (e.g., presubmit, postsubmit, nightly, manual).
// Returns an empty JSON value object if the suite is empty or errors occur
// during generation (though errors are primarily handled by printing
// warnings).
absl::StatusOr<Json::Value> BuildGitHubActionsMatrix(
    const BenchmarkSuite& suite, WorkflowType current_workflow_type);

// Attempts to find the absolute path to the registry file.
// Checks the provided path directly, then relative to BUILD_WORKSPACE_DIRECTORY
// (if set), and finally relative to the current working directory.
absl::StatusOr<std::string> FindRegistryFile(const std::string& registry_path);

}  // namespace benchmarks
}  // namespace tools
}  // namespace xla

#endif  // XLA_TOOLS_BENCHMARKS_UTILS_GENERATE_BENCHMARK_MATRICES_H_
