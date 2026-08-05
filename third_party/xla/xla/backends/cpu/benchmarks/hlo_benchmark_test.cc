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

// Microbenchmarks to run an individual HLO module.

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "benchmark/benchmark.h"
#include "xla/backends/cpu/benchmarks/aot_benchmark_helper.h"
#include "xla/backends/cpu/benchmarks/hlo_benchmark_runner.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/literal.h"
#include "xla/service/hlo.pb.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"
#include "tsl/platform/stacktrace_handler.h"

ABSL_FLAG(std::vector<std::string>, hlo_paths, std::vector<std::string>({}),
          "Comma-separated list of paths to HLO modules to benchmark");

ABSL_FLAG(
    bool, hlo_paths_absolute, false,
    "If true the HLO paths are absolute. Otherwise the paths are relative and "
    "the corresponding files must be in the data dependencies of the test. "
    "A relative path is interpreted first as relative to the test workspace "
    "directory; if the resulting path does not exist, the given path is "
    "interpreted as relative to the test source directory.");

ABSL_FLAG(int32_t, num_executions, 1,
          "Number of times to execute the HLO within a single benchmark "
          "iteration. By overlapping multiple independent execution we can "
          "measure how well XLA runtime handles concurrent requests, which is "
          "similar to production inference workloads.");

ABSL_FLAG(bool, aot_compiled_execution, false,
          "If true, when running the benchmark, the HLO will be compiled AOT.");

ABSL_FLAG(std::string, xla_flags, "", "Flags to append to XLA_FLAGS");

namespace xla::cpu {

namespace {

void Set_XLA_FLAGS() {
  const char* env_xla_flags = std::getenv("XLA_FLAGS");
  std::string xla_flags = absl::StrCat(env_xla_flags ? env_xla_flags : "",
                                       absl::GetFlag(FLAGS_xla_flags));
  tsl::setenv("XLA_FLAGS", xla_flags.data(), /*overwrite=*/1);
}

std::string GetHloAbsolutePath(absl::string_view hlo_path) {
  if (absl::GetFlag(FLAGS_hlo_paths_absolute)) {
    return std::string(hlo_path);
  }
  std::string workspace_dir;
  if (tsl::io::GetTestWorkspaceDir(&workspace_dir)) {
    std::string path = tsl::io::JoinPath(workspace_dir, hlo_path);
    if (tsl::Env::Default()->FileExists(path).ok()) {
      return path;
    }
  }
  return tsl::io::JoinPath(::testing::SrcDir(), hlo_path);
}

absl::Status RunBenchmark(benchmark::State* absl_nullable state,
                          absl::string_view hlo_path) {
  std::string hlo_abs_path = GetHloAbsolutePath(hlo_path);

  HloBenchmarkOptions benchmark_options;
  benchmark_options.num_executions = absl::GetFlag(FLAGS_num_executions);
  benchmark_options.aot_options = absl::GetFlag(FLAGS_aot_compiled_execution)
                                      ? GetAotCompilationOptions()
                                      : nullptr;

  ASSIGN_OR_RETURN(auto module_and_iteration_literals,
                   LoadHloModuleAndMaybeIterationLiterals(hlo_abs_path));

  std::unique_ptr<HloModule> hlo_module =
      std::move(module_and_iteration_literals.first);

  std::vector<Literal> args;
  args.reserve(module_and_iteration_literals.second->arguments_size());
  for (const auto& arg : module_and_iteration_literals.second->arguments()) {
    ASSIGN_OR_RETURN(args.emplace_back(), Literal::CreateFromProto(arg));
  }

  std::vector<Literal*> arg_ptrs;
  arg_ptrs.reserve(args.size());
  for (auto& arg : args) {
    arg_ptrs.push_back(&arg);
  }

  if (state) {
    return RunHloBenchmark(*state, std::move(hlo_module), arg_ptrs,
                           benchmark_options);
  }
  return RunHloBenchmarkOnce(std::move(hlo_module), arg_ptrs,
                             benchmark_options);
}

void BM_HloModule(benchmark::State& state, absl::string_view hlo_path) {
  CHECK_OK(RunBenchmark(&state, hlo_path));
}

void BM_CompileHloModule(benchmark::State& state, absl::string_view hlo_path) {
  std::string hlo_abs_path = GetHloAbsolutePath(hlo_path);

  HloBenchmarkOptions benchmark_options;
  benchmark_options.aot_options = absl::GetFlag(FLAGS_aot_compiled_execution)
                                      ? GetAotCompilationOptions()
                                      : nullptr;

  ASSERT_OK_AND_ASSIGN(auto module_and_iteration_literals,
                       LoadHloModuleAndMaybeIterationLiterals(hlo_abs_path));

  std::unique_ptr<HloModule> hlo_module =
      std::move(module_and_iteration_literals.first);

  CHECK_OK(
      CompileHloBenchmark(state, std::move(hlo_module), benchmark_options));
}

void RegisterBenchmarks() {
  std::vector<std::string> hlo_paths = absl::GetFlag(FLAGS_hlo_paths);

  for (const std::string& path : hlo_paths) {
    benchmark::RegisterBenchmark(
        absl::StrCat("BM_HloModule/", tsl::io::BasenamePrefix(path)),
        BM_HloModule, path)
        ->MeasureProcessCPUTime();
  }
  for (const std::string& path : hlo_paths) {
    benchmark::RegisterBenchmark(
        absl::StrCat("BM_CompileHloModule/", tsl::io::BasenamePrefix(path)),
        BM_CompileHloModule, path)
        ->MeasureProcessCPUTime();
  }
}

TEST(HloBenchmarkTest, RunBenchmarks) {
  std::vector<std::string> hlo_paths = absl::GetFlag(FLAGS_hlo_paths);
  for (const std::string& path : hlo_paths) {
    SCOPED_TRACE(absl::StrCat("Benchmark: ", path));
    EXPECT_OK(RunBenchmark(/*state=*/nullptr, path));
  }
}

}  // namespace

}  // namespace xla::cpu

GTEST_API_ int main(int argc, char** argv) {
  tsl::testing::InstallStacktraceHandler();
  ::benchmark::Initialize(&argc, argv);
  testing::InitGoogleTest(&argc, argv);
  absl::ParseCommandLine(argc, argv);
  xla::cpu::Set_XLA_FLAGS();
  xla::cpu::RegisterBenchmarks();
  if (::benchmark::GetBenchmarkFilter().empty()) {
    return RUN_ALL_TESTS();
  }
  ::benchmark::RunSpecifiedBenchmarks();
  return 0;
}
