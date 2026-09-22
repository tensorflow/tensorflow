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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "xla/tests/xla_test_backend_predicates.h"
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/backends/profiler/simple_multipass_tracer.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_tpu/xla_tpu_pjrt_client.h"
#include "xla/tools/multihost_hlo_runner/create_client.h"
#include "xla/tools/multihost_hlo_runner/functional_hlo_runner.h"
#include "xla/tools/multihost_hlo_runner/hlo_input_output_format.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace {

std::string GetHloPath(std::string file_name) {
  return tsl::io::JoinPath(tsl::testing::XlaSrcRoot(), "tools",
                           "multihost_hlo_runner", "data", file_name);
}

absl::StatusOr<std::unique_ptr<xla::PjRtClient>> GetPjRtClient() {
  if (test::DeviceTypeIs(test::kCpu)) {
    LOG(INFO) << "Running on CPU";
    return CreateHostClient();
  }
  if (test::DeviceTypeIs(test::kTpu)) {
    LOG(INFO) << "Running on TPU";
    return GetXlaPjrtTpuClient();
  }
  LOG(INFO) << "Running on GPU";
  return CreateGpuClient({});
}

TEST(MultiPassFunctionalHloRunnerTest, SimpleMultiPassTracerTest) {
  xla::profiler::RegisterSimpleMultiPassTracer();
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          GetPjRtClient());

  // 1. Run non-multipass profiling session.
  std::string profile_dump_path_non_multipass =
      tsl::io::JoinPath(testing::TempDir(), "xspace_non_multipass.pb");
  std::unique_ptr<HLORunnerProfiler> profiler_non_multipass;
  FunctionalHloRunner::RunningOptions running_options_non_multipass;
  TF_ASSERT_OK_AND_ASSIGN(
      profiler_non_multipass,
      HLORunnerProfiler::Create(profile_dump_path_non_multipass,
                                /*keep_xspace=*/true,
                                /*enable_multipass_profiling=*/false));
  running_options_non_multipass.profiler = profiler_non_multipass.get();
  running_options_non_multipass.enable_multipass_profiling = false;

  // We must use a valid HLO file to run.
  TF_EXPECT_OK(FunctionalHloRunner::LoadAndRunAndDump(
      *client,
      /* preproc_options= */ {},
      /* raw_compile_options = */ {}, running_options_non_multipass,
      GetHloPath("single_device.hlo"), InputFormat::kText));

  const tensorflow::profiler::XSpace* xspace_non_multipass =
      profiler_non_multipass->GetXSpace();
  ASSERT_NE(xspace_non_multipass, nullptr);

  // 2. Run multipass profiling session.
  std::string profile_dump_path_multipass =
      tsl::io::JoinPath(testing::TempDir(), "xspace_multipass.pb");
  std::unique_ptr<HLORunnerProfiler> profiler_multipass;
  FunctionalHloRunner::RunningOptions running_options_multipass;
  TF_ASSERT_OK_AND_ASSIGN(profiler_multipass,
                          HLORunnerProfiler::Create(
                              profile_dump_path_multipass, /*keep_xspace=*/true,
                              /*enable_multipass_profiling=*/true));
  running_options_multipass.profiler = profiler_multipass.get();
  running_options_multipass.enable_multipass_profiling = true;

  TF_EXPECT_OK(FunctionalHloRunner::LoadAndRunAndDump(
      *client,
      /* preproc_options= */ {},
      /* raw_compile_options = */ {}, running_options_multipass,
      GetHloPath("single_device.hlo"), InputFormat::kText));

  const tensorflow::profiler::XSpace* xspace_multipass =
      profiler_multipass->GetXSpace();
  ASSERT_NE(xspace_multipass, nullptr);

  // 3. Verify that the planes collected in both modes are identical.
  std::vector<std::string> plane_names_non_multipass;
  for (const auto& plane : xspace_non_multipass->planes()) {
    VLOG(1) << "Non multipass profiling results plane: " << plane.name();
    plane_names_non_multipass.push_back(plane.name());
  }

  std::vector<std::string> plane_names_multipass;
  for (const auto& plane : xspace_multipass->planes()) {
    VLOG(1) << "Multipass profiling results plane: " << plane.name();
    plane_names_multipass.push_back(plane.name());
  }

  std::sort(plane_names_non_multipass.begin(), plane_names_non_multipass.end());
  std::sort(plane_names_multipass.begin(), plane_names_multipass.end());
  EXPECT_EQ(plane_names_non_multipass, plane_names_multipass);
}

}  // namespace
}  // namespace xla

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
