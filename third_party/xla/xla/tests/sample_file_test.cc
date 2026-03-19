/* Copyright 2017 The OpenXLA Authors.

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

// This demonstrates how to create a file-based test case and compare results
// between gpu and cpu.

#include <memory>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/test.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/service/hlo_module_util.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/hlo_runner_pjrt.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/hlo_runner_agnostic_reference_mixin.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"

namespace xla {
namespace {

std::unique_ptr<HloRunnerInterface> GetReferenceRunner() {
  absl::StatusOr<std::unique_ptr<PjRtClient>> client = GetXlaPjrtCpuClient({});
  if (!client.ok()) {
    LOG(FATAL) << "Failed to create XLA:CPU PjRtClient: " << client.status();
  }
  return std::make_unique<HloRunnerPjRt>(*std::move(client));
}

class SampleFileTest : public HloRunnerAgnosticReferenceMixin<HloPjRtTestBase> {
 protected:
  SampleFileTest()
      : HloRunnerAgnosticReferenceMixin<HloPjRtTestBase>(
            /*reference_runner=*/GetReferenceRunner()) {}
};

TEST_F(SampleFileTest, Convolution) {
  const std::string filename = tsl::io::JoinPath(
      tsl::testing::XlaSrcRoot(), "tests", "isolated_convolution.hlo");
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ReadModuleFromHloTextFile(filename));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_cpu_parallel_codegen_split_count(1);

  module->mutable_config().mutable_debug_options().set_xla_gpu_autotune_level(
      4);
  EXPECT_TRUE(RunAndCompare(std::move(module), ErrorSpec{0.01}));
}

}  // namespace
}  // namespace xla
