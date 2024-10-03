/* Copyright 2024 The OpenXLA Authors.

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

#include <memory>
#include <utility>

#include "absl/strings/string_view.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/service/backend.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/service/hlo_runner_pjrt.h"
#include "xla/service/platform_util.h"
#include "xla/tests/hlo_runner_agnostic_test_base.h"
#include "xla/tests/pjrt_client_registry.h"
#include "xla/tests/test_macros.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace {

// Returns either an HloRunner or HloRunnerPjRt implementation depending on
// whether there exists a registered PjRtClientFactory.
std::unique_ptr<HloRunnerInterface> GetHloRunnerForTest(
    int xla_pjrt_cpu_intra_op_threads = -1) {
  if (ShouldUsePjRt()) {
    PjRtClientTestFactoryRegistry& pjrt_registry =
        GetGlobalPjRtClientTestFactory();
    std::unique_ptr<PjRtClient> client = pjrt_registry.Get()().value();
    PjRtClientTestFactoryRegistry::DeviceShapeRepresentationFn
        device_shape_representation_fn =
            pjrt_registry.GetDeviceShapeRepresentationFn(client.get());
    PjRtClientTestFactoryRegistry::DeviceShapeSizeFn device_shape_size_fn =
        pjrt_registry.GetDeviceShapeSizeFn(client.get());

    return std::make_unique<HloRunnerPjRt>(std::move(client),
                                           device_shape_representation_fn,
                                           device_shape_size_fn);
  }

  const int max_allowed_parallelism = xla_pjrt_cpu_intra_op_threads;
  return std::make_unique<HloRunner>(PlatformUtil::GetDefaultPlatform().value(),
                                     max_allowed_parallelism);
}
// The purpose of this test is to let us see how the flags that control the
// number of threads used for intra-op parallelism through the flag
// --xla_cpu_intra_op_parallelism_threads affect the performance of
// a simple HLO module that has a lot of parallelism.
class EmbarrassinglyParallelTest : public HloRunnerAgnosticTestBase {
 public:
  explicit EmbarrassinglyParallelTest(int xla_pjrt_cpu_intra_op_threads = 3)
      : HloRunnerAgnosticTestBase(
            GetHloRunnerForTest(xla_pjrt_cpu_intra_op_threads),
            GetHloRunnerForTest(xla_pjrt_cpu_intra_op_threads)) {}
  Backend& backend() {
    return static_cast<HloRunner*>(&test_runner())->backend();
  }
};

XLA_TEST_F(EmbarrassinglyParallelTest, Simple) {
  absl::string_view hlo_string = R"(
    HloModule EmbarrassinglyParallelModule

    ENTRY Compute {
      input           = f32[1000000000]{0} parameter(0)
      constant        = f32[] constant(1.0001)
      broadcast_const = f32[1000000000]{0} broadcast(constant), dimensions={}
      power_op        = f32[1000000000]{0} power(input, broadcast_const)
      ROOT result     = f32[1000000000]{0} multiply(power_op, input)
  })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string, {}));
  EXPECT_TRUE(Run(std::move(module), /*run_hlo_passes=*/false));
  EXPECT_EQ(backend().eigen_intra_op_thread_pool()->NumThreads(), 3);
}

XLA_TEST_F(EmbarrassinglyParallelTest, TfrtCpuClient) {
  absl::string_view hlo_string = R"(
    HloModule EmbarrassinglyParallelModule

    ENTRY Compute {
      input           = f32[1000000000]{0} parameter(0)
      constant        = f32[] constant(1.0001)
      broadcast_const = f32[1000000000]{0} broadcast(constant), dimensions={}
      power_op        = f32[1000000000]{0} power(input, broadcast_const)
      ROOT result     = f32[1000000000]{0} multiply(power_op, input)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnUnverifiedModule(hlo_string, {}));

  CpuClientOptions cpu_options;
  cpu_options.cpu_device_count = 1;
  TF_ASSERT_OK_AND_ASSIGN(auto client,
                          GetXlaPjrtCpuClient(std::move(cpu_options)));

  XlaComputation xla_computation(module->ToProto());
  TF_ASSERT_OK_AND_ASSIGN(auto pjrt_executable,
                          client->Compile(xla_computation, /*options=*/{}));
  auto result = pjrt_executable->Execute(/*argument_handles=*/{},
                                         /*options=*/{});
  EXPECT_EQ(backend().eigen_intra_op_thread_pool()->NumThreads(), 3);
}

}  // namespace
}  // namespace xla
