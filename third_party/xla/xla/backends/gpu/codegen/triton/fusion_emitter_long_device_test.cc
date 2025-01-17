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

#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/autotuning.pb.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

class TritonEmitterLongDeviceTest : public GpuCodegenTest {
 public:
  const stream_executor::GpuComputeCapability& GpuComputeComp() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .gpu_compute_capability();
  }
};

TEST_F(TritonEmitterLongDeviceTest,
       FusionWithOutputContainingMoreThanInt32MaxElementsExecutesCorrectly) {
  // Note: if you break this test, the test infrastructure will break trying to
  // write the output to a file because it is larger than the maximum protobuf
  // size. To bypass this crash, it's possible to comment out the `Write*`
  // callbacks, in `literal_test_util.cc`'s `OnMiscompare`).
  constexpr absl::string_view kTritonHloText = R"(
computation {
  p0 = s8[256]{0} parameter(0)
  ROOT broadcast = s8[16777217,256]{1,0} broadcast(p0), dimensions={1}
}

ENTRY entry_computation {
  p0 = s8[256]{0} parameter(0)
  ROOT fusion = s8[16777217,256]{1,0} fusion(p0), kind=kCustom,
    calls=computation,
    backend_config={
      "fusion_backend_config":{
        "kind":"__triton",
        "block_level_fusion_config":{
          "output_tile_sizes":["2","256"],"num_warps":"1"}}}
})";

  constexpr absl::string_view kEmittersHloText = R"(
computation {
  p0 = s8[256]{0} parameter(0)
  ROOT broadcast = s8[16777217,256]{1,0} broadcast(p0), dimensions={1}
}

ENTRY entry_computation {
  p0 = s8[256]{0} parameter(0)
  ROOT fusion = s8[16777217,256]{1,0} fusion(p0), kind=kCustom,
    calls=computation
})";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> triton_module,
                          ParseAndReturnVerifiedModule(kTritonHloText));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> emitters_module,
                          ParseAndReturnVerifiedModule(kEmittersHloText));

  const Shape& output_shape =
      triton_module->entry_computation()->root_instruction()->shape();

  ASSERT_GT(Product(output_shape.dimensions()), 1l << 32);
  EXPECT_TRUE(RunAndCompareTwoModules(
      std::move(triton_module), std::move(emitters_module),
      ErrorSpec{/*aabs=*/0, /*arel=*/0}, /*run_hlo_passes=*/false));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
