/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <optional>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/service/gpu/fusion_merger.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info_for_tests.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"
#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/multi_output_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

class GpuFusionPipelineTest : public GpuCodegenTest {
  HloCostAnalysis::ShapeSizeFunction ShapeSizeBytesFunction() const {
    return [&](const Shape& shape) {
      constexpr int64_t kPointerSize = 8;
      return ShapeUtil::ByteSizeOf(shape, kPointerSize);
    };
  }

 public:
  void CheckGpuFusionPipeline(absl::string_view hlo,
                              std::optional<absl::string_view> expected) {
    HloPassPipeline pipeline("gpu-fusion");
    pipeline.AddPass<GpuInstructionFusion>(/*may_duplicate=*/false);
    pipeline.AddPass<GpuInstructionFusion>(/*may_duplicate=*/true);
    pipeline.AddPass<FusionMerger>(TestGpuDeviceInfo::RTXA6000DeviceInfo(),
                                   ShapeSizeBytesFunction());
    pipeline.AddPass<GpuMultiOutputFusion>(
        TestGpuDeviceInfo::RTXA6000DeviceInfo(), ShapeSizeBytesFunction());

    RunAndFilecheckHloRewrite(hlo, std::move(pipeline), expected);
  }
};

TEST_F(GpuFusionPipelineTest, TransposeAndOtherDifferentShape) {
  const char* hlo = R"(
HloModule module

ENTRY computation {
    p = f32[5000,6000]{1,0} parameter(0)
    e = f32[5000,6000]{1,0} sqrt(p)
    c = f32[6000,5000] transpose(p), dimensions={1,0}
    r = f32[300,20,5000] reshape(c)
    ROOT out = (f32[5000,6000], f32[300,20,5000]) tuple(e,r)
}
)";
  CheckGpuFusionPipeline(hlo, R"(
// CHECK: %fused_computation (param_0.1: f32[5000,6000]) -> (f32[300,20,5000], f32[5000,6000]) {
// CHECK-NEXT:   [[param_0_1_0:%[^ ]+]] = f32[5000,6000]{1,0} parameter(0)
// CHECK-NEXT:   [[c_1_1:%[^ ]+]] = f32[6000,5000]{1,0} transpose([[param_0_1_0]]), dimensions={1,0}
// CHECK-NEXT:   [[r_1_2:%[^ ]+]] = f32[300,20,5000]{2,1,0} reshape([[c_1_1]])
// CHECK-NEXT:   [[e_1_3:%[^ ]+]] = f32[5000,6000]{1,0} sqrt([[param_0_1_0]])
// CHECK-NEXT:   ROOT [[tuple_4:%[^ ]+]] = (f32[300,20,5000]{2,1,0}, f32[5000,6000]{1,0}) tuple([[r_1_2]], [[e_1_3]])
// CHECK-NEXT: }
// CHECK:   [[fusion_0:%[^ ]+]] = (f32[300,20,5000]{2,1,0}, f32[5000,6000]{1,0}) fusion([[p_1:%[^ ]+]]), kind=kInput, calls=[[fused_computation_2:%[^ ]+]]
  )");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
