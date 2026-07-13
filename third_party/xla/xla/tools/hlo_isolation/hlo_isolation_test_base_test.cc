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

#include "xla/tools/hlo_isolation/hlo_isolation_test_base.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest-spi.h>
#include "absl/base/nullability.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/array2d.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/computation_placer.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_interpreter_reference_mixin.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_utils.h"
#include "xla/tools/hlo_isolation/hlo_isolation.pb.h"
#include "xla/tools/hlo_isolation/hlo_isolation_api.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "tsl/platform/path.h"

namespace xla {
namespace hlo_isolation {
namespace {

class HloIsolationTest
    : public HloIsolationTestMixin<HloInterpreterReferenceMixin<HloTestBase>> {
};

TEST_F(HloIsolationTest, RunSimpleModule) {
  const char* hlo_text = R"(
HloModule SimpleModule

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  RunAndVerifyIsolationTest(*module);
}

TEST_F(HloIsolationTest, RunWithMultipleOutputs) {
  const char* hlo_text = R"(
HloModule MultiOutputModule

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  add = f32[] add(a, b)
  mul = f32[] multiply(a, b)
  ROOT root = (f32[], f32[]) tuple(add, mul)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  RunAndVerifyIsolationTest(*module);
}

TEST_F(HloIsolationTest, TestExtractTopMismatches) {
  std::string error_message = R"(
Mismatch count 45846 (91.3704%) in shape bf16[50176] (50176 elements), abs bound 0.01, rel bound 0.1
Top relative error mismatches:
  actual      0.06885, expected            0, index {1}, rel error      inf, abs error   0.0688
  actual        0.052, expected            0, index {1}, rel error      inf, abs error    0.052
Elements exceeding abs error bound 0.01: 37627769 (99.6795%)
Elements exceeding rel error bound 0.1: 37365238 (98.9841%)
)";

  const char* hlo_text = R"(
HloModule SimpleModule

ENTRY main {
  a = f32[50176] parameter(0)
  ROOT add = f32[50176] add(a, a)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  ASSERT_OK_AND_ASSIGN(
      std::vector<NumericMismatch> mismatches,
      ExtractAndEnrichTopMismatches(error_message, module.get()));
  ASSERT_FALSE(mismatches.empty());
  EXPECT_DOUBLE_EQ(mismatches[0].actual(), 0.06885);
  EXPECT_DOUBLE_EQ(mismatches[0].expected(), 0.0);
  EXPECT_DOUBLE_EQ(mismatches[0].percentage_of_elems_exceeding_abs_error(),
                   99.6795);
  EXPECT_DOUBLE_EQ(mismatches[0].percentage_of_elems_exceeding_rel_error(),
                   98.9841);
}

TEST_F(HloIsolationTest, TestOutputIsReduce) {
  std::string hlo_string = R"hlo(
HloModule multiply_reduce_fusion.94, is_scheduled=true, entry_computation_layout={(f32[14,40,100,128]{3,2,1,0:T(8,128)}, f32[14,40,100]{0,2,1:T(8,128)}, f32[14,40,100]{0,2,1:T(8,128)}, f32[128]{0:T(128)}, bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)}, /*index=5*/bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)}, bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)}, bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)}, bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)}, bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)}, /*index=10*/bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)}, bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)}, bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)}, bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)}, bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)}, /*index=15*/bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)}, bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)}, bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)}, bf16[14,40,100,16]{0,3,2,1:T(8,128)(2,1)}, f32[1,1,128,16]{2,3,1,0:T(8,128)}, /*index=20*/f32[128]{0:T(128)})->(f32[128]{0:T(128)}, f32[14,40,100]{0,2,1:T(8,128)}, f32[14,40,100]{0,2,1:T(8,128)}, f32[128]{0:T(128)}, bf16[14,40,100,128]{3,0,2,1:T(8,128)(2,1)})}

%bitcast_fusion.22.clone (bitcast_input.22: bf16[14,40,100,16]) -> bf16[14,40,100,16] {
  %bitcast_input.22 = bf16[14,40,100,16]{0,3,2,1:T(8,128)(2,1)S(1)} parameter(0)
  ROOT %bitcast.24651 = bf16[14,40,100,16]{0,3,2,1:T(8,128)(2,1)} bitcast(%bitcast_input.22)
}

%bitcast_fusion.436.clone (bitcast_input.436: f32[1,1,128,16]) -> f32[1,1,128,16] {
  %bitcast_input.436 = f32[1,1,128,16]{2,3,1,0:T(8,128)S(1)} parameter(0)
  ROOT %bitcast.25065 = f32[1,1,128,16]{2,3,1,0:T(8,128)} bitcast(%bitcast_input.436)
}

%region_3155.3539 (reduce_sum.21164: f32[], reduce_sum.21165: f32[]) -> f32[] {
  %reduce_sum.21164 = f32[]{:T(128)} parameter(0), metadata={op_name="reduce_sum"}
  %reduce_sum.21165 = f32[]{:T(128)} parameter(1), metadata={op_name="reduce_sum"}
  ROOT %reduce_sum.21166 = f32[]{:T(128)} add(%reduce_sum.21164, %reduce_sum.21165), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/lss_decoder.call_from_flax_module/lss_decoder._call_with_mesh/lss_decoder/skip_1/Normalization_0/LayerNorm_0/reduce_sum" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}, backend_config={"flag_configs":[],"scoped_memory_configs":[],"used_scoped_memory_configs":[],"aliasing_operands":{"lists":[]}}
}

%region_3156.3540 (reduce_sum.21171: f32[], reduce_sum.21172: f32[]) -> f32[] {
  %reduce_sum.21171 = f32[]{:T(128)} parameter(0), metadata={op_name="reduce_sum"}
  %reduce_sum.21172 = f32[]{:T(128)} parameter(1), metadata={op_name="reduce_sum"}
  ROOT %reduce_sum.21173 = f32[]{:T(128)} add(%reduce_sum.21171, %reduce_sum.21172), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/lss_decoder.call_from_flax_module/lss_decoder._call_with_mesh/lss_decoder/skip_1/Normalization_0/LayerNorm_0/reduce_sum" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}, backend_config={"flag_configs":[],"scoped_memory_configs":[],"used_scoped_memory_configs":[],"aliasing_operands":{"lists":[]}}
}

%region_3157.3541 (reduce_sum.21178: f32[], reduce_sum.21179: f32[]) -> f32[] {
  %reduce_sum.21178 = f32[]{:T(128)} parameter(0), metadata={op_name="reduce_sum"}
  %reduce_sum.21179 = f32[]{:T(128)} parameter(1), metadata={op_name="reduce_sum"}
  ROOT %reduce_sum.21180 = f32[]{:T(128)} add(%reduce_sum.21178, %reduce_sum.21179), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/lss_decoder.call_from_flax_module/lss_decoder._call_with_mesh/lss_decoder/skip_1/Normalization_0/LayerNorm_0/reduce_sum" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}, backend_config={"flag_configs":[],"scoped_memory_configs":[],"used_scoped_memory_configs":[],"aliasing_operands":{"lists":[]}}
}

%region_3154.3538 (reduce_sum.21157: f32[], reduce_sum.21158: f32[]) -> f32[] {
  %reduce_sum.21157 = f32[]{:T(128)} parameter(0), metadata={op_name="reduce_sum"}
  %reduce_sum.21158 = f32[]{:T(128)} parameter(1), metadata={op_name="reduce_sum"}
  ROOT %reduce_sum.21159 = f32[]{:T(128)} add(%reduce_sum.21157, %reduce_sum.21158), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/lss_decoder.call_from_flax_module/lss_decoder._call_with_mesh/lss_decoder/skip_1/Normalization_0/LayerNorm_0/reduce_sum" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}, backend_config={"flag_configs":[],"scoped_memory_configs":[],"used_scoped_memory_configs":[],"aliasing_operands":{"lists":[]}}
}

%fused_computation.1520.clone (param_0.102180: f32[14,40,100,128], param_1.121470: f32[14,40,100], param_2.102997: f32[14,40,100], param_3.80528: f32[128], param_4.56314: bf16[1,1,40,100,128], param_5.46641: bf16[1,1,40,100,128], param_6.42711: bf16[1,1,40,100,128], param_7.34103: bf16[1,1,40,100,128], param_8.29161: bf16[1,1,40,100,128], param_9.16553: bf16[1,1,40,100,128], param_10.15954: bf16[1,1,40,100,128], param_11.16327: bf16[1,1,40,100,128], param_12.12685: bf16[1,1,40,100,128], param_13.11276: bf16[1,1,40,100,128], param_14.5517: bf16[1,1,40,100,128], param_15.4621: bf16[1,1,40,100,128], param_16.4256: bf16[1,1,40,100,128], param_17.4162: bf16[1,1,40,100,128], param_18.4086: bf16[14,40,100,16], param_19.4135: f32[1,1,128,16], param_20.4038: f32[128]) -> (f32[128], f32[14,40,100], f32[14,40,100], f32[128], bf16[14,40,100,128]) {
  %param_1.121470 = f32[14,40,100]{0,2,1:T(8,128)S(1)} parameter(1)
  %mul.26407 = f32[14,40,100,128]{3,0,2,1:T(8,128)} broadcast(%param_1.121470), dimensions={0,1,2}, metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/rematted_computation/lss_backbone._internal_call/lss_decoder.call_from_flax_module/lss_decoder._call_with_mesh/lss_decoder/skip_1/Normalization_0/LayerNorm_0/mul" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %param_0.102180 = f32[14,40,100,128]{3,2,1,0:T(8,128)S(1)} parameter(0)
  %copy.125581 = f32[14,40,100,128]{3,0,2,1:T(8,128)} copy(%param_0.102180), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/rematted_computation/lss_backbone._internal_call/lss_decoder.call_from_flax_module/lss_decoder._call_with_mesh/lss_decoder/skip_1/Normalization_0/LayerNorm_0/convert_element_type" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %param_2.102997 = f32[14,40,100]{0,2,1:T(8,128)S(1)} parameter(2)
  %sub.4972 = f32[14,40,100,128]{3,0,2,1:T(8,128)} broadcast(%param_2.102997), dimensions={0,1,2}, metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/rematted_computation/lss_backbone._internal_call/lss_decoder.call_from_flax_module/lss_decoder._call_with_mesh/lss_decoder/skip_1/Normalization_0/LayerNorm_0/sub" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %sub.4712 = f32[14,40,100,128]{3,0,2,1:T(8,128)} subtract(%copy.125581, %sub.4972), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/rematted_computation/lss_backbone._internal_call/lss_decoder.call_from_flax_module/lss_decoder._call_with_mesh/lss_decoder/skip_1/Normalization_0/LayerNorm_0/sub" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %param_3.80528 = f32[128]{0:T(128)S(1)} parameter(3)
  %mul.125427 = f32[14,40,100,128]{3,0,2,1:T(8,128)} broadcast(%param_3.80528), dimensions={3}, metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/rematted_computation/lss_backbone._internal_call/lss_decoder.call_from_flax_module/lss_decoder._call_with_mesh/lss_decoder/skip_1/Normalization_0/LayerNorm_0/mul" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %mul.125426 = f32[14,40,100,128]{3,0,2,1:T(8,128)} multiply(%mul.26407, %mul.125427), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/rematted_computation/lss_backbone._internal_call/lss_decoder.call_from_flax_module/lss_decoder._call_with_mesh/lss_decoder/skip_1/Normalization_0/LayerNorm_0/mul" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %mul.125425 = f32[14,40,100,128]{3,0,2,1:T(8,128)} multiply(%sub.4712, %mul.125426), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/rematted_computation/lss_backbone._internal_call/lss_decoder.call_from_flax_module/lss_decoder._call_with_mesh/lss_decoder/skip_1/Normalization_0/LayerNorm_0/mul" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %param_20.4038 = f32[128]{0:T(128)S(1)} parameter(20)
  %add.111088 = f32[14,40,100,128]{3,0,2,1:T(8,128)} broadcast(%param_20.4038), dimensions={3}, metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/rematted_computation/lss_backbone._internal_call/lss_decoder.call_from_flax_module/lss_decoder._call_with_mesh/lss_decoder/skip_1/Normalization_0/LayerNorm_0/add" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %add.111087 = f32[14,40,100,128]{3,0,2,1:T(8,128)} add(%mul.125425, %add.111088), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/rematted_computation/lss_backbone._internal_call/lss_decoder.call_from_flax_module/lss_decoder._call_with_mesh/lss_decoder/skip_1/Normalization_0/LayerNorm_0/add" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %convert_element_type.42958 = bf16[14,40,100,128]{3,0,2,1:T(8,128)(2,1)} convert(%add.111087), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/rematted_computation/lss_backbone._internal_call/lss_decoder.call_from_flax_module/lss_decoder._call_with_mesh/lss_decoder/skip_1/Normalization_0/LayerNorm_0/convert_element_type" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %constant.104152.clone.1 = bf16[]{:T(256)} constant(0), metadata={op_name="jit(run)/jvp(ModuleBuilder._call_with_mesh)/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_ybr_backbones/ybr_perspective_backbones_0/range_image_backbone/range_image_backbone._compute_perspective_output/range_image_backbone._compute_laser_and_task_features/encoder/stem/stem_conv_0/jit(relu)"}
  %broadcast.63152.clone.1 = bf16[14,40,100,128]{3,0,2,1:T(8,128)(2,1)} broadcast(%constant.104152.clone.1), dimensions={}
  %gt.1695.clone.1 = pred[14,40,100,128]{3,0,2,1:T(8,128)(4,1)} compare(%convert_element_type.42958, %broadcast.63152.clone.1), direction=GT, metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/rematted_computation/lss_backbone._internal_call/lss_decoder.call_from_flax_module/lss_decoder._call_with_mesh/lss_decoder/skip_1/gt" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %max.5877.clone.1 = bf16[14,40,100,128]{3,0,2,1:T(8,128)(2,1)} maximum(%convert_element_type.42958, %broadcast.63152.clone.1), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/rematted_computation/lss_backbone._internal_call/lss_decoder.call_from_flax_module/lss_decoder._call_with_mesh/lss_decoder/skip_1/jit(relu)/max" source_file="waymo/ml/pmc/modules/convolution.py" source_line=86 source_end_line=86 source_column=10 source_end_column=28}
  %is_finite.46.clone.1 = pred[14,40,100,128]{3,0,2,1:T(8,128)(4,1)} is-finite(%max.5877.clone.1), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/rematted_computation/lss_backbone._internal_call/is_finite" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %param_18.4086 = bf16[14,40,100,16]{0,3,2,1:T(8,128)(2,1)S(1)} parameter(18)
  %fusion.23695 = bf16[14,40,100,16]{0,3,2,1:T(8,128)(2,1)} fusion(%param_18.4086), kind=kLoop, calls=%bitcast_fusion.22.clone
  %param_19.4135 = f32[1,1,128,16]{2,3,1,0:T(8,128)S(1)} parameter(19)
  %fusion.24109 = f32[1,1,128,16]{2,3,1,0:T(8,128)} fusion(%param_19.4135), kind=kLoop, calls=%bitcast_fusion.436.clone
  %convolution.2294.clone.1 = bf16[14,40,100,128]{3,0,2,1:T(8,128)(2,1)} convolution(%fusion.23695, %fusion.24109), window={size=1x1}, dim_labels=b01f_01oi->b01f, metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/shared_camera_head_conv.call_from_flax_module/shared_camera_head_conv._call_with_mesh/shared_camera_head_conv/Conv_0/conv_general_dilated" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %param_17.4162 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)} parameter(17)
  %pad.3204.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} pad(%param_17.4162, %constant.104152.clone.1), padding=0_0x13_0x0_0x0_0x0_0, metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/pad" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %param_16.4256 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)S(1)} parameter(16)
  %pad.3203.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} pad(%param_16.4256, %constant.104152.clone.1), padding=0_0x12_1x0_0x0_0x0_0, metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/pad" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %add_any.3234.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} add(%pad.3204.clone.1, %pad.3203.clone.1), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/add_any" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %param_15.4621 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)S(1)} parameter(15)
  %pad.3202.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} pad(%param_15.4621, %constant.104152.clone.1), padding=0_0x11_2x0_0x0_0x0_0, metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/pad" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %add_any.3233.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} add(%add_any.3234.clone.1, %pad.3202.clone.1), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/add_any" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %param_14.5517 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)S(1)} parameter(14)
  %pad.3201.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} pad(%param_14.5517, %constant.104152.clone.1), padding=0_0x10_3x0_0x0_0x0_0, metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/pad" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %add_any.3232.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} add(%add_any.3233.clone.1, %pad.3201.clone.1), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/add_any" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %param_13.11276 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)S(1)} parameter(13)
  %pad.3200.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} pad(%param_13.11276, %constant.104152.clone.1), padding=0_0x9_4x0_0x0_0x0_0, metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/pad" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %add_any.3231.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} add(%add_any.3232.clone.1, %pad.3200.clone.1), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/add_any" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %param_12.12685 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)S(1)} parameter(12)
  %pad.3199.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} pad(%param_12.12685, %constant.104152.clone.1), padding=0_0x8_5x0_0x0_0x0_0, metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/pad" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %add_any.3230.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} add(%add_any.3231.clone.1, %pad.3199.clone.1), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/add_any" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %param_11.16327 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)S(1)} parameter(11)
  %pad.3198.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} pad(%param_11.16327, %constant.104152.clone.1), padding=0_0x7_6x0_0x0_0x0_0, metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/pad" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %add_any.3229.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} add(%add_any.3230.clone.1, %pad.3198.clone.1), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/add_any" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %param_10.15954 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)S(1)} parameter(10)
  %pad.3197.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} pad(%param_10.15954, %constant.104152.clone.1), padding=0_0x6_7x0_0x0_0x0_0, metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/add_any" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %add_any.3228.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} add(%add_any.3229.clone.1, %pad.3197.clone.1), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/add_any" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %param_9.16553 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)} parameter(9)
  %pad.3196.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} pad(%param_9.16553, %constant.104152.clone.1), padding=0_0x5_8x0_0x0_0x0_0, metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/pad" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %add_any.3227.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} add(%add_any.3228.clone.1, %pad.3196.clone.1), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/add_any" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %param_8.29161 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)S(1)} parameter(8)
  %pad.3195.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} pad(%param_8.29161, %constant.104152.clone.1), padding=0_0x4_9x0_0x0_0x0_0, metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/pad" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %add_any.3226.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} add(%add_any.3227.clone.1, %pad.3195.clone.1), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/add_any" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %param_7.34103 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)} parameter(7)
  %pad.3194.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} pad(%param_7.34103, %constant.104152.clone.1), padding=0_0x3_10x0_0x0_0x0_0, metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/pad" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %add_any.3225.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} add(%add_any.3226.clone.1, %pad.3194.clone.1), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/add_any" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %param_6.42711 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)} parameter(6)
  %pad.3193.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} pad(%param_6.42711, %constant.104152.clone.1), padding=0_0x2_11x0_0x0_0x0_0, metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/pad" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %add_any.3224.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} add(%add_any.3225.clone.1, %pad.3193.clone.1), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/add_any" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %param_5.46641 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)} parameter(5)
  %pad.3192.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} pad(%param_5.46641, %constant.104152.clone.1), padding=0_0x1_12x0_0x0_0x0_0, metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/pad" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %add_any.3223.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} add(%add_any.3224.clone.1, %pad.3192.clone.1), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/add_any" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %param_4.56314 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)} parameter(4)
  %pad.3191.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} pad(%param_4.56314, %constant.104152.clone.1), padding=0_0x0_13x0_0x0_0x0_0, metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/pad" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %add_any.3222.clone.1 = bf16[1,14,40,100,128]{4,1,3,2,0:T(8,128)(2,1)} add(%add_any.3223.clone.1, %pad.3191.clone.1), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/front/add_any" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %bitcast.9678.clone.1 = bf16[14,40,100,128]{3,0,2,1:T(8,128)(2,1)} bitcast(%add_any.3222.clone.1), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/reshape" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %add_any.3219.clone.1 = bf16[14,40,100,128]{3,0,2,1:T(8,128)(2,1)} add(%convolution.2294.clone.1, %bitcast.9678.clone.1), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/add_any" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %select_n.6128.clone.1 = bf16[14,40,100,128]{3,0,2,1:T(8,128)(2,1)} select(%is_finite.46.clone.1, %add_any.3219.clone.1, %broadcast.63152.clone.1), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/jit(_where)/select_n" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=867 source_end_line=867 source_column=9 source_end_column=57}
  %select_n.6005.clone.1 = bf16[14,40,100,128]{3,0,2,1:T(8,128)(2,1)S(1)} select(%gt.1695.clone.1, %select_n.6128.clone.1, %broadcast.63152.clone.1), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/lss_decoder.call_from_flax_module/lss_decoder._call_with_mesh/lss_decoder/skip_1/select_n" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %convert_element_type.16305 = f32[14,40,100,128]{3,0,2,1:T(8,128)} convert(%select_n.6005.clone.1), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/lss_decoder.call_from_flax_module/lss_decoder._call_with_mesh/lss_decoder/skip_1/Normalization_0/LayerNorm_0/convert_element_type" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %mul.25107 = f32[14,40,100,128]{3,0,2,1:T(8,128)} multiply(%sub.4712, %convert_element_type.16305), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/lss_decoder.call_from_flax_module/lss_decoder._call_with_mesh/lss_decoder/skip_1/Normalization_0/LayerNorm_0/mul" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %mul.24449 = f32[14,40,100,128]{3,0,2,1:T(8,128)} multiply(%mul.26407, %mul.25107), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/lss_decoder.call_from_flax_module/lss_decoder._call_with_mesh/lss_decoder/skip_1/Normalization_0/LayerNorm_0/mul" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %constant.108606 = f32[]{:T(128)} constant(0), metadata={op_name="jit(run)/jvp(ModuleBuilder._call_with_mesh)/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_ybr_backbones/ybr_perspective_backbones_0/range_image_backbone/range_image_backbone._compute_perspective_output/range_image_backbone._sensor_drop/Dropout_0/jit(_bernoulli)/jit(_uniform)/max" source_file="third_party/py/flax/linen/stochastic.py" source_line=105 source_end_line=105 source_column=11 source_end_column=68}
  %reduce.7929 = f32[128]{0:T(128)} reduce(%mul.24449, %constant.108606), dimensions={0,1,2}, to_apply=%region_3155.3539, metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/lss_decoder.call_from_flax_module/lss_decoder._call_with_mesh/lss_decoder/skip_1/Normalization_0/LayerNorm_0/reduce_sum" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %mul.24465.clone.1 = f32[14,40,100,128]{3,0,2,1:T(8,128)} multiply(%mul.25107, %mul.125427), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/lss_decoder.call_from_flax_module/lss_decoder._call_with_mesh/lss_decoder/skip_1/Normalization_0/LayerNorm_0/reduce_sum" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %reduce.7935.clone.1 = f32[14,40,100]{0,2,1:T(8,128)S(1)} reduce(%mul.24465.clone.1, %constant.108606), dimensions={3}, to_apply=%region_3156.3540, metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/lss_decoder.call_from_flax_module/lss_decoder._call_with_mesh/lss_decoder/skip_1/Normalization_0/LayerNorm_0/reduce_sum" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %mul.23884.clone.1 = f32[14,40,100,128]{3,0,2,1:T(8,128)} multiply(%convert_element_type.16305, %mul.125426), metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/lss_decoder.call_from_flax_module/lss_decoder._call_with_mesh/lss_decoder/skip_1/Normalization_0/LayerNorm_0/mul" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %reduce.7837.clone.1 = f32[14,40,100]{0,2,1:T(8,128)S(1)} reduce(%mul.23884.clone.1, %constant.108606), dimensions={3}, to_apply=%region_3157.3541, metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/lss_decoder.call_from_flax_module/lss_decoder._call_with_mesh/lss_decoder/skip_1/Normalization_0/LayerNorm_0/reduce_sum" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  %reduce.7809.clone.1 = f32[128]{0:T(128)} reduce(%convert_element_type.16305, %constant.108606), dimensions={0,1,2}, to_apply=%region_3154.3538, metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/lss_decoder.call_from_flax_module/lss_decoder._call_with_mesh/lss_decoder/skip_1/Normalization_0/LayerNorm_0/reduce_sum" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}
  ROOT %tuple.8259 = (f32[128]{0:T(128)}, f32[14,40,100]{0,2,1:T(8,128)S(1)}, f32[14,40,100]{0,2,1:T(8,128)S(1)}, f32[128]{0:T(128)}, bf16[14,40,100,128]{3,0,2,1:T(8,128)(2,1)S(1)}) tuple(%reduce.7929, %reduce.7935.clone.1, %reduce.7837.clone.1, %reduce.7809.clone.1, %select_n.6005.clone.1)
}

ENTRY %multiply_reduce_fusion.94 (parameter.0: f32[14,40,100,128], parameter.1: f32[14,40,100], parameter.2: f32[14,40,100], parameter.3: f32[128], parameter.4: bf16[1,1,40,100,128], parameter.5: bf16[1,1,40,100,128], parameter.6: bf16[1,1,40,100,128], parameter.7: bf16[1,1,40,100,128], parameter.8: bf16[1,1,40,100,128], parameter.9: bf16[1,1,40,100,128], parameter.10: bf16[1,1,40,100,128], parameter.11: bf16[1,1,40,100,128], parameter.12: bf16[1,1,40,100,128], parameter.13: bf16[1,1,40,100,128], parameter.14: bf16[1,1,40,100,128], parameter.15: bf16[1,1,40,100,128], parameter.16: bf16[1,1,40,100,128], parameter.17: bf16[1,1,40,100,128], parameter.18: bf16[14,40,100,16], parameter.19: f32[1,1,128,16], parameter.20: f32[128]) -> (f32[128], f32[14,40,100], f32[14,40,100], f32[128], bf16[14,40,100,128]) {
  %parameter.0 = f32[14,40,100,128]{3,2,1,0:T(8,128)} parameter(0)
  %copy = f32[14,40,100,128]{3,2,1,0:T(8,128)S(1)} copy(%parameter.0)
  %parameter.1 = f32[14,40,100]{0,2,1:T(8,128)} parameter(1)
  %copy.1 = f32[14,40,100]{0,2,1:T(8,128)S(1)} copy(%parameter.1)
  %parameter.2 = f32[14,40,100]{0,2,1:T(8,128)} parameter(2)
  %copy.2 = f32[14,40,100]{0,2,1:T(8,128)S(1)} copy(%parameter.2)
  %parameter.3 = f32[128]{0:T(128)} parameter(3)
  %copy.3 = f32[128]{0:T(128)S(1)} copy(%parameter.3)
  %parameter.4 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)} parameter(4)
  %copy.4 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)S(1)} copy(%parameter.4)
  %parameter.5 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)} parameter(5)
  %copy.5 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)S(1)} copy(%parameter.5)
  %parameter.6 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)} parameter(6)
  %copy.6 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)S(1)} copy(%parameter.6)
  %parameter.7 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)} parameter(7)
  %copy.7 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)S(1)} copy(%parameter.7)
  %parameter.8 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)} parameter(8)
  %copy.8 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)S(1)} copy(%parameter.8)
  %parameter.9 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)} parameter(9)
  %copy.9 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)S(1)} copy(%parameter.9)
  %parameter.10 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)} parameter(10)
  %copy.10 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)S(1)} copy(%parameter.10)
  %parameter.11 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)} parameter(11)
  %copy.11 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)S(1)} copy(%parameter.11)
  %parameter.12 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)} parameter(12)
  %copy.12 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)S(1)} copy(%parameter.12)
  %parameter.13 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)} parameter(13)
  %copy.13 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)S(1)} copy(%parameter.13)
  %parameter.14 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)} parameter(14)
  %copy.14 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)S(1)} copy(%parameter.14)
  %parameter.15 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)} parameter(15)
  %copy.15 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)S(1)} copy(%parameter.15)
  %parameter.16 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)} parameter(16)
  %copy.16 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)S(1)} copy(%parameter.16)
  %parameter.17 = bf16[1,1,40,100,128]{4,1,3,2,0:T(2,128)(2,1)} parameter(17)
  %parameter.18 = bf16[14,40,100,16]{0,3,2,1:T(8,128)(2,1)} parameter(18)
  %copy.17 = bf16[14,40,100,16]{0,3,2,1:T(8,128)(2,1)S(1)} copy(%parameter.18)
  %parameter.19 = f32[1,1,128,16]{2,3,1,0:T(8,128)} parameter(19)
  %copy.18 = f32[1,1,128,16]{2,3,1,0:T(8,128)S(1)} copy(%parameter.19)
  %parameter.20 = f32[128]{0:T(128)} parameter(20)
  %copy.19 = f32[128]{0:T(128)S(1)} copy(%parameter.20)
  %multiply_reduce_fusion.94 = (f32[128]{0:T(128)}, f32[14,40,100]{0,2,1:T(8,128)S(1)}, f32[14,40,100]{0,2,1:T(8,128)S(1)}, f32[128]{0:T(128)}, bf16[14,40,100,128]{3,0,2,1:T(8,128)(2,1)S(1)}) fusion(%copy, %copy.1, %copy.2, %copy.3, %copy.4, /*index=5*/%copy.5, %copy.6, %copy.7, %copy.8, %copy.9, /*index=10*/%copy.10, %copy.11, %copy.12, %copy.13, %copy.14, /*index=15*/%copy.15, %copy.16, %parameter.17, %copy.17, %copy.18, /*index=20*/%copy.19), kind=kOutput, calls=%fused_computation.1520.clone, metadata={op_name="jit(run)/transpose(jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/jvp(ModuleBuilder._call_with_mesh))/ModuleBuilder/ModuleBuilder._call_modules_on_inputs/_module_builder_submodules_backbone/_module_builder_submodules_backbone.apply_sensor_backbones/_module_builder_submodules_backbone.apply_camera_backbones/camera_backbones_0/lss_backbone/checkpoint/lss_backbone._internal_call/shared_camera_head_conv.call_from_flax_module/shared_camera_head_conv._call_with_mesh/shared_camera_head_conv/Conv_0/conv_general_dilated" source_file="waymo/ml/pmc/models/bev/modules/camera_backbone.py" source_line=498 source_end_line=500 source_column=13 source_end_column=7}, backend_config={"flag_configs":[],"window_config":{"kernel_window_bounds":["1","1","2","1"],"output_window_bounds":["4","100","2","1"],"input_window_bounds":[],"estimated_cycles":"0","iteration_bounds":[],"cost_model_type":"COST_MODEL_TYPE_INVALID","ml_estimated_microseconds":0,"is_mask":false,"pad_output_on_minor_dim":"0","pad_input_on_minor_dim":"0"},"scoped_memory_configs":[],"used_scoped_memory_configs":[],"aliasing_operands":{"lists":[]}}
  %get-tuple-element = f32[128]{0:T(128)} get-tuple-element(%multiply_reduce_fusion.94), index=0
  %get-tuple-element.1 = f32[14,40,100]{0,2,1:T(8,128)S(1)} get-tuple-element(%multiply_reduce_fusion.94), index=1
  %get-tuple-element.2 = f32[14,40,100]{0,2,1:T(8,128)S(1)} get-tuple-element(%multiply_reduce_fusion.94), index=2
  %get-tuple-element.3 = f32[128]{0:T(128)} get-tuple-element(%multiply_reduce_fusion.94), index=3
  %get-tuple-element.4 = bf16[14,40,100,128]{3,0,2,1:T(8,128)(2,1)S(1)} get-tuple-element(%multiply_reduce_fusion.94), index=4
  %copy.20 = f32[14,40,100]{0,2,1:T(8,128)} copy(%get-tuple-element.1)
  %copy.21 = f32[14,40,100]{0,2,1:T(8,128)} copy(%get-tuple-element.2)
  %copy.22 = bf16[14,40,100,128]{3,0,2,1:T(8,128)(2,1)} copy(%get-tuple-element.4)
  ROOT %tuple = (f32[128]{0:T(128)}, f32[14,40,100]{0,2,1:T(8,128)}, f32[14,40,100]{0,2,1:T(8,128)}, f32[128]{0:T(128)}, bf16[14,40,100,128]{3,0,2,1:T(8,128)(2,1)}) tuple(%get-tuple-element, %copy.20, %copy.21, %get-tuple-element.3, %copy.22)
}
)hlo";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  ASSERT_OK_AND_ASSIGN(std::vector<bool> output_is_reduce,
                       DetectReducesInModuleOutput(module.get()));

  EXPECT_EQ(output_is_reduce.size(), 5);
  EXPECT_EQ(output_is_reduce[0], true);
  EXPECT_EQ(output_is_reduce[1], true);
  EXPECT_EQ(output_is_reduce[2], true);
  EXPECT_EQ(output_is_reduce[3], true);
  EXPECT_EQ(output_is_reduce[4], false);
}

TEST_F(HloIsolationTest, TestExtractTopRelativeErrorMismatch) {
  std::string error_message = R"(
Mismatch count 45846 (91.3704%) in shape bf16[49,8,128] (50176 elements), abs bound 0.01, rel bound 0.1
Top relative error mismatches:
  actual      0.06885, expected            0, index {1,0,4}, rel error      inf, abs error   0.0688
  actual        0.052, expected            0, index {1,0,3}, rel error      inf, abs error    0.052
  actual     -0.03662, expected            0, index {1,0,2}, rel error      inf, abs error   0.0366
  actual     -0.09863, expected            0, index {1,0,1}, rel error      inf, abs error   0.0986
  actual     -0.02368, expected            0, index {1,0,0}, rel error      inf, abs error   0.0237
Absolute magnitude breakdown of actual values:
  0      <= x < 0.0001 :    4486 (  0.0119%), mismatches 4484
  0.0001 <= x < 0.001  :      52 (  0.0001%), mismatches 52
  0.001  <= x < 0.01   :    4334 (  0.0115%), mismatches 4334
  0.01   <= x < 0.1    :   75305 (  0.1995%), mismatches 75262
  0.1    <= x < 1      :  760261 (  2.0140%), mismatches 759849
  1      <= x < inf    : 36904298 ( 97.7630%), mismatches 36521233
Elements exceeding abs error bound 0.01: 37627769 (99.6795%)
Relative error breakdown of elements exceeding abs error bound:
  <  0.0001 :       0 (0.0000%)
  >= 0.0001 : 37627769 (100.0000%)
  >= 0.001  : 37627769 (100.0000%)
  >= 0.01   : 37367521 (99.3084%)
  >= 0.1    : 33928842 (90.1697%)
  >= 1      : 2927511 (7.7802%)
Elements exceeding rel error bound 0.01: 37365238 (98.9841%)
Absolute error breakdown of elements exceeding rel error bound:
  <  0.0001 :       0 (0.0000%)
  >= 0.0001 : 37365238 (100.0000%)
  >= 0.001  : 37365238 (100.0000%)
  >= 0.01   : 37365214 (99.9999%)
  >= 0.1    : 37346271 (99.9492%)
  >= 1      : 35637554 (95.3762%)
)";

  ASSERT_OK_AND_ASSIGN(NumericMismatch mismatch,
                       ExtractTopRelativeErrorMismatch(error_message));
  EXPECT_EQ(mismatch.actual(), 0.06885);
  EXPECT_EQ(mismatch.expected(), 0);
  EXPECT_EQ(mismatch.rel_error(), std::numeric_limits<double>::infinity());
  EXPECT_EQ(mismatch.percentage_of_elems_exceeding_abs_error(), 99.6795);
  EXPECT_EQ(mismatch.percentage_of_elems_exceeding_rel_error(), 98.9841);
}

TEST_F(HloIsolationTest, TestExtractTopMismatches1) {
  std::string error_message = R"(
Mismatches in shape (f32[278600], f32[278600,3]) (1114400 elements):
Array at shape index {0}, 
Mismatch count 356 (0.1278%) in shape f32[278600] (278600 elements), abs bound 0.01, rel bound 0.1
nan mismatches 27
Top relative error mismatches:
  actual                    nan, expected             4846.56934, index {62482}, rel error      inf, abs error      inf
  actual             2950.32617, expected                    nan, index {37263}, rel error      inf, abs error      inf
Absolute magnitude breakdown of actual values:
  0      <= x < 0.0001 :       0 (  0.0000%)
  0.0001 <= x < 0.001  :       0 (  0.0000%)
  0.001  <= x < 0.01   :       0 (  0.0000%)
  0.01   <= x < 0.1    :       0 (  0.0000%)
  0.1    <= x < 1      :       1 (  0.0004%)
  1      <= x < inf    :  278599 ( 99.9996%), mismatches 356
Elements exceeding abs error bound 0.01: 225398 (80.9038%)
Relative error breakdown of elements exceeding abs error bound:
  <  0.0001 :      19 (0.0084%)
  >= 0.0001 :  225379 (99.9916%)
  >= 0.001  :   90291 (40.0585%)
  >= 0.01   :    2516 (1.1162%)
  >= 0.1    :     356 (0.1579%)
  >= 1      :      48 (0.0213%)
Elements exceeding rel error bound 0.1: 356 (0.1278%)
Absolute error breakdown of elements exceeding rel error bound:
  <  0.0001 :       0 (0.0000%)
  >= 0.0001 :     356 (100.0000%)
  >= 0.001  :     356 (100.0000%)
  >= 0.01   :     356 (100.0000%)
  >= 0.1    :     356 (100.0000%)
  >= 1      :     356 (100.0000%)
Array at shape index {1}, 
Mismatch count 123 (0.1%) in shape f32[278600,3]
Top relative error mismatches:
  actual             1234.0, expected             5678.0, index {100}, rel error      inf, abs error      inf
Elements exceeding abs error bound 0.01: 100 (50.5%)
Elements exceeding rel error bound 0.1: 50 (25.2%)
)";

  ASSERT_OK_AND_ASSIGN(std::vector<NumericMismatch> mismatches,
                       ExtractTopMismatches(error_message,
                                            /*is_tuple=*/true));
  EXPECT_EQ(mismatches.size(), 2);
  EXPECT_EQ(mismatches[0].output_shape_index(), 0);

  EXPECT_TRUE(std::isnan(mismatches[0].actual()));
  EXPECT_EQ(mismatches[0].expected(), 4846.56934);
  EXPECT_EQ(mismatches[0].rel_error(), std::numeric_limits<double>::infinity());
  EXPECT_EQ(mismatches[0].percentage_of_elems_exceeding_abs_error(), 80.9038);
  EXPECT_EQ(mismatches[0].percentage_of_elems_exceeding_rel_error(), 0.1278);
  EXPECT_EQ(mismatches[0].percentage_of_elems_exceeding_both_errors(), 0.1278);

  EXPECT_EQ(mismatches[1].output_shape_index(), 1);
  EXPECT_EQ(mismatches[1].actual(), 1234.0);
  EXPECT_EQ(mismatches[1].expected(), 5678.0);
  EXPECT_EQ(mismatches[1].rel_error(), std::numeric_limits<double>::infinity());
  EXPECT_EQ(mismatches[1].percentage_of_elems_exceeding_abs_error(), 50.5);
  EXPECT_EQ(mismatches[1].percentage_of_elems_exceeding_rel_error(), 25.2);
  EXPECT_EQ(mismatches[1].percentage_of_elems_exceeding_both_errors(), 0.1);
}

TEST_F(HloIsolationTest, TestExtractTopMismatches2) {
  std::string error_message = R"(
Mismatch count 356 (0.1278%) in shape f32[278600] (278600 elements), abs bound 0.01, rel bound 0.1
nan mismatches 27
Top relative error mismatches:
  actual                    nan, expected             4846.56934, index {62482}, rel error      inf, abs error      inf
  actual             2950.32617, expected                    nan, index {37263}, rel error      inf, abs error      inf
Absolute magnitude breakdown of actual values:
  0      <= x < 0.0001 :       0 (  0.0000%)
  0.0001 <= x < 0.001  :       0 (  0.0000%)
  0.001  <= x < 0.01   :       0 (  0.0000%)
  0.01   <= x < 0.1    :       0 (  0.0000%)
  0.1    <= x < 1      :       1 (  0.0004%)
  1      <= x < inf    :  278599 ( 99.9996%), mismatches 356
Elements exceeding abs error bound 0.01: 225398 (80.9038%)
Relative error breakdown of elements exceeding abs error bound:
  <  0.0001 :      19 (0.0084%)
  >= 0.0001 :  225379 (99.9916%)
  >= 0.001  :   90291 (40.0585%)
  >= 0.01   :    2516 (1.1162%)
  >= 0.1    :     356 (0.1579%)
  >= 1      :      48 (0.0213%)
Elements exceeding rel error bound 0.1: 356 (0.1278%)
Absolute error breakdown of elements exceeding rel error bound:
  <  0.0001 :       0 (0.0000%)
  >= 0.0001 :     356 (100.0000%)
  >= 0.001  :     356 (100.0000%)
  >= 0.01   :     356 (100.0000%)
  >= 0.1    :     356 (100.0000%)
  >= 1      :     356 (100.0000%)
)";

  ASSERT_OK_AND_ASSIGN(std::vector<NumericMismatch> mismatches,
                       ExtractTopMismatches(error_message,
                                            /*is_tuple=*/false));
  EXPECT_EQ(mismatches.size(), 1);
  EXPECT_EQ(mismatches[0].output_shape_index(), 0);

  EXPECT_TRUE(std::isnan(mismatches[0].actual()));
  EXPECT_EQ(mismatches[0].expected(), 4846.56934);
  EXPECT_EQ(mismatches[0].rel_error(), std::numeric_limits<double>::infinity());
  EXPECT_EQ(mismatches[0].percentage_of_elems_exceeding_abs_error(), 80.9038);
  EXPECT_EQ(mismatches[0].percentage_of_elems_exceeding_rel_error(), 0.1278);
  EXPECT_EQ(mismatches[0].percentage_of_elems_exceeding_both_errors(), 0.1278);
}

TEST_F(HloIsolationTest, TestLiteralContainsInfOrNan) {
  std::vector<std::string> nan_or_inf_strs = {"nan", "-nan", "inf", "-inf"};
  std::vector<float> nan_or_inf_values;
  for (const std::string& nan_or_inf_str : nan_or_inf_strs) {
    float nan_or_inf;
    CHECK(absl::SimpleAtof(nan_or_inf_str, &nan_or_inf));
    nan_or_inf_values.push_back(nan_or_inf);
  }
  Array2D<float> values = {{1.0f, 2.0f}, {3.0f, 4.0f}};
  for (float nan_or_inf : nan_or_inf_values) {
    values(1, 1) = nan_or_inf;
    Literal literal_with_nan = LiteralUtil::CreateR2FromArray2D<float>(values);
    EXPECT_TRUE(LiteralContainsInfOrNan(literal_with_nan));
  }
  // Let's test a tuple as well
  Literal literal_with_nan_tuple = LiteralUtil::MakeTupleOwned(
      LiteralUtil::CreateR2FromArray2D<float>(values),
      LiteralUtil::CreateR2FromArray2D<float>(values));
  EXPECT_TRUE(LiteralContainsInfOrNan(literal_with_nan_tuple));
}

TEST_F(HloIsolationTest, MakeFakeArgumentsForDynamicSliceKnownBits) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(R"(
HloModule test_module

ENTRY %main (param_1: s8[262144,2048], param_2: s32[]) -> s8[131072,2048] {
  %param_1 = s8[262144,2048] parameter(0)
  %param_2 = s32[] parameter(1)
  %constant = s32[] constant(0)
  ROOT %dynamic-slice = s8[131072,2048] dynamic-slice(%param_1, %param_2, %constant), dynamic_slice_sizes={131072,2048}
}
)"));

  ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> args,
      MakeFakeArguments(module.get(),
                        /*pseudo_random=*/true,
                        /*use_large_range=*/false,
                        /*treat_gte_as_data_formatting=*/false,
                        /*max_bits_of_precision=*/std::nullopt,
                        /*engine=*/nullptr,
                        /*generate_aligned_ds_indices=*/false,
                        [](const HloInstruction* use,
                           int64_t sliced_dim) -> std::optional<uint64_t> {
                          if (use->opcode() == HloOpcode::kDynamicSlice &&
                              sliced_dim == 0) {
                            return 131071;
                          }
                          return std::nullopt;
                        }));
  ASSERT_EQ(args.size(), 2);

  int32_t index = args[1].Get<int32_t>({});
  int32_t index_known_bits_zero = 131071;
  EXPECT_EQ(index & index_known_bits_zero, 0);
}

TEST_F(HloIsolationTest, MakeFakeArgumentsForDynamicUpdateSliceKnownBits) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnUnverifiedModule(R"(
HloModule test_module

ENTRY %main (param_1: s8[262144,2048], param_2: s8[131072,2048], param_3: s32[]) -> s8[262144,2048] {
  %param_1 = s8[262144,2048] parameter(0)
  %param_2 = s8[131072,2048] parameter(1)
  %param_3 = s32[] parameter(2)
  %constant = s32[] constant(0)
  ROOT %dynamic-update-slice = s8[262144,2048] dynamic-update-slice(%param_1, %param_2, %param_3, %constant)
}
)"));

  ASSERT_OK_AND_ASSIGN(
      std::vector<Literal> args,
      MakeFakeArguments(module.get(),
                        /*pseudo_random=*/true,
                        /*use_large_range=*/false,
                        /*treat_gte_as_data_formatting=*/false,
                        /*max_bits_of_precision=*/std::nullopt,
                        /*engine=*/nullptr,
                        /*generate_aligned_ds_indices=*/false,
                        [](const HloInstruction* use,
                           int64_t sliced_dim) -> std::optional<uint64_t> {
                          if (use->opcode() == HloOpcode::kDynamicUpdateSlice &&
                              sliced_dim == 0) {
                            return 131071;
                          }
                          return std::nullopt;
                        }));
  ASSERT_EQ(args.size(), 3);

  int32_t index = args[2].Get<int32_t>({});
  int32_t index_known_bits_zero = 131071;
  EXPECT_EQ(index & index_known_bits_zero, 0);
}

TEST_F(HloIsolationTest, TestHugeLiterals) {
  const char* hlo_text = R"(
HloModule huge_literal_module

%fused_computation (param.0: f32[100000000]) -> f32[100000000] {
  %param.0 = f32[100000000]{0} parameter(0)
  ROOT %copy = f32[100000000]{0} copy(%param.0)
}

ENTRY %main (param.0: f32[100000000]) -> f32[100000000] {
  %param.0 = f32[100000000]{0} parameter(0)
  ROOT %fusion = f32[100000000]{0} fusion(%param.0), kind=kLoop, calls=%fused_computation
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  RunAndVerifyIsolationTest(*module);
}

class DelegatingRunner : public HloRunnerInterface {
 public:
  explicit DelegatingRunner(HloRunnerInterface* delegate)
      : delegate_(delegate) {}
  HloRunnerInterface* delegate_;
  bool last_run_hlo_passes_ = false;
  int last_hlo_output_callbacks_size_ = -1;

  absl::StatusOr<std::unique_ptr<OpaqueExecutable>> CreateExecutable(
      std::unique_ptr<HloModule> module, bool run_hlo_passes) override {
    last_run_hlo_passes_ = run_hlo_passes;
    return delegate_->CreateExecutable(std::move(module), run_hlo_passes);
  }

  absl::StatusOr<std::unique_ptr<OpaqueExecutable>> DeserializeExecutable(
      absl::string_view serialized) const override {
    return delegate_->DeserializeExecutable(serialized);
  }

  absl::StatusOr<Literal> Execute(std::unique_ptr<HloModule> module,
                                  absl::Span<const Literal* const> arguments,
                                  bool run_hlo_passes) override {
    return delegate_->Execute(std::move(module), arguments, run_hlo_passes);
  }

  absl::StatusOr<Literal> ExecuteWithBufferAssignment(
      std::unique_ptr<HloModule> module,
      const BufferAssignmentProto* buffer_assignment_proto,
      absl::Span<const Literal* const> arguments,
      bool run_hlo_passes) override {
    return delegate_->ExecuteWithBufferAssignment(
        std::move(module), buffer_assignment_proto, arguments, run_hlo_passes);
  }

  absl::StatusOr<std::vector<absl::StatusOr<Literal>>> ExecuteWithExecutable(
      OpaqueExecutable* executable, absl::Span<const Literal* const> arguments,
      int64_t num_repeats) override {
    return delegate_->ExecuteWithExecutable(executable, arguments, num_repeats);
  }

  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::unique_ptr<HloModule> module,
      const ReplicatedExecuteOptions& options) override {
    return delegate_->ExecuteReplicated(std::move(module), options);
  }

  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      std::unique_ptr<HloModule> module,
      const ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment) override {
    return delegate_->ExecuteReplicated(std::move(module), options,
                                        device_assignment);
  }

  absl::StatusOr<std::vector<Literal>> ExecuteReplicatedWithExecutable(
      OpaqueExecutable* executable,
      const ReplicatedExecuteOptions& options) override {
    last_hlo_output_callbacks_size_ = options.hlo_output_callbacks.size();
    for (const auto& cb : options.hlo_output_callbacks) {
      EXPECT_EQ(cb.num_operands, 1);
    }
    return delegate_->ExecuteReplicatedWithExecutable(executable, options);
  }

  absl::StatusOr<std::vector<Literal>> ExecuteReplicatedWithExecutable(
      OpaqueExecutable* executable, const ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment) override {
    last_hlo_output_callbacks_size_ = options.hlo_output_callbacks.size();
    for (const auto& cb : options.hlo_output_callbacks) {
      EXPECT_EQ(cb.num_operands, 1);
    }
    return delegate_->ExecuteReplicatedWithExecutable(executable, options,
                                                      device_assignment);
  }

  absl::StatusOr<std::vector<Literal>> ExecuteReplicated(
      absl::AnyInvocable<OpaqueExecutable*(int64_t)> executable_provider,
      absl::AnyInvocable<int64_t(int64_t)> argument_count_provider,
      absl::AnyInvocable<const Literal*(int64_t, int64_t)> argument_provider,
      const ReplicatedExecuteOptions& options,
      DeviceAssignment* device_assignment) override {
    return delegate_->ExecuteReplicated(
        std::move(executable_provider), std::move(argument_count_provider),
        std::move(argument_provider), options, device_assignment);
  }

  absl::string_view Name() const override { return delegate_->Name(); }

  int device_count() const override { return delegate_->device_count(); }

  bool HasProperty(HloRunnerPropertyTag::Type tag) const override {
    return delegate_->HasProperty(tag);
  }

  absl::StatusOr<const HloModule* absl_nonnull> HloModuleFromWrapped(
      const OpaqueExecutable* wrapped) const override {
    return delegate_->HloModuleFromWrapped(wrapped);
  }

  bool ExecutablesAreEquivalent(const OpaqueExecutable* lhs,
                                const OpaqueExecutable* rhs) const override {
    return delegate_->ExecutablesAreEquivalent(lhs, rhs);
  }

  absl::StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override {
    return delegate_->GetDefaultDeviceAssignment(num_replicas, num_partitions);
  }
};

TEST_F(HloIsolationTest, TestDumpMiscompareFiles) {
  const absl::string_view hlo_string = R"hlo(
HloModule jit_f, is_scheduled=true, entry_computation_layout={(f32[3,6]{1,0:T(4,128)})->f32[3,6]{1,0:T(4,128)}}

fused_computation {
  param_0.1 = f32[3,6]{1,0:T(4,128)S(1)} parameter(0)
  add.0 = f32[3,6]{1,0:T(4,128)} add(param_0.1, param_0.1), metadata={op_name="jit(f)/add" stack_frame_id=13}
  ROOT abs.0 = f32[3,6]{1,0:T(4,128)} abs(add.0), metadata={op_name="jit(f)/abs" stack_frame_id=14}
}

fused_computation.1 {
  param_0.3 = f32[3,6]{1,0:T(4,128)} parameter(0)
  sin.0 = f32[3,6]{1,0:T(4,128)} sine(param_0.3), metadata={op_name="jit(f)/sin" stack_frame_id=10}
  ROOT abs.1 = f32[3,6]{1,0:T(4,128)S(1)} abs(sin.0), metadata={op_name="jit(f)/abs" stack_frame_id=11}
}

ENTRY main.1 {
  x.1 = f32[3,6]{1,0:T(4,128)} parameter(0), metadata={op_name="x"}
  sine_abs_fusion = f32[3,6]{1,0:T(4,128)S(1)} fusion(x.1), kind=kLoop, calls=fused_computation.1, metadata={op_name="jit(f)/abs" stack_frame_id=11}
  ROOT add_abs_fusion = f32[3,6]{1,0:T(4,128)} fusion(sine_abs_fusion), kind=kLoop, calls=fused_computation, metadata={op_name="jit(f)/abs" stack_frame_id=14}
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  const char* env_p = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");
  const std::string outputs_dir = env_p ? env_p : ::testing::TempDir();

  // Proactively clean up any stale matching files from previous runs to
  // prevent false positives.
  std::vector<std::string> stale_paths;
  tsl::Env* env = tsl::Env::Default();
  if (env->GetMatchingPaths(tsl::io::JoinPath(outputs_dir, "*sine_abs_fusion*"),
                            &stale_paths)
          .ok()) {
    for (const std::string& path : stale_paths) {
      (void)env->DeleteFile(path);
    }
  }

  PipelineIsolationOptions options;
  options.module_options.run_module_fn =
      [](std::unique_ptr<HloModule> m, HloRunnerInterface* runner,
         absl::Span<const Literal> input_data,
         const RunModuleOptions& run_opts) -> absl::StatusOr<Literal> {
    const bool should_inject = (m->name() == "sine_abs_fusion");
    ASSIGN_OR_RETURN(Literal output,
                     RunModule(std::move(m), runner, input_data, run_opts));
    if (should_inject) {
      // Flip an exponent bit in the first element to guarantee a mismatch.
      // Using untyped_data handles all primitive types without crashing.
      char* data_ptr = static_cast<char*>(output.untyped_data(
          output.shape().IsTuple() ? ShapeIndex{0} : ShapeIndex{}));
      data_ptr[2] ^= 0x80;
    }
    return output;
  };

  DelegatingRunner test_runner(&this->test_runner());
  DelegatingRunner reference_runner(&this->reference_runner());

  // Directly run the pipeline while intercepting testing failures.
  std::vector<HloIsolationTestResult> pipeline_results;
  ::testing::TestPartResultArray failures;
  {
    ::testing::ScopedFakeTestPartResultReporter reporter(
        ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS,
        &failures);
    ASSERT_OK_AND_ASSIGN(pipeline_results,
                         RunIsolationPipeline(*module, &test_runner,
                                              &reference_runner, options));
  }
  // We expect 2 failures:
  // 1. TPU_VS_DEFUSED_TPU mismatch.
  // 2. TPU_VS_INTERPRETER mismatch.
  EXPECT_EQ(failures.size(), 2);
  EXPECT_GT(test_runner.last_hlo_output_callbacks_size_, 0);

  // Check the pipeline results contains our mismatch results.
  bool found_sine_abs_fusion = false;
  for (const auto& res : pipeline_results) {
    if (res.module_name() == "sine_abs_fusion") {
      found_sine_abs_fusion = true;
      EXPECT_GE(res.numeric_checks_size(), 1);
      for (const auto& check : res.numeric_checks()) {
        EXPECT_GT(check.top_mismatches_size(), 0);
        EXPECT_TRUE(check.has_top_mismatch());
      }
    }
  }
  EXPECT_TRUE(found_sine_abs_fusion);

  // Check that the failed module file is present.
  std::vector<std::string> module_matches;
  EXPECT_TRUE(tsl::Env::Default()
                  ->GetMatchingPaths(
                      tsl::io::JoinPath(outputs_dir,
                                        "failed-module-sine_abs_fusion.txt"),
                      &module_matches)
                  .ok() &&
              !module_matches.empty());
  // Check that the actual, expected, and mismatches files are present.
  for (absl::string_view suffix : {"actual", "expected", "mismatches"}) {
    std::vector<std::string> matches;
    EXPECT_TRUE(tsl::Env::Default()
                    ->GetMatchingPaths(
                        tsl::io::JoinPath(
                            outputs_dir, absl::StrCat("failed-sine_abs_fusion-",
                                                      suffix, ".txt")),
                        &matches)
                    .ok() &&
                !matches.empty());
  }
}

TEST_F(HloIsolationTest, TestInitIsolatorOptionsRunHloPasses) {
  DelegatingRunner test_runner(&this->test_runner());
  DelegatingRunner reference_runner(&this->reference_runner());
  ModuleIsolationOptions options;
  options.run_hlo_passes = true;
  options.make_fake_arguments_fn =
      [](const HloModule&) -> absl::StatusOr<std::vector<Literal>> {
    return std::vector<Literal>{};
  };

  const char* hlo_text = R"(
HloModule SimpleModule

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_text));
  auto result_or = RunIsolationTestOnModule(*module, &test_runner,
                                            &reference_runner, options);
  EXPECT_TRUE(result_or.ok());
  EXPECT_TRUE(test_runner.last_run_hlo_passes_);
}

TEST_F(HloIsolationTest, TestRunModuleUseFusionDebuggerOption) {
  DelegatingRunner test_runner(&this->test_runner());

  const char* hlo_text = R"(
HloModule SimpleModule

ENTRY main {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT add = f32[] add(a, b)
}
)";

  // 1. When use_fusion_debugger is false, callbacks should be empty.
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(hlo_text));
    RunModuleOptions run_opts;
    run_opts.use_fusion_debugger = false;
    std::vector<Literal> args;
    args.push_back(LiteralUtil::CreateR0<float>(1.0f));
    args.push_back(LiteralUtil::CreateR0<float>(2.0f));
    auto output_or = RunModule(std::move(module), &test_runner, args, run_opts);
    ASSERT_TRUE(output_or.ok());
    EXPECT_EQ(test_runner.last_hlo_output_callbacks_size_, 0);
  }

  // 2. When use_fusion_debugger is true, callbacks should be non-empty.
  {
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                         ParseAndReturnVerifiedModule(hlo_text));
    RunModuleOptions run_opts;
    run_opts.use_fusion_debugger = true;
    std::vector<Literal> args;
    args.push_back(LiteralUtil::CreateR0<float>(1.0f));
    args.push_back(LiteralUtil::CreateR0<float>(2.0f));
    auto output_or = RunModule(std::move(module), &test_runner, args, run_opts);
    ASSERT_TRUE(output_or.ok());
    EXPECT_GT(test_runner.last_hlo_output_callbacks_size_, 0);
  }
}

TEST_F(HloIsolationTest, TestPopulateNumericCheckMismatches) {
  NumericCheck numeric_check;

  // 1. Error status
  absl::StatusOr<std::vector<NumericMismatch>> error_status =
      absl::InternalError("Failed extraction");
  PopulateNumericCheckMismatches(&numeric_check, error_status);
  EXPECT_EQ(numeric_check.top_mismatches_size(), 0);
  EXPECT_FALSE(numeric_check.has_top_mismatch());

  // 2. Empty vector
  absl::StatusOr<std::vector<NumericMismatch>> empty_vector =
      std::vector<NumericMismatch>{};
  PopulateNumericCheckMismatches(&numeric_check, empty_vector);
  EXPECT_EQ(numeric_check.top_mismatches_size(), 0);
  EXPECT_FALSE(numeric_check.has_top_mismatch());

  // 3. Ok status with mismatches
  NumericMismatch mismatch1;
  mismatch1.set_rel_error(0.5);
  NumericMismatch mismatch2;
  mismatch2.set_rel_error(1.5);
  absl::StatusOr<std::vector<NumericMismatch>> valid_vector =
      std::vector<NumericMismatch>{mismatch1, mismatch2};
  PopulateNumericCheckMismatches(&numeric_check, valid_vector);
  EXPECT_EQ(numeric_check.top_mismatches_size(), 2);
  ASSERT_TRUE(numeric_check.has_top_mismatch());
  EXPECT_DOUBLE_EQ(numeric_check.top_mismatch().rel_error(), 1.5);

  // 4. Calling PopulateNumericCheckMismatches again should clear the old list
  NumericMismatch mismatch3;
  mismatch3.set_rel_error(2.5);
  absl::StatusOr<std::vector<NumericMismatch>> valid_vector2 =
      std::vector<NumericMismatch>{mismatch3};
  PopulateNumericCheckMismatches(&numeric_check, valid_vector2);
  EXPECT_EQ(numeric_check.top_mismatches_size(), 1);
  ASSERT_TRUE(numeric_check.has_top_mismatch());
  EXPECT_DOUBLE_EQ(numeric_check.top_mismatch().rel_error(), 2.5);
}

}  // namespace
}  // namespace hlo_isolation
}  // namespace xla
