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

#include "xla/service/gpu/fusions/concatenate_mlir.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "xla/error_spec.h"
#include "xla/service/gpu/fusions/mlir/mlir_fusion_emitter.h"
#include "xla/service/gpu/fusions/mlir_emitter_test_base.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/tests/filecheck.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class MlirConcatenateFusionTest : public MlirEmitterTestBase {
 public:
  std::unique_ptr<MlirFusionEmitterBase> GetEmitter(
      const HloFusionAnalysis& analysis) override {
    return std::make_unique<MlirConcatenateFusion>(analysis);
  }
};

TEST_F(MlirConcatenateFusionTest, StandAloneConcatenate) {
  auto kHloString = R"(
    HloModule module

  fused_computation {
    param0 = f32[128] parameter(0)
    param1 = f32[128] parameter(1)
    ROOT concat = f32[256] concatenate(param0, param1), dimensions={0}
  }

  ENTRY main {
    param0 = f32[128] parameter(0)
    param1 = f32[128] parameter(1)
    ROOT fusion = f32[256] fusion(param0, param1), calls=fused_computation, kind=kLoop
  }
  )";

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(MlirConcatenateFusionTest, PrologueEpilogue) {
  auto kHloString = R"(
    HloModule module

  fused_computation {
    param0 = f32[128] parameter(0)
    param1 = f32[128] parameter(1)
    log = f32[128] log(param0)
    exp = f32[128] exponential(param1)
    concat = f32[256] concatenate(log, exp), dimensions={0}
    ROOT neg = f32[256] negate(concat)
  }

  ENTRY main {
    param0 = f32[128] parameter(0)
    param1 = f32[128] parameter(1)
    ROOT fusion = f32[256] fusion(param0, param1), calls=fused_computation, kind=kLoop
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto ir, EmitIR(kHloString));

  ASSERT_TRUE(RunFileCheck(ir, R"(
// CHECK-LABEL: fused_computation
// CHECK: %[[C_128:.*]] = arith.constant 128
// CHECK: %[[THREAD_ID:.*]] = gpu.thread_id x

// CHECK: %[[VAL_1_1:.*]] = xla_gpu.pure_call @fused_computation_log({{.*}}, %[[THREAD_ID]])
// CHECK: %[[VAL_1_2:.*]] = xla_gpu.pure_call @fused_computation_neg({{.*}}, %[[THREAD_ID]], %[[VAL_1_1]])
// CHECK: %[[INSERTED_1:.*]] = tensor.insert %[[VAL_1_2:.*]] into {{.*}}[%[[THREAD_ID]]]

// CHECK: %[[VAL_2_1:.*]] = xla_gpu.pure_call @fused_computation_exp({{.*}}, %[[THREAD_ID]])
// CHECK: %[[INDEX_2:.*]] = arith.addi %[[THREAD_ID]], %[[C_128]]
// CHECK: %[[VAL_2_2:.*]] = xla_gpu.pure_call @fused_computation_neg({{.*}}, %[[INDEX_2]], %[[VAL_2_1]])
// CHECK: %[[INSERTED_2:.*]] = tensor.insert %[[VAL_2_2:.*]] into {{.*}}[%[[INDEX_2]]]

// CHECK: return %[[INSERTED_2]]

// CHECK: func.func private @fused_computation_log
// CHECK: func.func private @fused_computation_exp
// CHECK: func.func private @fused_computation_neg
)")
                  .value());

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(MlirConcatenateFusionTest, Prologue) {
  auto kHloString = R"(
    HloModule module

  fused_computation {
    param0 = f32[128] parameter(0)
    param1 = f32[128] parameter(1)
    log = f32[128] log(param0)
    exp = f32[128] exponential(param1)
    ROOT concat = f32[256] concatenate(log, exp), dimensions={0}
  }

  ENTRY main {
    param0 = f32[128] parameter(0)
    param1 = f32[128] parameter(1)
    ROOT fusion = f32[256] fusion(param0, param1), calls=fused_computation, kind=kLoop
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto ir, EmitIR(kHloString));

  ASSERT_TRUE(RunFileCheck(ir, R"(
// CHECK-LABEL: fused_computation
// CHECK:       %[[C_128:.*]] = arith.constant 128
// CHECK:       %[[THREAD_ID:.*]] = gpu.thread_id x

// CHECK:       %[[VAL_1:.*]] = xla_gpu.pure_call @fused_computation_log({{.*}}, %[[THREAD_ID]])
// CHECK:       %[[INSERTED_1:.*]] = tensor.insert %[[VAL_1:.*]] into {{.*}}[%[[THREAD_ID]]]

// CHECK:       %[[VAL_2:.*]] = xla_gpu.pure_call @fused_computation_exp({{.*}}, %[[THREAD_ID]])
// CHECK:       %[[INDEX_2:.*]] = arith.addi %[[THREAD_ID]], %[[C_128]]
// CHECK:       %[[INSERTED_2:.*]] = tensor.insert %[[VAL_2:.*]] into {{.*}}[%[[INDEX_2]]]

// CHECK:       return %[[INSERTED_2]]

// CHECK: func.func private @fused_computation_log
// CHECK: func.func private @fused_computation_exp
)")
                  .value());

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(MlirConcatenateFusionTest, MajorDimension) {
  auto kHloString = R"(
  HloModule module

  fused_computation {
    param0 = f32[16,16] parameter(0)
    param1 = f32[16,16] parameter(1)
    ROOT concat = f32[32,16] concatenate(param0, param1), dimensions={0}
  }

  ENTRY main {
    param0 = f32[16,16] parameter(0)
    param1 = f32[16,16] parameter(1)
    ROOT %fusion = f32[32,16] fusion(param0, param1), kind=kInput, calls=fused_computation
  }
  )";

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloString, ErrorSpec{1e-3}));
}

TEST_F(MlirConcatenateFusionTest, DifferentDimensions) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule module

  fused_computation {
    param0 = f32[400] parameter(0)
    param1 = f32[200] parameter(1)
    log = f32[400] log(param0)
    exp = f32[200] exponential(param1)
    ROOT concat = f32[600] concatenate(log, exp), dimensions={0}
  }

  ENTRY main {
    param0 = f32[400] parameter(0)
    param1 = f32[200] parameter(1)
    ROOT fusion = f32[600] fusion(param0, param1), calls=fused_computation, kind=kLoop
  }
  )"));

  auto* root = module->entry_computation()->root_instruction();
  auto analysis = AnalyzeFusion(*root, device_info_);

  // TODO: Add support for different operand sizes.
  EXPECT_FALSE(MlirConcatenateFusion::IsSupported(analysis));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
