/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <utility>

#include "tensorflow/compiler/xla/service/gpu/gpu_executable.h"
#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/filecheck.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {
namespace gpu {

namespace {

class ReductionVectorizationTest : public GpuCodegenTest {};

TEST_F(ReductionVectorizationTest, Power2) {
  const char* hlo_text = R"(
HloModule ReducePower2

%max_ {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %maximum.7 = f32[] maximum(f32[] %x, f32[] %y)
}

ENTRY %main {
  %param_0 = f32[5,131072] parameter(0)
  %constant.3 = f32[] constant(0)
  ROOT %reduce.8 = f32[5] reduce(f32[5,131072] %param_0, f32[] %constant.3), dimensions={1}, to_apply=%max_
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> optimized_module,
                          ParseAndReturnVerifiedModule(hlo_text));
  CompileAndOptionallyVerifyPtx(std::move(optimized_module),
                                R"(
CHECK: ld.global.nc.v2.f32
CHECK: ld.global.nc.v2.f32
CHECK: ld.global.nc.v2.f32
CHECK: ld.global.nc.v2.f32
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(ReductionVectorizationTest, TileFit) {
  const char* hlo_text = R"(
HloModule ReduceTileFit

%max_ {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %maximum.7 = f32[] maximum(f32[] %x, f32[] %y)
}

ENTRY %main {
  %param_0 = f32[5,122880] parameter(0)
  %constant.3 = f32[] constant(0)
  ROOT %reduce.8 = f32[5] reduce(f32[5,122880] %param_0, f32[] %constant.3), dimensions={1}, to_apply=%max_
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> optimized_module,
                          ParseAndReturnVerifiedModule(hlo_text));
  CompileAndOptionallyVerifyPtx(std::move(optimized_module),
                                R"(
CHECK: ld.global.nc.v2.f32
CHECK: ld.global.nc.v2.f32
CHECK: ld.global.nc.v2.f32
CHECK: ld.global.nc.v2.f32
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(ReductionVectorizationTest, EvenColumns) {
  const char* hlo_text = R"(
HloModule ReducePower2

%max_ {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %maximum.7 = f32[] maximum(f32[] %x, f32[] %y)
}

ENTRY %main {
  %param_0 = f32[5,131070] parameter(0)
  %constant.3 = f32[] constant(0)
  ROOT %reduce.8 = f32[5] reduce(f32[5,131070] %param_0, f32[] %constant.3), dimensions={1}, to_apply=%max_
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> optimized_module,
                          ParseAndReturnVerifiedModule(hlo_text));
  CompileAndOptionallyVerifyPtx(std::move(optimized_module),
                                R"(
CHECK: ld.global.nc.f32
CHECK: ld.global.nc.f32
CHECK: ld.global.nc.v2.f32
CHECK: ld.global.nc.v2.f32
CHECK: ld.global.nc.v2.f32
CHECK-NOT: ld.global.nc.v2.f32
// TODO: Make this a vectorized load
CHECK: ld.global.nc.f32
CHECK: ld.global.nc.f32
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(ReductionVectorizationTest, DisableOddColumns) {
  const char* hlo_text = R"(
HloModule ReduceTileFit

%max_ {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %maximum.7 = f32[] maximum(%x, %y)
}

ENTRY %main {
  %param_0 = f32[5,131071] parameter(0)
  %constant.3 = f32[] constant(0)
  ROOT %reduce.8 = f32[5] reduce(f32[5,131071] %param_0, f32[] %constant.3), dimensions={1}, to_apply=%max_
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> optimized_module,
                          ParseAndReturnVerifiedModule(hlo_text));
  CompileAndOptionallyVerifyPtx(std::move(optimized_module),
                                R"(
CHECK-NOT: ld.global.nc.v2.f32
CHECK-NOT: ld.global.nc.v4.f32
CHECK-NOT: ld.global.nc.u64
CHECK-NOT: ld.global.u64
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(ReductionVectorizationTest, Exp) {
  const char* hlo_text = R"(
HloModule DisableSin

%add_float {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add.17 = f32[] add(f32[] %x, f32[] %y)
}

ENTRY %main {
  %arg0.1 = f32[5,131072] parameter(0)
  %sine = f32[5,131072] exponential(f32[5,131072] %arg0.1)
  %constant.0 = f32[] constant(0)
  ROOT %reduce.18 = f32[5] reduce(f32[5,131072] %sine, f32[] %constant.0), dimensions={1}, to_apply=%add_float
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> optimized_module,
                          ParseAndReturnVerifiedModule(hlo_text));
  CompileAndOptionallyVerifyPtx(std::move(optimized_module),
                                R"(
CHECK: ld.global.nc.v2.f32
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(ReductionVectorizationTest, DisableSin) {
  const char* hlo_text = R"(
HloModule DisableSin

%add_float {
  %x = f32[] parameter(0)
  %y = f32[] parameter(1)
  ROOT %add.17 = f32[] add(f32[] %x, f32[] %y)
}

ENTRY %main {
  %arg0.1 = f32[5,131072] parameter(0)
  %sine = f32[5,131072] sine(f32[5,131072] %arg0.1)
  %constant.0 = f32[] constant(0)
  ROOT %reduce.18 = f32[5] reduce(f32[5,131072] %sine, f32[] %constant.0), dimensions={1}, to_apply=%add_float
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> optimized_module,
                          ParseAndReturnVerifiedModule(hlo_text));
  CompileAndOptionallyVerifyPtx(std::move(optimized_module),
                                R"(
CHECK-NOT: ld.global.nc.v2.f32
CHECK-NOT: ld.global.nc.v4.f32
CHECK-NOT: ld.global.nc.u64
CHECK-NOT: ld.global.u64
)");

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
