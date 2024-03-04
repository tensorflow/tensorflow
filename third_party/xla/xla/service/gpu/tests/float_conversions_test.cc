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

#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {

class FloatConversionTest : public GpuCodegenTest {};

TEST_F(FloatConversionTest, F8E5M2ToF16) {
  EXPECT_TRUE(RunAndCompare(R"(ENTRY m {
                                 %p = f8e5m2[] parameter(0)
                                 ROOT %c = f16[] convert(%p)
                               })",
                            ErrorSpec{1e-5, 1e-5}));
}

TEST_F(FloatConversionTest, F8E4M3FNToF16) {
  EXPECT_TRUE(RunAndCompare(R"(ENTRY m {
                                 %p = f8e4m3fn[] parameter(0)
                                 ROOT %c = f16[] convert(%p)
                               })",
                            ErrorSpec{1e-5, 1e-5}));
}

TEST_F(FloatConversionTest, F8E4M3B11FNUZToF16) {
  EXPECT_TRUE(RunAndCompare(R"(ENTRY m {
                                 %p = f8e4m3b11fnuz[] parameter(0)
                                 ROOT %c = f16[] convert(%p)
                               })",
                            ErrorSpec{1e-5, 1e-5}));
}

TEST_F(FloatConversionTest, F8E5M2FNUZToF16) {
  EXPECT_TRUE(RunAndCompare(R"(ENTRY m {
                                 %p = f8e5m2fnuz[] parameter(0)
                                 ROOT %c = f16[] convert(%p)
                               })",
                            ErrorSpec{1e-5, 1e-5}));
}

TEST_F(FloatConversionTest, F8E4M3FNUZToF16) {
  EXPECT_TRUE(RunAndCompare(R"(ENTRY m {
                                 %p = f8e4m3fnuz[] parameter(0)
                                 ROOT %c = f16[] convert(%p)
                               })",
                            ErrorSpec{1e-5, 1e-5}));
}

TEST_F(FloatConversionTest, BF16ToF32) {
  EXPECT_TRUE(RunAndCompare(R"(ENTRY m {
                                 %p = bf16[] parameter(0)
                                 ROOT %c = f32[] convert(%p)
                               })",
                            ErrorSpec{1e-5, 1e-5}));
}

TEST_F(FloatConversionTest, F16ToF32) {
  EXPECT_TRUE(RunAndCompare(R"(ENTRY m {
                                 %p = f16[] parameter(0)
                                 ROOT %c = f32[] convert(%p)
                               })",
                            ErrorSpec{1e-5, 1e-5}));
}

TEST_F(FloatConversionTest, F64ToF32) {
  EXPECT_TRUE(RunAndCompare(R"(ENTRY m {
                                 %p = f64[] parameter(0)
                                 ROOT %c = f32[] convert(%p)
                               })",
                            ErrorSpec{1e-5, 1e-5}));
}

TEST_F(FloatConversionTest, F16ToF8E5M2) {
  EXPECT_TRUE(RunAndCompare(R"(ENTRY m {
                                 %p = f16[] parameter(0)
                                 ROOT %c = f8e5m2[] convert(%p)
                               })",
                            ErrorSpec{1e-5, 1e-5}));
}

TEST_F(FloatConversionTest, F16ToF8E4M3FN) {
  EXPECT_TRUE(RunAndCompare(R"(ENTRY m {
                                 %p = f16[] parameter(0)
                                 ROOT %c = f8e4m3fn[] convert(%p)
                               })",
                            ErrorSpec{1e-5, 1e-5}));
}

TEST_F(FloatConversionTest, F16ToF8E4M3B11FNUZ) {
  EXPECT_TRUE(RunAndCompare(R"(ENTRY m {
                                 %p = f16[] parameter(0)
                                 ROOT %c = f8e4m3b11fnuz[] convert(%p)
                               })",
                            ErrorSpec{1e-5, 1e-5}));
}

TEST_F(FloatConversionTest, F16ToF8E5M2FNUZ) {
  EXPECT_TRUE(RunAndCompare(R"(ENTRY m {
                                 %p = f16[] parameter(0)
                                 ROOT %c = f8e5m2fnuz[] convert(%p)
                               })",
                            ErrorSpec{1e-5, 1e-5}));
}

TEST_F(FloatConversionTest, F16ToF8E4M3FNUZ) {
  EXPECT_TRUE(RunAndCompare(R"(ENTRY m {
                                 %p = f16[] parameter(0)
                                 ROOT %c = f8e4m3fnuz[] convert(%p)
                               })",
                            ErrorSpec{1e-5, 1e-5}));
}

TEST_F(FloatConversionTest, F32ToBF16) {
  EXPECT_TRUE(RunAndCompare(R"(ENTRY m {
                                 %p = f32[] parameter(0)
                                 ROOT %c = bf16[] convert(%p)
                               })",
                            ErrorSpec{1e-5, 1e-5}));
}

TEST_F(FloatConversionTest, F32ToF16) {
  EXPECT_TRUE(RunAndCompare(R"(ENTRY m {
                                 %p = f32[] parameter(0)
                                 ROOT %c = f16[] convert(%p)
                               })",
                            ErrorSpec{1e-5, 1e-5}));
}

TEST_F(FloatConversionTest, F32ToF64) {
  EXPECT_TRUE(RunAndCompare(R"(ENTRY m {
                                 %p = f32[] parameter(0)
                                 ROOT %c = f64[] convert(%p)
                               })",
                            ErrorSpec{1e-5, 1e-5}));
}

}  // namespace gpu
}  // namespace xla
