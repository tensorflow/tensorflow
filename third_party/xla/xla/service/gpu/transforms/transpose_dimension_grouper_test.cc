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

#include "xla/service/gpu/transforms/transpose_dimension_grouper.h"

#include <optional>

#include "absl/strings/string_view.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {
using ::testing::HasSubstr;
using ::tsl::testing::StatusIs;

class TransposeDimensionGrouperTest : public HloHardwareIndependentTestBase {
 public:
  void CheckDimensionGrouper(absl::string_view hlo,
                             std::optional<absl::string_view> expected) {
    RunAndFilecheckHloRewrite(hlo, TransposeDimensionGrouper{}, expected);
  }
  void CheckDimensionGrouperUnchanged(absl::string_view hlo) {
    CheckDimensionGrouper(hlo, /*expected=*/std::nullopt);
  }
};

TEST_F(TransposeDimensionGrouperTest, NoTranspose) {
  const char* hlo = R"(
HloModule NoTranspose

ENTRY main {
  input = f32[64,128,1]{2,1,0} parameter(0)
  ROOT out = f32[64,1,128]{2,1,0} transpose(input), dimensions={0,2,1}
}
)";

  // After normalization, it becomes the identity permutation, so we don't
  // normalize the transpose in this pass. It would be replaced with a bitcast
  // by AlgebraicSimplifier.
  CheckDimensionGrouperUnchanged(hlo);
}

TEST_F(TransposeDimensionGrouperTest, NoTranspose2) {
  const char* hlo = R"(
HloModule NoTranspose2

ENTRY main {
  input = f32[32,128,64]{2,1,0} parameter(0)
  ROOT out = f32[32,64,128]{0,1,2} transpose(input), dimensions={0,2,1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  TransposeDimensionGrouper dimension_grouper;
  EXPECT_THAT(dimension_grouper.Run(module.get()),
              StatusIs(tsl::error::FAILED_PRECONDITION,
                       HasSubstr("Layout normalization")));
}

TEST_F(TransposeDimensionGrouperTest, NoTranspose3) {
  const char* hlo = R"(
HloModule NoTranspose3

ENTRY main {
  input = f32[32,128,64]{0,1,2} parameter(0)
  ROOT out = f32[32,64,128]{2,1,0} transpose(input), dimensions={0,2,1}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  TransposeDimensionGrouper dimension_grouper;
  EXPECT_THAT(dimension_grouper.Run(module.get()),
              StatusIs(tsl::error::FAILED_PRECONDITION,
                       HasSubstr("Layout normalization")));
}

TEST_F(TransposeDimensionGrouperTest, Simple2D) {
  const char* hlo = R"(
HloModule Simple2D

ENTRY main {
  input = f32[128,64]{1,0} parameter(0)
  ROOT out = f32[64,128]{1,0} transpose(input), dimensions={1,0}
}
)";

  CheckDimensionGrouperUnchanged(hlo);
}

TEST_F(TransposeDimensionGrouperTest, Simple3D_021) {
  const char* hlo = R"(
HloModule Simple3D_021

ENTRY main {
  input = f32[8,32768,16]{2,1,0} parameter(0)
  ROOT out = f32[8,16,32768]{2,1,0} transpose(input), dimensions={0,2,1}
}
)";

  // The transpose is already normalized.
  CheckDimensionGrouperUnchanged(hlo);
}

TEST_F(TransposeDimensionGrouperTest, Simple3D_210) {
  const char* hlo = R"(
HloModule Simple3D_210

ENTRY main {
  input = f32[8,32768,16]{2,1,0} parameter(0)
  ROOT out = f32[16,32768,8]{2,1,0} transpose(input), dimensions={2,1,0}
}
)";

  // The transpose is already normalized.
  CheckDimensionGrouperUnchanged(hlo);
}

TEST_F(TransposeDimensionGrouperTest, Simple4D) {
  const char* hlo = R"(
HloModule Simple4D

ENTRY main {
  input = f32[32768,4,16,8]{3,2,1,0} parameter(0)
  ROOT out = f32[16,32768,8,4]{3,2,1,0} transpose(input), dimensions={2,0,3,1}
}
)";

  // The transpose is already normalized.
  CheckDimensionGrouperUnchanged(hlo);
}

TEST_F(TransposeDimensionGrouperTest, NormalizeTo3D) {
  const char* hlo = R"(
HloModule NormalizeTo3D

ENTRY main {
  input = f32[8,32,32,32,16]{4,3,2,1,0} parameter(0)
  ROOT out = f32[8,16,32,32,32]{4,3,2,1,0} transpose(input), dimensions={0,4,1,2,3}
}
)";

  CheckDimensionGrouper(hlo,
                        R"(
// CHECK:  [[input_0:%[^ ]+]] = f32[8,32,32,32,16]{4,3,2,1,0} parameter(0)
// CHECK:  [[bitcast_1:%[^ ]+]] = f32[8,32768,16]{2,1,0} bitcast([[input_0]])
// CHECK:  [[transpose:%[^ ]+]] = f32[8,16,32768]{2,1,0} transpose([[bitcast_1]]), dimensions={0,2,1}
// CHECK:  ROOT {{.*}} = f32[8,16,32,32,32]{4,3,2,1,0} bitcast([[transpose]])
      )");
}

TEST_F(TransposeDimensionGrouperTest, LargeShapeSizeOverflow) {
  const char* hlo = R"(
  HloModule LargeShapeSizeOverflow

ENTRY main {
  input = f32[4096,4096,128,16]{3,2,1,0} parameter(0)
  ROOT out = f32[16,4096,4096,128]{3,2,1,0} transpose(input), dimensions={3,0,1,2}
}
)";

  CheckDimensionGrouper(hlo,
                        R"(
// CHECK:  [[input_0:%[^ ]+]] = f32[4096,4096,128,16]{3,2,1,0} parameter(0)
// CHECK:  [[bitcast_1:%[^ ]+]] = f32[2147483648,16]{1,0} bitcast([[input_0]])
// CHECK:  [[transpose:%[^ ]+]] = f32[16,2147483648]{1,0} transpose([[bitcast_1]]), dimensions={1,0}
// CHECK:  ROOT {{.*}} = f32[16,4096,4096,128]{3,2,1,0} bitcast([[transpose]])
      )");
}

TEST_F(TransposeDimensionGrouperTest, DegenerateDims) {
  const char* hlo = R"(
  HloModule DegenerateDims

ENTRY main {
  input = f32[1,32,1,3,1,64,1]{6,5,4,3,2,1,0} parameter(0)
  ROOT out = f32[1,32,1,64,1,3,1]{6,5,4,3,2,1,0} transpose(input), dimensions={6,1,4,5,2,3,0}
}
)";

  CheckDimensionGrouper(hlo,
                        R"(
// CHECK:  [[input_0:%[^ ]+]] = f32[1,32,1,3,1,64,1]{6,5,4,3,2,1,0} parameter(0)
// CHECK:  [[bitcast_1:%[^ ]+]] = f32[32,3,64]{2,1,0} bitcast([[input_0]])
// CHECK:  [[transpose:%[^ ]+]] = f32[32,64,3]{2,1,0} transpose([[bitcast_1]]), dimensions={0,2,1}
// CHECK:  ROOT {{.*}} = f32[1,32,1,64,1,3,1]{6,5,4,3,2,1,0} bitcast([[transpose]])
      )");
}

TEST_F(TransposeDimensionGrouperTest, TransposeWithGrouping) {
  const char* hlo = R"(
HloModule TransposeWithGrouping

ENTRY main {
  input = f32[100,1,10,32,2]{4,3,2,1,0} parameter(0)
  ROOT out = f32[10,1,32,100,2]{4,3,2,1,0} transpose(input), dimensions={2,1,3,0,4}
}
)";

  CheckDimensionGrouper(hlo,
                        R"(
// CHECK:  [[input_0:%[^ ]+]] = f32[100,1,10,32,2]{4,3,2,1,0} parameter(0)
// CHECK:  [[bitcast_1:%[^ ]+]] = f32[100,320,2]{2,1,0} bitcast([[input_0]])
// CHECK:  [[transpose:%[^ ]+]] = f32[320,100,2]{2,1,0} transpose([[bitcast_1]]), dimensions={1,0,2}
// CHECK:  ROOT {{.*}} = f32[10,1,32,100,2]{4,3,2,1,0} bitcast([[transpose]])
      )");
}

TEST_F(TransposeDimensionGrouperTest, NormalizeTo2D) {
  const char* hlo = R"(
HloModule Normalize2DTo3D

ENTRY main {
  input = f32[50,20,30]{2,1,0} parameter(0)
  ROOT out = f32[20,30,50]{2,1,0} transpose(input), dimensions={1,2,0}
}
)";

  CheckDimensionGrouper(hlo,
                        R"(
// CHECK:  [[input_0:%[^ ]+]] = f32[50,20,30]{2,1,0} parameter(0)
// CHECK:  [[bitcast_1:%[^ ]+]] = f32[50,600]{1,0} bitcast([[input_0]])
// CHECK:  [[transpose:%[^ ]+]] = f32[600,50]{1,0} transpose([[bitcast_1]]), dimensions={1,0}
// CHECK:  ROOT {{.*}} = f32[20,30,50]{2,1,0} bitcast([[transpose]])
      )");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
