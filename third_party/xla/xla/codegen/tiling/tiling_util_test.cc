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

#include "xla/codegen/tiling/tiling_util.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/shape.h"

namespace xla {
namespace {

using TilingUtilTest = HloHardwareIndependentTestBase;

TEST_F(TilingUtilTest, IsSameShapeMultiOutputFusion_SameShapes) {
  const char* hlo_string = R"(
    HloModule m

    f {
      p0 = f32[10,20] parameter(0)
      p1 = f32[10,20] parameter(1)
      ROOT tuple = (f32[10,20], f32[10,20]) tuple(p0, p1)
    }

    ENTRY entry {
      p0 = f32[10,20] parameter(0)
      p1 = f32[10,20] parameter(1)
      ROOT fusion = (f32[10,20], f32[10,20]) fusion(p0, p1), kind=kLoop, calls=f
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  std::unique_ptr<HloFusionAdaptor> adaptor = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->root_instruction());
  EXPECT_TRUE(
      IsSameShapeMultiOutputFusion(adaptor->GetRoots(), Shape::Equal()));
}

TEST_F(TilingUtilTest, IsSameShapeMultiOutputFusion_DifferentShapes) {
  const char* hlo_string = R"(
    HloModule m

    f {
      p0 = f32[10,20] parameter(0)
      p1 = f32[10,21] parameter(1)
      ROOT tuple = (f32[10,20], f32[10,21]) tuple(p0, p1)
    }

    ENTRY entry {
      p0 = f32[10,20] parameter(0)
      p1 = f32[10,21] parameter(1)
      ROOT fusion = (f32[10,20], f32[10,21]) fusion(p0, p1), kind=kLoop, calls=f
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  std::unique_ptr<HloFusionAdaptor> adaptor = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->root_instruction());
  EXPECT_FALSE(
      IsSameShapeMultiOutputFusion(adaptor->GetRoots(), Shape::Equal()));
}

TEST_F(TilingUtilTest, IsSameShapeMultiOutputFusion_CustomComparator) {
  const char* hlo_string = R"(
    HloModule m

    f {
      p0 = f32[10,20] parameter(0)
      p1 = s32[10,20] parameter(1)
      ROOT tuple = (f32[10,20], s32[10,20]) tuple(p0, p1)
    }

    ENTRY entry {
      p0 = f32[10,20] parameter(0)
      p1 = s32[10,20] parameter(1)
      ROOT fusion = (f32[10,20], s32[10,20]) fusion(p0, p1), kind=kLoop, calls=f
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  std::unique_ptr<HloFusionAdaptor> adaptor = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->root_instruction());
  EXPECT_TRUE(IsSameShapeMultiOutputFusion(adaptor->GetRoots(),
                                           Shape::Equal().IgnoreElementType()));
}

TEST_F(TilingUtilTest, IsSameShapeMultiOutputFusion_SingleOutput) {
  const char* hlo_string = R"(
    HloModule m

    f {
      ROOT p0 = f32[10,20] parameter(0)
    }

    ENTRY entry {
      p0 = f32[10,20] parameter(0)
      ROOT fusion = f32[10,20] fusion(p0), kind=kLoop, calls=f
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  std::unique_ptr<HloFusionAdaptor> adaptor = HloFusionAdaptor::ForInstruction(
      module->entry_computation()->root_instruction());
  EXPECT_FALSE(
      IsSameShapeMultiOutputFusion(adaptor->GetRoots(), Shape::Equal()));
}

}  // namespace
}  // namespace xla
