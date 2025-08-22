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

#include "xla/service/computation_layout.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace {
using ::testing::ElementsAre;

TEST(ComputationLayoutTest, ParameterShapes) {
  ProgramShape program_shape;
  program_shape.AddParameter(ShapeUtil::MakeShape(F32, {1, 2, 3}), "p0");
  program_shape.AddParameter(ShapeUtil::MakeShape(U8, {1}), "p1");
  *program_shape.mutable_result() = ShapeUtil::MakeShape(F64, {2});
  ComputationLayout layout(program_shape, /*ignore_layouts=*/true);

  EXPECT_THAT(layout.parameter_shapes(),
              ElementsAre(ShapeUtil::MakeShape(F32, {1, 2, 3}),
                          ShapeUtil::MakeShape(U8, {1})));
}

}  // namespace
}  // namespace xla
