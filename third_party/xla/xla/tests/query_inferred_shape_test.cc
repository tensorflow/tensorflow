/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/builder/xla_builder.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

TEST(QueryInferredShapeTest, OnePlusOneShape) {
  XlaBuilder builder("one_plus_one");
  XlaOp one = ConstantR0<float>(&builder, 1.0);
  XlaOp result = Add(one, one);
  TF_ASSERT_OK_AND_ASSIGN(const Shape shape, builder.GetShape(result));
  ASSERT_TRUE(ShapeUtil::Equal(shape, ShapeUtil::MakeShape(F32, {})));
}

}  // namespace
}  // namespace xla
