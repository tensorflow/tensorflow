/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace {

class QueryInferredShapeTest : public ClientLibraryTestBase {};

TEST_F(QueryInferredShapeTest, OnePlusOneShape) {
  XlaBuilder builder("one_plus_one");
  auto one = ConstantR0<float>(&builder, 1.0);
  auto result = Add(one, one);
  StatusOr<Shape> shape_status = builder.GetShape(result);
  ASSERT_IS_OK(shape_status.status());
  auto shape = shape_status.value();
  ASSERT_TRUE(ShapeUtil::Equal(shape, ShapeUtil::MakeShape(F32, {})));
}

}  // namespace
}  // namespace xla
