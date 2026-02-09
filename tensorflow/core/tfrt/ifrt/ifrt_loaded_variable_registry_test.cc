/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/ifrt/ifrt_loaded_variable_registry.h"

#include <memory>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/hash/hash.h"
#include "absl/hash/hash_testing.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace tensorflow {
namespace ifrt_serving {
namespace {

TEST(IfrtLoadedVariableRegistryTest, KeyEquality) {
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::F32, {2, 2});
  auto shape_ptr1 = std::make_shared<xla::Shape>(shape);
  auto shape_ptr2 = std::make_shared<xla::Shape>(shape);
  // shape_ptr3 points to a different shape value.
  xla::Shape different_shape = xla::ShapeUtil::MakeShape(xla::F32, {3, 3});
  auto shape_ptr3 = std::make_shared<xla::Shape>(different_shape);

  IfrtLoadedVariableRegistry::Key key1{
      .device_ids = {0, 1},
      .input_name = "input",
      .hlo_sharding = xla::HloSharding::Replicate(),
      .shape_on_device = shape_ptr1,
  };

  IfrtLoadedVariableRegistry::Key key2{
      .device_ids = {0, 1},
      .input_name = "input",
      .hlo_sharding = xla::HloSharding::Replicate(),
      .shape_on_device = shape_ptr2,
  };

  IfrtLoadedVariableRegistry::Key key3{
      .device_ids = {0, 1},
      .input_name = "input",
      .hlo_sharding = xla::HloSharding::Replicate(),
      .shape_on_device = shape_ptr3,
  };

  EXPECT_EQ(key1, key2);
  EXPECT_NE(key1, key3);
}

TEST(IfrtLoadedVariableRegistryTest, KeyHash) {
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::F32, {2, 2});
  auto shape_ptr1 = std::make_shared<xla::Shape>(shape);
  auto shape_ptr2 = std::make_shared<xla::Shape>(shape);
  xla::Shape different_shape = xla::ShapeUtil::MakeShape(xla::F32, {3, 3});
  auto shape_ptr3 = std::make_shared<xla::Shape>(different_shape);

  IfrtLoadedVariableRegistry::Key key1{
      .device_ids = {0, 1},
      .input_name = "input",
      .hlo_sharding = xla::HloSharding::Replicate(),
      .shape_on_device = shape_ptr1,
  };

  IfrtLoadedVariableRegistry::Key key2{
      .device_ids = {0, 1},
      .input_name = "input",
      .hlo_sharding = xla::HloSharding::Replicate(),
      .shape_on_device = shape_ptr2,
  };
  IfrtLoadedVariableRegistry::Key key3{
      .device_ids = {0, 1},
      .input_name = "input",
      .hlo_sharding = xla::HloSharding::Replicate(),
      .shape_on_device = shape_ptr3,
  };

  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({key1, key2, key3}));
  EXPECT_EQ(absl::Hash<IfrtLoadedVariableRegistry::Key>()(key1),
            absl::Hash<IfrtLoadedVariableRegistry::Key>()(key2));
  // Note: hash collision is possible but unlikely for key1 and key3.
}

TEST(IfrtLoadedVariableRegistryTest, KeyToString) {
  xla::Shape shape = xla::ShapeUtil::MakeShape(xla::F32, {2, 2});
  auto shape_ptr = std::make_shared<xla::Shape>(shape);

  IfrtLoadedVariableRegistry::Key key{
      .device_ids = {0, 1},
      .input_name = "input",
      .hlo_sharding = xla::HloSharding::Replicate(),
      .shape_on_device = shape_ptr,
  };

  EXPECT_THAT(key.ToString(),
              ::testing::HasSubstr("input:0,1:{replicated}:f32[2,2]"));
}

}  // namespace
}  // namespace ifrt_serving
}  // namespace tensorflow
