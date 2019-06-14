/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/tools/data_initializer.h"

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

#include <stdlib.h>

namespace xla {
namespace poplarplugin {
namespace {

using DataInitializerTest = HloTestBase;

const std::vector<PrimitiveType> supported_types = {
    PRED, S8, U8, S16, U16, S32, U32, S64, U64, F16, F32};
const std::vector<std::vector<int64>> dimensions_to_test = {{}, {4}, {4, 4}};

TEST_F(DataInitializerTest, TestRandomDataInitializer) {
  // Check that the initializer parses correctly.
  auto initializer = DataInitializer::GetDataInitializer("random");
  for (auto dimensions : dimensions_to_test) {
    for (auto type : supported_types) {
      auto shape = ShapeUtil::MakeShape(type, dimensions);
      // Check that the literal can be created given the type and shape.
      TF_ASSERT_OK_AND_ASSIGN(auto literal, initializer->GetData(shape));
      // For floating point, check that non of the values are NaNs/Infs.
      if (ShapeUtil::ElementIsFloating(shape)) {
        int64 num_elements = ShapeUtil::ElementsIn(shape);
        TF_ASSERT_OK_AND_ASSIGN(auto literal_flat,
                                literal.Reshape({num_elements}));
        for (int64 i = 0; i < num_elements; i++) {
          float value;
          switch (type) {
            case F16:
              value = Eigen::half_impl::half_to_float(
                  literal_flat.Get<Eigen::half>({i}));
              break;
            case F32:
              value = literal_flat.Get<float>({i});
              break;
            default:
              // Unsupported type - should never happen.
              EXPECT_TRUE(false);
              break;
          }
          EXPECT_FALSE(std::isnan(value));
          EXPECT_FALSE(std::isinf(value));
        }
      }
    }
  }
}

TEST_F(DataInitializerTest, TestRandomNormalDataInitializer) {
  // Check that the initializer parses correctly.
  auto initializer = DataInitializer::GetDataInitializer("normal");
  for (auto dimensions : dimensions_to_test) {
    for (auto type : supported_types) {
      auto shape = ShapeUtil::MakeShape(type, dimensions);
      // Check that the literal can be created given the type and shape.
      TF_ASSERT_OK_AND_ASSIGN(auto literal, initializer->GetData(shape));
      // For floating point, check that non of the values are NaNs/Infs.
      if (ShapeUtil::ElementIsFloating(shape)) {
        int64 num_elements = ShapeUtil::ElementsIn(shape);
        TF_ASSERT_OK_AND_ASSIGN(auto literal_flat,
                                literal.Reshape({num_elements}));
        for (int64 i = 0; i < num_elements; i++) {
          float value;
          switch (type) {
            case F16:
              value = Eigen::half_impl::half_to_float(
                  literal_flat.Get<Eigen::half>({i}));
              break;
            case F32:
              value = literal_flat.Get<float>({i});
              break;
            default:
              // Unsupported type - should never happen.
              EXPECT_TRUE(false);
              break;
          }
          EXPECT_FALSE(std::isnan(value));
          EXPECT_FALSE(std::isinf(value));
        }
      }
    }
  }
}

TEST_F(DataInitializerTest, TestConstantDataInitializer) {
  for (auto value : {0, 1}) {
    // Check that the initializer parses correctly.
    auto initializer =
        DataInitializer::GetDataInitializer(std::to_string(value));
    for (auto dimensions : dimensions_to_test) {
      for (auto type : supported_types) {
        auto shape = ShapeUtil::MakeShape(type, dimensions);
        // Check that the literal can be created given the type and shape.
        TF_ASSERT_OK_AND_ASSIGN(auto literal, initializer->GetData(shape));
        // Check all the values are the same.
        EXPECT_TRUE(literal.IsAll(value));
      }
    }
  }
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
