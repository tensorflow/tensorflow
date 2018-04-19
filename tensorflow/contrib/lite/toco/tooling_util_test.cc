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
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"

namespace toco {

enum class Agreement { kBroadcast, kExtend, kBroadcastNotExtend, kNeither };

// A pair of Shapes and whether they should agree up to broadcasting, extending
// or neither.
struct ShapePair {
  Shape left;
  Shape right;
  Agreement agreement;
};

std::vector<ShapePair> CreateShapePairs() {
  return std::vector<ShapePair>(
      {// These agree up to broadcast.
       {Shape({3}), Shape({3}), Agreement::kBroadcast},
       {Shape({256, 256, 3}), Shape({256, 256, 3}), Agreement::kBroadcast},
       {Shape({256, 256, 3}), Shape({3}), Agreement::kBroadcast},
       {Shape({8, 1, 6, 1}), Shape({7, 1, 5}), Agreement::kBroadcast},

       // These extend (and therefore broadcast).
       {Shape({3}), Shape({3}), Agreement::kExtend},
       {Shape({256, 256, 3}), Shape({256, 256, 3}), Agreement::kExtend},
       {Shape({1, 1, 3}), Shape({1, 1, 3}), Agreement::kExtend},
       {Shape({1, 1, 3}), Shape({3}), Agreement::kExtend},
       {Shape({1, 1, 3}), Shape({1, 3}), Agreement::kExtend},

       // These strictly broadcast and do not extend.
       {Shape({256, 256, 3}), Shape({3}), Agreement::kBroadcastNotExtend},
       {Shape({5, 4}), Shape({1}), Agreement::kBroadcastNotExtend},
       {Shape({5, 4}), Shape({4}), Agreement::kBroadcastNotExtend},
       {Shape({15, 3, 5}), Shape({15, 1, 5}), Agreement::kBroadcastNotExtend},
       {Shape({15, 3, 5}), Shape({3, 5}), Agreement::kBroadcastNotExtend},
       {Shape({15, 3, 5}), Shape({3, 1}), Agreement::kBroadcastNotExtend},

       // These do not broadcast (and therefore also do not extend).
       {Shape({3}), Shape({4}), Agreement::kNeither},
       {Shape({2, 1}), Shape({8, 4, 3}), Agreement::kNeither}});
}

// ShapeTest is an empty parameterized test fixture since there is no state.
class ShapeTest : public ::testing::TestWithParam<ShapePair> {};

TEST_P(ShapeTest, Agrees) {
  const ShapePair& param = GetParam();

  switch (param.agreement) {
    case Agreement::kBroadcast: {
      EXPECT_TRUE(ShapesAgreeUpToBroadcasting(param.left, param.right));
      break;
    }
    case Agreement::kExtend: {
      EXPECT_TRUE(ShapesAgreeUpToExtending(param.left, param.right));
      // Anything that extends should also broadcast.
      EXPECT_TRUE(ShapesAgreeUpToBroadcasting(param.left, param.right));
      break;
    }
    case Agreement::kBroadcastNotExtend: {
      // Verify that it strictly broadcasts but does not extend.
      EXPECT_TRUE(ShapesAgreeUpToBroadcasting(param.left, param.right));
      EXPECT_FALSE(ShapesAgreeUpToExtending(param.left, param.right));
      break;
    }
    case Agreement::kNeither: {
      EXPECT_FALSE(ShapesAgreeUpToExtending(param.left, param.right));
      EXPECT_FALSE(ShapesAgreeUpToBroadcasting(param.left, param.right));
      break;
    }
  }
}

INSTANTIATE_TEST_CASE_P(AgreeBroadcast, ShapeTest,
                        ::testing::ValuesIn(CreateShapePairs()));

}  // namespace toco
