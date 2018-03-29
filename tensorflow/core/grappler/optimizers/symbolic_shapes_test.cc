/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/symbolic_shapes.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class SymbolicShapesTest : public ::testing::Test {
 protected:
  TensorShapeProto MakeUnknown() {
    TensorShapeProto shape;
    shape.set_unknown_rank(true);
    return shape;
  }

  TensorShapeProto MakeShape(std::vector<int> dims) {
    TensorShapeProto shape;
    for (int dim_size : dims) {
      TensorShapeProto::Dim dim;
      dim.set_size(dim_size);
      *shape.add_dim() = dim;
    }
    return shape;
  }
};

bool operator<(const TensorShapeProto& lhs, const TensorShapeProto& rhs) {
  return CompareSymbolicallyShapedTensorSizes(lhs, rhs);
}

TEST_F(SymbolicShapesTest, ShapeIsSymbolicallyDefined) {
  EXPECT_FALSE(ShapeIsSymbolicallyDefined(MakeUnknown()));
  EXPECT_FALSE(ShapeIsSymbolicallyDefined(MakeShape({-1, 2})));

  EXPECT_TRUE(ShapeIsSymbolicallyDefined(MakeShape({1, 2})));
  EXPECT_TRUE(ShapeIsSymbolicallyDefined(MakeShape({-2, 2})));
}

TEST_F(SymbolicShapesTest, ShapesSymbolicallyEqual) {
  EXPECT_FALSE(ShapesSymbolicallyEqual(MakeUnknown(), MakeUnknown()));
  EXPECT_FALSE(ShapesSymbolicallyEqual(MakeShape({-1, 2}), MakeShape({-1, 2})));
  EXPECT_FALSE(ShapesSymbolicallyEqual(MakeShape({-2, 2}), MakeShape({-3, 2})));

  EXPECT_TRUE(ShapesSymbolicallyEqual(MakeShape({1, 2}), MakeShape({1, 2})));
  EXPECT_TRUE(ShapesSymbolicallyEqual(MakeShape({-2, 2}), MakeShape({-2, 2})));
}

TEST_F(SymbolicShapesTest, ShapesBroadcastable) {
  EXPECT_FALSE(ShapesBroadcastable(MakeUnknown(), MakeUnknown()));
  EXPECT_FALSE(ShapesBroadcastable(MakeShape({-2}), MakeShape({1, -3})));
  EXPECT_FALSE(ShapesBroadcastable(MakeShape({-1, 2}), MakeShape({-1, 2})));
  EXPECT_FALSE(ShapesBroadcastable(MakeShape({-2, 2}), MakeShape({-3, 2})));
  EXPECT_FALSE(ShapesBroadcastable(MakeShape({-2, 4}), MakeShape({-2, 8})));

  EXPECT_TRUE(ShapesBroadcastable(MakeShape({1, 2}), MakeShape({1, 2})));
  EXPECT_TRUE(ShapesBroadcastable(MakeShape({-2, 2}), MakeShape({-2, 2})));
  EXPECT_TRUE(ShapesBroadcastable(MakeShape({-2, 32}), MakeShape({-2, 1})));
  EXPECT_TRUE(ShapesBroadcastable(MakeShape({-2, 1}), MakeShape({1, -2})));
  EXPECT_TRUE(ShapesBroadcastable(MakeShape({-2, 1}), MakeShape({1, -3})));
  EXPECT_TRUE(ShapesBroadcastable(MakeShape({-3}), MakeShape({-2, -3})));
}

TEST_F(SymbolicShapesTest, CompareSymbolicallyShapedTensorSizes) {
  EXPECT_TRUE(MakeShape({1, 1, 32}) < MakeShape({32, 32}));
  EXPECT_TRUE(MakeShape({1, 32, 32}) < MakeShape({2048}));
  EXPECT_TRUE(MakeShape({1, -2, 32}) < MakeShape({-2, 32, 32}));
  EXPECT_TRUE(MakeShape({1, 32, 32}) < MakeShape({-2, 32, 32}));
  EXPECT_TRUE(MakeShape({1, 32, 32}) < MakeShape({-1, 32, 32}));
  EXPECT_TRUE(MakeShape({1, -2, 32}) < MakeShape({-2, -2, 32}));

  EXPECT_FALSE(MakeShape({1, -2, 32}) < MakeShape({-3, 32, 32}));
  EXPECT_FALSE(MakeShape({1, -1, 32}) < MakeShape({1, -1, 32}));
  EXPECT_FALSE(MakeShape({1, -1, 32}) < MakeShape({-1, -1, 32}));
  EXPECT_FALSE(MakeShape({-1, -1, 32}) < MakeShape({1, -1, 32}));
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
