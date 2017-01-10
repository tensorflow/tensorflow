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

#include "tensorflow/compiler/xla/shape_tree.h"

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class ShapeTreeTest : public ::testing::Test {
 protected:
  ShapeTreeTest() {
    array_shape_ = ShapeUtil::MakeShape(F32, {42, 42, 123});
    tuple_shape_ =
        ShapeUtil::MakeTupleShape({array_shape_, array_shape_, array_shape_});
    nested_tuple_shape_ = ShapeUtil::MakeTupleShape(
        {array_shape_, ShapeUtil::MakeTupleShape({array_shape_, array_shape_}),
         ShapeUtil::MakeTupleShape(
             {ShapeUtil::MakeTupleShape({array_shape_, array_shape_}),
              array_shape_})});
  }

  // An array shape (non-tuple).
  Shape array_shape_;

  // A three element tuple shape.
  Shape tuple_shape_;

  // A nested tuple shape of the following form: (a, (c, d), ((e, f), g))
  Shape nested_tuple_shape_;
};

TEST_F(ShapeTreeTest, ArrayShape) {
  ShapeTree<int> shape_tree{array_shape_};
  *shape_tree.mutable_element({}) = 42;
  EXPECT_EQ(42, shape_tree.element({}));
  *shape_tree.mutable_element({}) = 123;
  EXPECT_EQ(123, shape_tree.element({}));

  EXPECT_TRUE(ShapeUtil::Compatible(array_shape_, shape_tree.shape()));

  // Test the copy constructor.
  ShapeTree<int> copy{shape_tree};
  EXPECT_EQ(123, copy.element({}));
}

TEST_F(ShapeTreeTest, TupleShape) {
  ShapeTree<int> shape_tree{tuple_shape_};
  *shape_tree.mutable_element({}) = 1;
  *shape_tree.mutable_element({0}) = 42;
  *shape_tree.mutable_element({1}) = 123;
  *shape_tree.mutable_element({2}) = -100;
  EXPECT_EQ(1, shape_tree.element({}));
  EXPECT_EQ(42, shape_tree.element({0}));
  EXPECT_EQ(123, shape_tree.element({1}));
  EXPECT_EQ(-100, shape_tree.element({2}));

  EXPECT_TRUE(ShapeUtil::Compatible(tuple_shape_, shape_tree.shape()));

  // Sum all elements in the shape.
  int sum = 0;
  TF_CHECK_OK(shape_tree.ForEachElement(
      [&sum](const ShapeIndex& /*index*/, bool /*is_leaf*/, int data) {
        sum += data;
        return tensorflow::Status::OK();
      }));
  EXPECT_EQ(66, sum);

  // Test the copy constructor.
  ShapeTree<int> copy{shape_tree};
  EXPECT_EQ(1, copy.element({}));
  EXPECT_EQ(42, copy.element({0}));
  EXPECT_EQ(123, copy.element({1}));
  EXPECT_EQ(-100, copy.element({2}));

  // Write zero to all data elements.
  TF_CHECK_OK(shape_tree.ForEachMutableElement(
      [&sum](const ShapeIndex& /*index*/, bool /*is_leaf*/, int* data) {
        *data = 0;
        return tensorflow::Status::OK();
      }));
  EXPECT_EQ(0, shape_tree.element({}));
  EXPECT_EQ(0, shape_tree.element({0}));
  EXPECT_EQ(0, shape_tree.element({1}));
  EXPECT_EQ(0, shape_tree.element({2}));
}

TEST_F(ShapeTreeTest, NestedTupleShape) {
  ShapeTree<int> shape_tree{nested_tuple_shape_};
  *shape_tree.mutable_element({0}) = 42;
  *shape_tree.mutable_element({1, 1}) = 123;
  *shape_tree.mutable_element({2, 0, 1}) = -100;
  EXPECT_EQ(42, shape_tree.element({0}));
  EXPECT_EQ(123, shape_tree.element({1, 1}));
  EXPECT_EQ(-100, shape_tree.element({2, 0, 1}));

  EXPECT_TRUE(ShapeUtil::Compatible(nested_tuple_shape_, shape_tree.shape()));

  // Test the copy constructor.
  ShapeTree<int> copy{shape_tree};
  EXPECT_EQ(42, copy.element({0}));
  EXPECT_EQ(123, copy.element({1, 1}));
  EXPECT_EQ(-100, copy.element({2, 0, 1}));
}

TEST_F(ShapeTreeTest, InvalidIndexingTuple) {
  ShapeTree<int> shape_tree{tuple_shape_};

  EXPECT_DEATH(shape_tree.element({4}), "");
}

TEST_F(ShapeTreeTest, InvalidIndexingNestedTuple) {
  ShapeTree<int> shape_tree{nested_tuple_shape_};

  EXPECT_DEATH(shape_tree.element({0, 0}), "");
}

}  // namespace
}  // namespace xla
