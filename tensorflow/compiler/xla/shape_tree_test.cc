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
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

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

  void TestShapeConstructor(const Shape& shape, int expected_num_nodes);
  void TestInitValueConstructor(const Shape& shape, int expected_num_nodes);

  // An array shape (non-tuple).
  Shape array_shape_;

  // A three element tuple shape.
  Shape tuple_shape_;

  // A nested tuple shape of the following form: (a, (c, d), ((e, f), g))
  Shape nested_tuple_shape_;
};

TEST_F(ShapeTreeTest, DefaultConstructor) {
  ShapeTree<int> int_tree;
  EXPECT_TRUE(ShapeUtil::IsNil(int_tree.shape()));

  ShapeTree<bool> bool_tree;
  EXPECT_TRUE(ShapeUtil::IsNil(bool_tree.shape()));
}

void ShapeTreeTest::TestShapeConstructor(const Shape& shape,
                                         int expected_num_nodes) {
  ShapeTree<int> int_tree(shape);
  int num_nodes = 0;
  int_tree.ForEachElement([&num_nodes](const ShapeIndex& /*index*/, int data) {
    EXPECT_EQ(0, data);
    ++num_nodes;
  });
  EXPECT_EQ(expected_num_nodes, num_nodes);

  ShapeTree<bool> bool_tree(shape);
  num_nodes = 0;
  bool_tree.ForEachElement(
      [&num_nodes](const ShapeIndex& /*index*/, bool data) {
        EXPECT_EQ(false, data);
        ++num_nodes;
      });
  EXPECT_EQ(expected_num_nodes, num_nodes);
}

TEST_F(ShapeTreeTest, ShapeConstructor) {
  TestShapeConstructor(array_shape_, 1);
  TestShapeConstructor(tuple_shape_, 4);
  TestShapeConstructor(nested_tuple_shape_, 10);
}

void ShapeTreeTest::TestInitValueConstructor(const Shape& shape,
                                             int expected_num_nodes) {
  ShapeTree<int> tree(shape, 42);
  int num_nodes = 0;
  tree.ForEachElement([&num_nodes](const ShapeIndex& /*index*/, int data) {
    EXPECT_EQ(42, data);
    ++num_nodes;
  });
  EXPECT_EQ(expected_num_nodes, num_nodes);

  num_nodes = 0;
  tree.ForEachMutableElement(
      [&num_nodes](const ShapeIndex& /*index*/, int* data) {
        EXPECT_EQ(42, *data);
        *data = num_nodes;
        ++num_nodes;
      });
  EXPECT_EQ(expected_num_nodes, num_nodes);

  num_nodes = 0;
  tree.ForEachElement([&num_nodes](const ShapeIndex& /*index*/, int data) {
    EXPECT_EQ(num_nodes, data);
    ++num_nodes;
  });
  EXPECT_EQ(expected_num_nodes, num_nodes);
}

TEST_F(ShapeTreeTest, InitValueConstructor) {
  TestInitValueConstructor(array_shape_, 1);
  TestInitValueConstructor(tuple_shape_, 4);
  TestInitValueConstructor(nested_tuple_shape_, 10);
}

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

  // Mutate the copy, and ensure the original doesn't change.
  *copy.mutable_element({}) = 99;
  EXPECT_EQ(99, copy.element({}));
  EXPECT_EQ(123, shape_tree.element({}));

  // Test the assignment operator.
  copy = shape_tree;
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
  shape_tree.ForEachElement(
      [&sum](const ShapeIndex& /*index*/, int data) { sum += data; });
  EXPECT_EQ(66, sum);

  // Test the copy constructor.
  ShapeTree<int> copy{shape_tree};
  EXPECT_EQ(1, copy.element({}));
  EXPECT_EQ(42, copy.element({0}));
  EXPECT_EQ(123, copy.element({1}));
  EXPECT_EQ(-100, copy.element({2}));

  // Write zero to all data elements.
  shape_tree.ForEachMutableElement(
      [&sum](const ShapeIndex& /*index*/, int* data) { *data = 0; });
  EXPECT_EQ(0, shape_tree.element({}));
  EXPECT_EQ(0, shape_tree.element({0}));
  EXPECT_EQ(0, shape_tree.element({1}));
  EXPECT_EQ(0, shape_tree.element({2}));
  EXPECT_EQ(1, copy.element({}));
  EXPECT_EQ(42, copy.element({0}));
  EXPECT_EQ(123, copy.element({1}));
  EXPECT_EQ(-100, copy.element({2}));

  // Test the assignment operator.
  copy = shape_tree;
  EXPECT_EQ(0, copy.element({}));
  EXPECT_EQ(0, copy.element({0}));
  EXPECT_EQ(0, copy.element({1}));
  EXPECT_EQ(0, copy.element({2}));
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

  // Mutate the copy, and ensure the original doesn't change.
  *copy.mutable_element({0}) = 1;
  *copy.mutable_element({1, 1}) = 2;
  *copy.mutable_element({2, 0, 1}) = 3;
  EXPECT_EQ(1, copy.element({0}));
  EXPECT_EQ(2, copy.element({1, 1}));
  EXPECT_EQ(3, copy.element({2, 0, 1}));
  EXPECT_EQ(42, shape_tree.element({0}));
  EXPECT_EQ(123, shape_tree.element({1, 1}));
  EXPECT_EQ(-100, shape_tree.element({2, 0, 1}));

  // Test the assignment operator.
  copy = shape_tree;
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

TEST_F(ShapeTreeTest, ShapeTreeOfNonCopyableType) {
  ShapeTree<std::unique_ptr<int>> shape_tree{tuple_shape_};
  EXPECT_EQ(shape_tree.element({2}).get(), nullptr);
  *shape_tree.mutable_element({2}) = MakeUnique<int>(42);
  EXPECT_EQ(*shape_tree.element({2}), 42);
}

TEST_F(ShapeTreeTest, CopySubtreeFromArrayShape) {
  // Test CopySubtreeFrom method for a single value copied between array-shaped
  // ShapeTrees.
  ShapeTree<int> source(array_shape_);
  *source.mutable_element(/*index=*/{}) = 42;
  ShapeTree<int> destination(array_shape_, 123);

  EXPECT_EQ(destination.element(/*index=*/{}), 123);
  destination.CopySubtreeFrom(source, /*source_base_index=*/{},
                              /*target_base_index=*/{});
  EXPECT_EQ(destination.element(/*index=*/{}), 42);
}

TEST_F(ShapeTreeTest, FullCopySubtreeFromTupleShape) {
  // Test CopySubtreeFrom method for a copy of all elements from one
  // tuple-shaped ShapeTree to another.
  ShapeTree<int> source(tuple_shape_);
  *source.mutable_element(/*index=*/{}) = 10;
  *source.mutable_element(/*index=*/{0}) = 11;
  *source.mutable_element(/*index=*/{1}) = 12;
  *source.mutable_element(/*index=*/{2}) = 13;

  ShapeTree<int> destination(tuple_shape_, 0);

  destination.CopySubtreeFrom(source, /*source_base_index=*/{},
                              /*target_base_index=*/{});
  EXPECT_EQ(destination.element(/*index=*/{}), 10);
  EXPECT_EQ(destination.element(/*index=*/{0}), 11);
  EXPECT_EQ(destination.element(/*index=*/{1}), 12);
  EXPECT_EQ(destination.element(/*index=*/{2}), 13);
}

TEST_F(ShapeTreeTest, SingleElementCopySubtreeFromTupleShape) {
  // Test CopySubtreeFrom method for a copy of a single element from one
  // tuple-shaped ShapeTree to another.
  ShapeTree<int> source(tuple_shape_);
  *source.mutable_element(/*index=*/{}) = 10;
  *source.mutable_element(/*index=*/{0}) = 11;
  *source.mutable_element(/*index=*/{1}) = 12;
  *source.mutable_element(/*index=*/{2}) = 13;

  ShapeTree<int> destination(tuple_shape_, 0);

  destination.CopySubtreeFrom(source, /*source_base_index=*/{0},
                              /*target_base_index=*/{1});
  EXPECT_EQ(destination.element(/*index=*/{}), 0);
  EXPECT_EQ(destination.element(/*index=*/{0}), 0);
  EXPECT_EQ(destination.element(/*index=*/{1}), 11);
  EXPECT_EQ(destination.element(/*index=*/{2}), 0);
}

TEST_F(ShapeTreeTest, CopySubtreeIntoNestedShape) {
  // Test CopySubtreeFrom method for a copy of a tuple-shaped ShapeTree into a
  // nested-tuple-shaped ShapeTree.
  ShapeTree<int> source(
      ShapeUtil::MakeTupleShape({array_shape_, array_shape_}));
  *source.mutable_element(/*index=*/{}) = 10;
  *source.mutable_element(/*index=*/{0}) = 11;
  *source.mutable_element(/*index=*/{1}) = 12;

  ShapeTree<int> destination(nested_tuple_shape_, 0);

  destination.CopySubtreeFrom(source, /*source_base_index=*/{},
                              /*target_base_index=*/{2, 0});

  EXPECT_EQ(destination.element(/*index=*/{}), 0);
  EXPECT_EQ(destination.element(/*index=*/{0}), 0);
  EXPECT_EQ(destination.element(/*index=*/{1}), 0);
  EXPECT_EQ(destination.element(/*index=*/{1, 0}), 0);
  EXPECT_EQ(destination.element(/*index=*/{1, 1}), 0);
  EXPECT_EQ(destination.element(/*index=*/{2}), 0);
  EXPECT_EQ(destination.element(/*index=*/{2, 0}), 10);
  EXPECT_EQ(destination.element(/*index=*/{2, 0, 0}), 11);
  EXPECT_EQ(destination.element(/*index=*/{2, 0, 1}), 12);
  EXPECT_EQ(destination.element(/*index=*/{2, 1}), 0);
}

TEST_F(ShapeTreeTest, CopySubtreeFromNestedShape) {
  // Test CopySubtreeFrom method for a copy from a nested-tuple-shape.
  ShapeTree<int> source(nested_tuple_shape_, 42);
  *source.mutable_element(/*index=*/{1}) = 10;
  *source.mutable_element(/*index=*/{1, 0}) = 11;
  *source.mutable_element(/*index=*/{1, 1}) = 12;

  ShapeTree<int> destination(
      ShapeUtil::MakeTupleShape({array_shape_, array_shape_}), 0);

  destination.CopySubtreeFrom(source, /*source_base_index=*/{1},
                              /*target_base_index=*/{});

  EXPECT_EQ(destination.element(/*index=*/{}), 10);
  EXPECT_EQ(destination.element(/*index=*/{0}), 11);
  EXPECT_EQ(destination.element(/*index=*/{1}), 12);
}

TEST_F(ShapeTreeTest, OperatorEquals) {
  {
    ShapeTree<int> a(array_shape_, 123);
    ShapeTree<int> b(array_shape_, 42);
    ShapeTree<int> c(array_shape_, 42);
    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_TRUE(b == c);
  }
  {
    ShapeTree<int> a(tuple_shape_);
    *a.mutable_element(/*index=*/{}) = 10;
    *a.mutable_element(/*index=*/{0}) = 11;
    *a.mutable_element(/*index=*/{1}) = 12;

    ShapeTree<int> b(tuple_shape_);
    *b.mutable_element(/*index=*/{}) = 10;
    *b.mutable_element(/*index=*/{0}) = 42;
    *b.mutable_element(/*index=*/{1}) = 11;

    ShapeTree<int> c(tuple_shape_);
    *c.mutable_element(/*index=*/{}) = 10;
    *c.mutable_element(/*index=*/{0}) = 42;
    *c.mutable_element(/*index=*/{1}) = 11;

    EXPECT_FALSE(a == b);
    EXPECT_TRUE(a != b);
    EXPECT_TRUE(b == c);
    EXPECT_FALSE(b != c);
  }
}

}  // namespace
}  // namespace xla
