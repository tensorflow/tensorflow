/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/public/tensor_shape.h"

#include <gtest/gtest.h>

namespace tensorflow {
namespace {

TEST(TensorShapeTest, Default) {
  // The default TensorShape constructor constructs a shape of 0-dim
  // and 1-element.
  TensorShape s;
  EXPECT_EQ(s.dims(), 0);
  EXPECT_EQ(s.num_elements(), 1);
}

TEST(TensorShapeTest, set_dim) {
  TensorShape s({10, 5});

  s.set_dim(0, 20);
  ASSERT_EQ(2, s.dims());
  EXPECT_EQ(20, s.dim_size(0));
  EXPECT_EQ(100, s.num_elements());

  s.set_dim(1, 2);
  ASSERT_EQ(2, s.dims());
  EXPECT_EQ(2, s.dim_size(1));
  EXPECT_EQ(40, s.num_elements());
}

TEST(TensorShapeTest, RemoveDim) {
  TensorShape s({10, 5});
  s.RemoveDim(0);
  EXPECT_EQ(5, s.num_elements());
  ASSERT_EQ(1, s.dims());
}

TEST(TensorShapeTest, RemoveAndAddDim) {
  TensorShape s({10, 5, 20});
  s.RemoveDim(1);
  s.AddDim(100);

  EXPECT_EQ(20000, s.num_elements());
  ASSERT_EQ(3, s.dims());
}

TEST(TensorShapeTest, InvalidShapeProto) {
  TensorShapeProto proto;
  EXPECT_TRUE(TensorShape::IsValid(proto));

  proto.add_dim()->set_size(357);
  proto.add_dim()->set_size(982);
  EXPECT_TRUE(TensorShape::IsValid(proto));

  proto.Clear();
  proto.add_dim()->set_size(-357);
  proto.add_dim()->set_size(-982);
  EXPECT_FALSE(TensorShape::IsValid(proto));

  proto.Clear();
  proto.add_dim()->set_size(1LL << 20);
  proto.add_dim()->set_size((1LL << 20) + 1);
  EXPECT_FALSE(TensorShape::IsValid(proto));
}

TEST(TensorShapeTest, SetDimForEmptyTensor) {
  TensorShape s({10, 5, 20});
  EXPECT_EQ(1000, s.num_elements());
  s.set_dim(1, 0);
  EXPECT_EQ(0, s.num_elements());
  s.set_dim(1, 7);
  EXPECT_EQ(1400, s.num_elements());
}

}  // namespace
}  // namespace tensorflow
