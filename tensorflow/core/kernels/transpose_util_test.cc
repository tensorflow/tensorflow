/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

class TransposeUtilTest : public ::testing::Test {
 protected:
  void TestDimensionReduction(const TensorShape& shape,
                              const gtl::ArraySlice<int32> perm,
                              const gtl::ArraySlice<int32> expected_perm,
                              const gtl::ArraySlice<int64_t> expected_dims) {
    internal::TransposePermsVec new_perm;
    internal::TransposeDimsVec new_dims;
    internal::ReduceTransposeDimensions(shape, perm, &new_perm, &new_dims);

    gtl::ArraySlice<int32> computed_perm(new_perm);
    gtl::ArraySlice<int64_t> computed_dims(new_dims);
    EXPECT_EQ(computed_perm, expected_perm);
    EXPECT_EQ(computed_dims, expected_dims);
  }
};

TEST_F(TransposeUtilTest, NormalDimensionReduction) {
  TestDimensionReduction({2, 3, 4}, {0, 2, 1}, {0, 2, 1}, {2, 3, 4});

  TestDimensionReduction({2, 3, 4}, {1, 0, 2}, {1, 0, 2}, {2, 3, 4});

  TestDimensionReduction({2, 3, 4}, {2, 1, 0}, {2, 1, 0}, {2, 3, 4});

  TestDimensionReduction({2, 3, 4, 5}, {0, 2, 3, 1}, {0, 2, 1}, {2, 3, 20});

  TestDimensionReduction({2, 3, 4, 5}, {0, 3, 1, 2}, {0, 2, 1}, {2, 12, 5});

  TestDimensionReduction({2, 3, 4, 5}, {3, 1, 2, 0}, {2, 1, 0}, {2, 12, 5});

  TestDimensionReduction({2, 3, 4, 5}, {2, 3, 1, 0}, {2, 1, 0}, {2, 3, 20});

  TestDimensionReduction({2, 3, 4, 5, 6}, {0, 2, 3, 4, 1}, {0, 2, 1},
                         {2, 3, 120});

  TestDimensionReduction({2, 3, 4, 5, 6}, {0, 4, 1, 2, 3}, {0, 2, 1},
                         {2, 60, 6});

  TestDimensionReduction({2, 3, 4, 5, 6}, {4, 1, 2, 3, 0}, {2, 1, 0},
                         {2, 60, 6});

  TestDimensionReduction({2, 3, 4, 5, 6}, {3, 4, 1, 2, 0}, {2, 1, 0},
                         {2, 12, 30});

  TestDimensionReduction({2, 3}, {1, 0}, {1, 0}, {2, 3});

  TestDimensionReduction({2, 3, 4}, {2, 0, 1}, {1, 0}, {6, 4});

  TestDimensionReduction({2, 3, 4}, {1, 2, 0}, {1, 0}, {2, 12});

  TestDimensionReduction({2, 3, 4, 5}, {2, 3, 0, 1}, {1, 0}, {6, 20});

  TestDimensionReduction({2, 3, 4, 5}, {1, 2, 3, 0}, {1, 0}, {2, 60});

  TestDimensionReduction({2, 3, 4, 5, 6}, {2, 3, 4, 0, 1}, {1, 0}, {6, 120});

  TestDimensionReduction({2, 3, 4, 5, 6}, {4, 0, 1, 2, 3}, {1, 0}, {120, 6});

  TestDimensionReduction({2, 3, 4, 5, 6}, {0, 1, 2, 3, 4}, {0}, {720});

  TestDimensionReduction({2, 3, 4, 5}, {0, 1, 2, 3}, {0}, {120});

  TestDimensionReduction({2, 3, 4}, {0, 1, 2}, {0}, {24});

  TestDimensionReduction({2, 3}, {0, 1}, {0}, {6});
}

TEST_F(TransposeUtilTest, LargeDimensionReduction) {
  TestDimensionReduction({2, 3, 4, 5, 6, 7, 8, 9, 10, 20},
                         {0, 2, 3, 4, 5, 6, 7, 8, 9, 1}, {0, 2, 1},
                         {2, 3, 12096000});
  TestDimensionReduction({2, 3, 4, 5, 6, 7, 8, 9, 10, 20},
                         {0, 1, 2, 3, 4, 5, 6, 7, 9, 8}, {0, 2, 1},
                         {362880, 10, 20});
  TestDimensionReduction({2, 3, 4, 5, 6, 7, 8, 9, 10, 20},
                         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {0}, {72576000});
}

TEST_F(TransposeUtilTest, NonSingletonDimensionAlignment) {
  // Non-singleton dims 0, 2
  EXPECT_TRUE(internal::NonSingletonDimensionsAlign({2, 1, 2}, {1, 0, 2}));
  EXPECT_TRUE(internal::NonSingletonDimensionsAlign({2, 1, 2}, {0, 2, 1}));
  EXPECT_FALSE(internal::NonSingletonDimensionsAlign({2, 1, 2}, {2, 0, 1}));
  EXPECT_FALSE(internal::NonSingletonDimensionsAlign({2, 1, 2}, {2, 1, 0}));

  // Non-singleton dims 0, 2, 4
  EXPECT_TRUE(
      internal::NonSingletonDimensionsAlign({2, 1, 2, 1, 2}, {0, 2, 4, 1, 3}));
  EXPECT_TRUE(
      internal::NonSingletonDimensionsAlign({2, 1, 2, 1, 2}, {0, 2, 1, 4, 3}));
  EXPECT_TRUE(
      internal::NonSingletonDimensionsAlign({2, 1, 2, 1, 2}, {1, 3, 0, 2, 4}));
  EXPECT_TRUE(
      internal::NonSingletonDimensionsAlign({2, 1, 2, 1, 2}, {3, 0, 1, 2, 4}));
  EXPECT_FALSE(
      internal::NonSingletonDimensionsAlign({2, 1, 2, 1, 2}, {3, 2, 0, 1, 4}));

  // Non-singleton dims 2, 4, 5
  EXPECT_TRUE(internal::NonSingletonDimensionsAlign({1, 1, 2, 1, 2, 2},
                                                    {3, 2, 1, 4, 0, 5}));
  EXPECT_TRUE(internal::NonSingletonDimensionsAlign({1, 1, 2, 1, 2, 2},
                                                    {3, 1, 0, 2, 4, 5}));
  EXPECT_TRUE(internal::NonSingletonDimensionsAlign({1, 1, 2, 1, 2, 2},
                                                    {2, 4, 5, 0, 3, 1}));
  EXPECT_FALSE(internal::NonSingletonDimensionsAlign({1, 1, 2, 1, 2, 2},
                                                     {0, 1, 5, 2, 4, 3}));
  EXPECT_FALSE(internal::NonSingletonDimensionsAlign({1, 1, 2, 1, 2, 2},
                                                     {0, 1, 2, 5, 4, 3}));
}

}  // namespace tensorflow
