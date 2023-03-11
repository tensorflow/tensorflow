/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/experimental/remat/rematerializer.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
namespace TFL {

namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::FieldsAre;

class RematTest : public ::testing::Test {
 protected:
  class TestableRematerializer : public Rematerializer {
   public:
    using Rematerializer::AddOperation;
    using Rematerializer::AddTensor;
    using Rematerializer::AddUse;
    using Rematerializer::DelUse;
    using Rematerializer::Remat;
  };
  TestableRematerializer r_;
};

TEST_F(RematTest, TensorUseSimple) {
  for (int i = 0; i < 6; ++i) {
    r_.AddOperation();
    r_.AddTensor(/*size=*/1 << i);
  }

  r_.AddUse(/*ioperation=*/2, /*itensor=*/2);
  EXPECT_THAT(r_.GetMemProfile(), ElementsAre(0, 0, 4, 0, 0, 0));
  EXPECT_THAT(r_.GetPeakMemory(), FieldsAre(Eq(2), Eq(4)));

  r_.AddUse(/*ioperation=*/2, /*itensor=*/2);
  EXPECT_THAT(r_.GetMemProfile(), ElementsAre(0, 0, 4, 0, 0, 0));
  EXPECT_THAT(r_.GetPeakMemory(), FieldsAre(Eq(2), Eq(4)));

  r_.AddUse(/*ioperation=*/4, /*itensor=*/2);
  EXPECT_THAT(r_.GetMemProfile(), ElementsAre(0, 0, 4, 4, 4, 0));
  EXPECT_THAT(r_.GetPeakMemory(), FieldsAre(Eq(4), Eq(4)));

  r_.DelUse(/*ioperation=*/2, /*itensor=*/2);
  EXPECT_THAT(r_.GetMemProfile(), ElementsAre(0, 0, 0, 0, 4, 0));
  EXPECT_THAT(r_.GetPeakMemory(), FieldsAre(Eq(4), Eq(4)));

  r_.DelUse(/*ioperation=*/2, /*itensor=*/2);
  EXPECT_THAT(r_.GetMemProfile(), ElementsAre(0, 0, 0, 0, 4, 0));
  EXPECT_THAT(r_.GetPeakMemory(), FieldsAre(Eq(4), Eq(4)));

  r_.DelUse(/*ioperation=*/4, /*itensor=*/2);
  EXPECT_THAT(r_.GetMemProfile(), ElementsAre(0, 0, 0, 0, 0, 0));
  EXPECT_THAT(r_.GetPeakMemory(), FieldsAre(Eq(5), Eq(0)));
}

TEST_F(RematTest, TensorUseMany) {
  constexpr int n = 6;
  for (int i = 0; i < n; ++i) {
    r_.AddUse(/*ioperation=*/r_.AddOperation(),
              /*itensor=*/r_.AddTensor(1 << (n - i - 1)));
  }
  for (int i = 0; i < n; ++i) {
    r_.AddUse(/*ioperation=*/r_.AddOperation(),
              /*itensor=*/n - 1 - i);
  }

  EXPECT_THAT(r_.GetMemProfile(), ElementsAreArray({32, 48, 56, 60, 62, 63, 63,
                                                    62, 60, 56, 48, 32}));
  EXPECT_THAT(r_.GetPeakMemory(), FieldsAre(Eq(6), Eq(63)));
}

TEST_F(RematTest, PeakTiesAreBrokenInFavorOfLaterOperations) {
  r_.AddUse(/*ioperation=*/r_.AddOperation(),
            /*itensor=*/r_.AddTensor(100));
  r_.AddUse(/*ioperation=*/r_.AddOperation(),
            /*itensor=*/r_.AddTensor(1));
  r_.AddUse(/*ioperation=*/r_.AddOperation(),
            /*itensor=*/r_.AddTensor(100));
  ASSERT_THAT(r_.GetMemProfile(), ElementsAreArray({100, 1, 100}));
  EXPECT_THAT(r_.GetPeakMemory(), FieldsAre(Eq(2), Eq(100)));
}

TEST_F(RematTest, RematRecreatesOutput) {
  r_.AddUse(r_.AddOperation(), r_.AddTensor(100));
  r_.AddOperation();
  // /* before: */
  // %0 = f1()
  // f2()
  ASSERT_THAT(r_.GetMemProfile(), ElementsAre(100, 0));
  r_.Remat({/*begin=*/0, /*end=*/1, /*insert=*/2});
  // /* after: */
  // %0 = f1()
  // f2()
  // %1 = f1()
  EXPECT_THAT(r_.GetMemProfile(), ElementsAre(100, 0, 100));
  EXPECT_THAT(r_.AddTensor(0), 2);
}

TEST_F(RematTest, RematExtendsInputAndRecreatesOutput) {
  r_.AddUse(/*ioperation=*/r_.AddOperation(), /*itensor=*/r_.AddTensor(1));
  r_.AddUse(/*ioperation=*/r_.AddOperation(), /*itensor=*/r_.AddTensor(100));
  r_.AddUse(/*ioperation=*/1, 0);
  r_.AddOperation();
  r_.AddOperation();

  // /* before: */
  // %0 = f1()
  // %1 = f2(%0)
  // f3()
  // f4()

  ASSERT_THAT(r_.GetMemProfile(), ElementsAre(1, 101, 0, 0));
  r_.Remat({/*begin=*/1, /*end=*/2, /*insert=*/3});

  // /* after: */
  // %0 = f1()
  // %1 = f2(%0)
  // f3()  /* %0 is kept alive */
  // %2 = f2(%0)
  // f4()
  EXPECT_THAT(r_.GetMemProfile(), ElementsAre(1, 101, 1, 101, 0));
  EXPECT_THAT(r_.AddTensor(0), 3);
}

TEST_F(RematTest, BlockRematDuplicatesIntraBlockValues) {
  r_.AddUse(/*ioperation=*/r_.AddOperation(), /*itensor=*/r_.AddTensor(1));
  r_.AddUse(/*ioperation=*/r_.AddOperation(), /*itensor=*/r_.AddTensor(10));
  r_.AddUse(/*ioperation=*/r_.AddOperation(), /*itensor=*/r_.AddTensor(100));
  r_.AddUse(/*ioperation=*/r_.AddOperation(), /*itensor=*/r_.AddTensor(1000));
  r_.AddOperation();
  r_.AddUse(/*ioperation=*/1, /*itensor=*/0);
  r_.AddUse(/*ioperation=*/2, /*itensor=*/0);
  r_.AddUse(/*ioperation=*/2, /*itensor=*/1);
  r_.AddUse(/*ioperation=*/3, /*itensor=*/0);
  r_.AddUse(/*ioperation=*/3, /*itensor=*/1);
  r_.AddUse(/*ioperation=*/3, /*itensor=*/2);

  // /* before */
  // %0 = f1()
  // %1 = f2(%0)
  // %2 = f3(%0,%1)
  // %3 = f4(%0,%1,%2)
  // f5()
  ASSERT_THAT(r_.GetMemProfile(), ElementsAre(1, 11, 111, 1111, 0));

  r_.Remat({/*begin=*/1, /*end=*/4, /*insert=*/5});

  EXPECT_THAT(r_.GetMemProfile(),
              ElementsAre(1, 11, 111, 1111, 1, 11, 111, 1111));
  EXPECT_THAT(r_.AddTensor(0), 7);
  // /* after */
  // %0 = f1()
  // %1 = f2(%0)
  // %2 = f3(%0,%1)
  // %3 = f4(%0,%1,%2)
  // f5()
  // %4 = f2(%0)
  // %5 = f3(%0,%4)
  // %6 = f4(%0,%4,%5)
}

}  // namespace

}  // namespace TFL
}  // namespace mlir
