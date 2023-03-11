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

#include <algorithm>
#include <array>
#include <random>

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

  EXPECT_THAT(r_.GetMemProfile({/*begin=*/0, /*end=*/1, /*insert=*/2}),
              ElementsAre(100, 0, 100));
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

  EXPECT_THAT(r_.GetMemProfile({/*begin=*/1, /*end=*/2, /*insert=*/3}),
              ElementsAre(1, 101, 1, 101, 0));
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

  EXPECT_THAT(r_.GetMemProfile({/*begin=*/1, /*end=*/4, /*insert=*/5}),
              ElementsAre(1, 11, 111, 1111, 1, 11, 111, 1111));

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

class RematSimulationTest : public testing::Test {
 protected:
  class RandomRemat : public Rematerializer {
   public:
    using Rematerializer::Remat;  // For testing.
    RandomRemat(const int num_operations, const int num_tensors,
                const int num_uses, std::mt19937& rng) {
      std::uniform_int_distribution<int> some_size_log(0, 16);
      std::uniform_int_distribution<int> some_tensor(0, num_tensors - 1);
      std::uniform_int_distribution<int> some_operation(0, num_operations - 1);

      for (int i = 0; i < num_tensors; ++i) {
        AddTensor(SizeT{1} << some_size_log(rng));
      }
      for (int i = 0; i < num_operations; ++i) {
        AddOperation();
      }
      for (int i = 0; i < num_uses; ++i) {
        AddUse(some_operation(rng), some_tensor(rng));
      }
    }
  };
};

TEST_F(RematSimulationTest, SimulationAgreesWithReality) {
  constexpr int kNumOperations = 128;
  constexpr int kNumTensors = 32;
  constexpr int kNumUses = kNumOperations * kNumTensors / 4;

  std::mt19937 rng;
  for (int i = 0; i < 1024; ++i) {
    RandomRemat remat(kNumOperations, kNumTensors, kNumUses, rng);
    // Worst-case scenario: we might double the length of the computation
    // schedule each time we remat, so only a few iterations...
    std::array<int, 3> randos;
    const auto& [begin, end, insert] = randos;
    for (int i = 0, num_operations = kNumOperations; i < 4;
         ++i, num_operations += end - begin) {
      std::uniform_int_distribution<int> some_op(0, num_operations - 1);
      for (auto& rando : randos) {
        rando = some_op(rng);
      }
      // We need begin <= end <= insert.
      std::sort(randos.begin(), randos.end());
      const Rematerializer::RematSpec spec{begin, end, insert};
      const auto simulated_profile = remat.GetMemProfile(spec);
      remat.Remat(spec);
      const auto actual_profile = remat.GetMemProfile();
      EXPECT_THAT(simulated_profile, ElementsAreArray(actual_profile));
    }
  }
}

}  // namespace

}  // namespace TFL
}  // namespace mlir
