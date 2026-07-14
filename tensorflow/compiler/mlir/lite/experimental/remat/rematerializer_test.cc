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
#include <cstdlib>
#include <initializer_list>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
namespace TFL {

namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::FieldsAre;
using ::testing::StrictMock;

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
    r_.AddOperation(/*is_stateful=*/false);
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
    r_.AddUse(/*ioperation=*/r_.AddOperation(/*is_stateful=*/false),
              /*itensor=*/r_.AddTensor(1 << (n - i - 1)));
  }
  for (int i = 0; i < n; ++i) {
    r_.AddUse(/*ioperation=*/r_.AddOperation(/*is_stateful=*/false),
              /*itensor=*/n - 1 - i);
  }

  EXPECT_THAT(r_.GetMemProfile(), ElementsAreArray({32, 48, 56, 60, 62, 63, 63,
                                                    62, 60, 56, 48, 32}));
  EXPECT_THAT(r_.GetPeakMemory(), FieldsAre(Eq(6), Eq(63)));
}

TEST_F(RematTest, PeakTiesAreBrokenInFavorOfLaterOperations) {
  r_.AddUse(/*ioperation=*/r_.AddOperation(/*is_stateful=*/false),
            /*itensor=*/r_.AddTensor(100));
  r_.AddUse(/*ioperation=*/r_.AddOperation(/*is_stateful=*/false),
            /*itensor=*/r_.AddTensor(1));
  r_.AddUse(/*ioperation=*/r_.AddOperation(/*is_stateful=*/false),
            /*itensor=*/r_.AddTensor(100));
  ASSERT_THAT(r_.GetMemProfile(), ElementsAreArray({100, 1, 100}));
  EXPECT_THAT(r_.GetPeakMemory(), FieldsAre(Eq(2), Eq(100)));
}

TEST_F(RematTest, RematRecreatesOutput) {
  r_.AddUse(r_.AddOperation(/*is_stateful=*/false), r_.AddTensor(100));
  r_.AddOperation(/*is_stateful=*/false);
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
  r_.AddUse(/*ioperation=*/r_.AddOperation(/*is_stateful=*/false),
            /*itensor=*/r_.AddTensor(1));
  r_.AddUse(/*ioperation=*/r_.AddOperation(/*is_stateful=*/false),
            /*itensor=*/r_.AddTensor(100));
  r_.AddUse(/*ioperation=*/1, 0);
  r_.AddOperation(/*is_stateful=*/false);
  r_.AddOperation(/*is_stateful=*/false);

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
  r_.AddUse(/*ioperation=*/r_.AddOperation(/*is_stateful=*/false),
            /*itensor=*/r_.AddTensor(1));
  r_.AddUse(/*ioperation=*/r_.AddOperation(/*is_stateful=*/false),
            /*itensor=*/r_.AddTensor(10));
  r_.AddUse(/*ioperation=*/r_.AddOperation(/*is_stateful=*/false),
            /*itensor=*/r_.AddTensor(100));
  r_.AddUse(/*ioperation=*/r_.AddOperation(/*is_stateful=*/false),
            /*itensor=*/r_.AddTensor(1000));
  r_.AddOperation(/*is_stateful=*/false);
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
        AddOperation(/*is_stateful=*/false);
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

class GreedyRematTest : public testing::Test {
 protected:
  // A Rematerializer for a graph whose memory profile is a sequence of
  // "rainbows" due to nested ops (similar to what happens in backwards pass
  // gradient calculations.)
  class RainbowRemat : public Rematerializer {
   public:
    // `sizes` is a vector of vector of sizes.  Each sizes vector of length n
    // generates a `rainbow` profile of the SSA form
    //
    // %1 = f1()
    // %2 = f2()
    // ...
    // %n = fn()
    // gn(%n)
    // ...
    // g2(%2)
    // g1(%1)
    //
    // with the sizes of intermediates %1,...%n given by the sizes entries.
    //
    // If extra_ops > 0, each assignment above becomes a sequence
    // %i.1 = fi.1()
    // %i.2 = fi.1(%i.1)
    // %i.3 = fi.1(%i.2)
    // ...
    // %i = f1(%i.m)
    // where the 'dotted' intermediates have size `extra_size`. This
    // allows for testing the effect of window sizes.
    explicit RainbowRemat(const std::vector<std::vector<int>>& sizes,
                          int extra_ops = 0, SizeT extra_size = 0) {
      for (const auto& rainbow : sizes) {
        int tensor = 0;
        int op = 0;
        for (const auto& size : rainbow) {
          for (int i = 0; i < extra_ops; ++i) {
            op = AddOperation(/*is_stateful=*/false);
            if (i != 0) {
              AddUse(op, tensor);
            }
            tensor = AddTensor(extra_size);
            AddUse(op, tensor);
          }
          // using negative sizes to signal forbidden operations.
          op = AddOperation(/*is_stateful=*/size < 0);
          if (extra_ops > 0) {
            AddUse(op, tensor);
          }
          tensor = AddTensor(std::abs(size));
          AddUse(op, tensor);
        }
        for (int i = 0; i < rainbow.size(); ++i) {
          op = AddOperation(/*is_stateful=*/false);
          AddUse(op, tensor - i);
        }
      }
    }
  };

  // Similar to above: A multilayer perceptron. The forward pass is
  // f[n](f[n-1](...(f[1]())), with dimensions |f[1]|...|f[n]| handed in the
  // constructor. The backward pass calculates the loss gradient for a single
  // weight in the first layer.
  class MlpRemat : public Rematerializer {
   public:
    explicit MlpRemat(const std::vector<int>& sizes) {
      int forward_tensor = -1;
      int backward_tensor = -1;
      int op = -1;
      // Forward pass:
      for (const int size : sizes) {
        op = AddOperation(/*is_stateful=*/false);
        if (forward_tensor >= 0) AddUse(op, forward_tensor);
        forward_tensor = AddTensor(size);
        AddUse(op, forward_tensor);
      }

      // Backward pass: Right-multiply the jacobian from the outside in.
      // dLoss/df[n] * df[n]/df[n-1] * ...
      // The i-th term g[i] depends on g[i-1] and f[n-i] and has size |f[n-i]|.
      for (; forward_tensor >= 0; --forward_tensor) {
        op = AddOperation(/*is_stateful=*/false);
        AddUse(op, forward_tensor);
        if (backward_tensor >= 0) AddUse(op, backward_tensor);
        backward_tensor = AddTensor(sizes[forward_tensor]);
        AddUse(op, backward_tensor);
      }
    }
    // We will also instrument the actual optimizations performed.
    MOCK_METHOD(void, ApplyRemat, (const RematSpec&));
  };
};

TEST_F(GreedyRematTest, MlpBasic) {
  StrictMock<MlpRemat> remat(std::vector<int>({1, 1, 1}));
  // (o)ut, (i)n, (l)ive: 0 1 2 3 4 5 Sum
  // %0 = f0()            o           1
  // %1 = f1(%0)          i o         2
  // %2 = f2(%1)          l i o       3
  // %3 = g2(   %2)       l l i o     4
  // %4 = g1(%3,%1)       l i   i o   4
  // %5 = g0(%4,%0)       i       i o 3
  ASSERT_THAT(remat.GetMemProfile(), ElementsAreArray({1, 2, 3, 4, 4, 3}));

  // Little we can do here -- remat %0 before %5
  EXPECT_CALL(remat, ApplyRemat(FieldsAre(/*begin=*/0,
                                          /*end=*/1,
                                          /*insert=*/5)));
  // (o)ut, (i)n, (l)ive: 0 1 2 3 4 5 Sum of live sizes
  // %0 = f0()            o           1
  // %1 = f1(%0)          i o         2
  // %2 = f2(%1)            i o       2
  // %3 = g2(   %2)         l i o     3
  // %4 = g1(%3,%1)         i   i o   3
  // %0' = f0()           o       l   2
  // %5 = g0(%4,%0')      i       i o 3
  remat.RunGreedyAlgorithm(/*max_cost=*/-1, /*max_block_length=*/1,
                           /*min_savings=*/1);

  EXPECT_THAT(remat.GetMemProfile(), ElementsAreArray({1, 2, 2, 3, 3, 2, 3}));
}

TEST_F(GreedyRematTest, MlpBinary) {
  StrictMock<MlpRemat> remat(std::vector<int>({1, 2, 4, 8}));
  // (o)ut, (i)n, (l)ive: 0 1 2 3 4 5 6 7 Sum of live sizes
  // %0 = f0()            o               1
  // %1 = f1(%0)          i o             3
  // %2 = f2(%1)          l i o           7
  // %3 = f3(%2)          l l i o         15
  // %4 = g3(   %3)       l l l i o       23
  // %5 = g2(%4 %2)       l l i   i o     19
  // %6 = g1(%5,%1)       l i       i o   9
  // %7 = g0(%6,%0)       i           i o 4
  ASSERT_THAT(remat.GetMemProfile(),
              ElementsAreArray({1, 3, 7, 15, 23, 19, 9, 4}));

  EXPECT_CALL(remat, ApplyRemat(FieldsAre(/*begin=*/2,
                                          /*end=*/3,
                                          /*insert=*/5)));
  EXPECT_CALL(remat, ApplyRemat(FieldsAre(/*begin=*/0,
                                          /*end=*/1,
                                          /*insert=*/8)));

  // (o)ut, (i)n, (l)ive: 0 1 2 3 4 5 6 7 Sum of live sizes
  // %0 = f0()            o               1
  // %1 = f1(%0)          i o             3
  // %2 = f2(%1)            i o           6
  // %3 = f3(%2)            l i o         14
  // %4 = g3(   %3)         l   i o       18
  // %2'= f2(%1)            l o   l       14
  // %5 = g2(%4 %2)         l i   i o     18
  // %6 = g1(%5,%1)         i       i o   8
  // %0'= f(0)            o           l   3
  // %7 = g0(%6,%0)       i           i o 4
  remat.RunGreedyAlgorithm(/*max_cost=*/-1, /*max_block_length=*/4,
                           /*min_savings=*/1);
  EXPECT_THAT(remat.GetMemProfile(),
              ElementsAreArray({1, 3, 6, 14, 18, 14, 18, 8, 3, 4}));
}

TEST_F(GreedyRematTest, SimpleMax) {
  RainbowRemat remat({{1, 2, 4, 8, 16}});
  ASSERT_THAT(remat.GetMemProfile(),
              ElementsAreArray({1, 3, 7, 15, 31, 31, 15, 7, 3, 1}));
  remat.RunGreedyAlgorithm(/*max_cost=*/-1, /*max_block_length=*/1,
                           /*min_savings=*/1);
  // Profile is flattened to its minimum--16.
  EXPECT_THAT(remat.GetMemProfile(),
              ElementsAreArray({1, 2, 4, 8, 16, 16, 8, 8, 4, 4, 2, 2, 1, 1}));
}

TEST_F(GreedyRematTest, SimpleMaxLongWindow) {
  RainbowRemat remat({{1, 2, 4, 8, 16}});
  ASSERT_THAT(remat.GetMemProfile(),
              ElementsAreArray({1, 3, 7, 15, 31, 31, 15, 7, 3, 1}));
  remat.RunGreedyAlgorithm(/*max_cost=*/-1, /*max_block_length=*/4,
                           /*min_savings=*/1);
  // Profile is flattened to its minimum--16.
  EXPECT_THAT(remat.GetMemProfile(),
              ElementsAreArray({1, 2, 4, 8, 16, 16, 8, 8, 4, 4, 2, 2, 1, 1}));
}

TEST_F(GreedyRematTest, SimpleSizeThreshold) {
  RainbowRemat remat({{1, 2, 4, 8, 16}});
  ASSERT_THAT(remat.GetMemProfile(),
              ElementsAreArray({1, 3, 7, 15, 31, 31, 15, 7, 3, 1}));
  // Only do remats of at least 4 bytes of savings -- this will lower the
  // profile by 4 + 8 instead of 1 + 2 + 4 + 8 as before.
  remat.RunGreedyAlgorithm(/*max_cost=*/-1, /*max_block_length=*/1,
                           /*min_savings=*/4);
  EXPECT_THAT(remat.GetMemProfile(),
              ElementsAreArray({1, 3, 7, 11, 19, 19, 11, 11, 7, 7, 3, 1}));
}

TEST_F(GreedyRematTest, SimpleCostThreshold) {
  RainbowRemat remat({{1, 2, 4, 8, 16}});
  ASSERT_THAT(remat.GetMemProfile(),
              ElementsAreArray({1, 3, 7, 15, 31, 31, 15, 7, 3, 1}));
  // Do at most 1 remat -- this will lower the profile by 8.
  remat.RunGreedyAlgorithm(/*max_cost=*/1, /*max_block_length=*/1,
                           /*min_savings=*/1);
  // Only as single remat is done -- this will be the best one possible,
  // lowering the profile by 8 instead of the maximum 1 + 2 + 4 + 8.
  EXPECT_THAT(remat.GetMemProfile(),
              ElementsAreArray({1, 3, 7, 15, 23, 23, 15, 15, 7, 3, 1}));
}

TEST_F(GreedyRematTest, SimpleForbiddenOps) {
  // Operator generating size-4 tensor is stateful, so it won't be materialized.
  RainbowRemat remat({{1, 2, -4, 8, 16}});
  ASSERT_THAT(remat.GetMemProfile(),
              ElementsAreArray({1, 3, 7, 15, 31, 31, 15, 7, 3, 1}));
  // Best we can do is lower the profile to 8 + 2 + 1.
  remat.RunGreedyAlgorithm(/*max_cost=*/-1, /*max_block_length=*/1,
                           /*min_savings=*/1);
  EXPECT_THAT(remat.GetMemProfile(),
              ElementsAreArray({1, 2, 4, 12, 20, 20, 12, 12, 4, 2, 2, 1, 1}));
}

TEST_F(GreedyRematTest, DoubleMax) {
  // Generate a profile with two local maxima of size 31 and 28.
  RainbowRemat remat({{1, 2, 4, 8, 16}, {4, 8, 16}});
  ASSERT_THAT(remat.GetMemProfile(),
              ElementsAreArray(
                  {1, 3, 7, 15, 31, 31, 15, 7, 3, 1, 4, 12, 28, 28, 12, 4}));
  remat.RunGreedyAlgorithm(/*max_cost=*/-1, /*max_block_length=*/1,
                           /*min_savings=*/1);
  // Two rainbows; the first is lowered by 1 + 2 + 4 + 8 == 15, the second by 4
  // + 8 == 12.
  EXPECT_THAT(remat.GetMemProfile(),
              ElementsAreArray({1, 2, 4, 8, 16, 16, 8,  8, 4, 4, 2,
                                2, 1, 1, 4, 8,  16, 16, 8, 8, 4, 4}));
}

TEST_F(GreedyRematTest, DoubleCostThreshold) {
  // Generate a profile with two local maxima of size 31 and 28.
  RainbowRemat remat({{1, 2, 4, 8, 16}, {4, 8, 16}});
  ASSERT_THAT(remat.GetMemProfile(),
              ElementsAreArray(
                  {1, 3, 7, 15, 31, 31, 15, 7, 3, 1, 4, 12, 28, 28, 12, 4}));
  remat.RunGreedyAlgorithm(/*max_cost=*/2, /*max_block_length=*/1,
                           /*min_savings=*/1);
  // Profile can be flattened twice--first, the global maximum of 31 is reduced
  // by 8; then the newly-global maximum of 28 is reduced by 8.
  EXPECT_THAT(remat.GetMemProfile(),
              ElementsAreArray({1, 3, 7, 15, 23, 23, 15, 15, 7, 3, 1, 4, 12, 20,
                                20, 12, 12, 4}));
}

TEST_F(GreedyRematTest, SingleLongerBlocksByWindowSize) {
  std::vector<Rematerializer::SizeT> best_for_window_size;
  for (int window_size : {0, 1, 2, 3, 4, 5}) {
    RainbowRemat remat({{1, 2, 4, 8}}, 2, 16);
    remat.RunGreedyAlgorithm(/*max_cost=*/-1, /*max_block_length=*/window_size,
                             /*min_savings=*/1);
    best_for_window_size.push_back(remat.GetPeakMemory().size);
  }
  EXPECT_THAT(best_for_window_size, ElementsAreArray({44, 36, 36, 32, 32, 32}));
}

}  // namespace

}  // namespace TFL
}  // namespace mlir
