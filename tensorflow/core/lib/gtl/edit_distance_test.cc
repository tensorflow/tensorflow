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

#include "tensorflow/core/lib/gtl/edit_distance.h"

#include <cctype>
#include <vector>
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace gtl {
namespace {

class LevenshteinDistanceTest : public ::testing::Test {
 protected:
  std::vector<char> empty_;
  std::string s1_;
  std::string s1234_;
  std::string s567_;
  std::string kilo_;
  std::string kilogram_;
  std::string mother_;
  std::string grandmother_;
  std::string lower_;
  std::string upper_;
  std::vector<char> ebab_;
  std::vector<char> abcd_;

  void SetUp() override {
    s1_ = "1";
    s1234_ = "1234";
    s567_ = "567";
    kilo_ = "kilo";
    kilogram_ = "kilogram";
    mother_ = "mother";
    grandmother_ = "grandmother";
    lower_ = "lower case";
    upper_ = "UPPER case";
    ebab_ = {'e', 'b', 'a', 'b'};
    abcd_ = {'a', 'b', 'c', 'd'};
  }
};

TEST_F(LevenshteinDistanceTest, BothEmpty) {
  ASSERT_EQ(LevenshteinDistance(empty_, empty_, std::equal_to<char>()), 0);
}

TEST_F(LevenshteinDistanceTest, Symmetry) {
  ASSERT_EQ(LevenshteinDistance(ebab_, abcd_, std::equal_to<char>()), 3);
  ASSERT_EQ(LevenshteinDistance(abcd_, ebab_, std::equal_to<char>()), 3);
}

TEST_F(LevenshteinDistanceTest, OneEmpty) {
  ASSERT_EQ(LevenshteinDistance(s1234_, empty_, std::equal_to<char>()), 4);
  ASSERT_EQ(LevenshteinDistance(empty_, s567_, std::equal_to<char>()), 3);
}

TEST_F(LevenshteinDistanceTest, SingleElement) {
  ASSERT_EQ(LevenshteinDistance(s1234_, s1_, std::equal_to<char>()), 3);
  ASSERT_EQ(LevenshteinDistance(s1_, s1234_, std::equal_to<char>()), 3);
}

TEST_F(LevenshteinDistanceTest, Prefix) {
  ASSERT_EQ(LevenshteinDistance(kilo_, kilogram_, std::equal_to<char>()), 4);
  ASSERT_EQ(LevenshteinDistance(kilogram_, kilo_, std::equal_to<char>()), 4);
}

TEST_F(LevenshteinDistanceTest, Suffix) {
  ASSERT_EQ(LevenshteinDistance(mother_, grandmother_, std::equal_to<char>()),
            5);
  ASSERT_EQ(LevenshteinDistance(grandmother_, mother_, std::equal_to<char>()),
            5);
}

TEST_F(LevenshteinDistanceTest, DifferentComparisons) {
  ASSERT_EQ(LevenshteinDistance(lower_, upper_, std::equal_to<char>()), 5);
  ASSERT_EQ(LevenshteinDistance(upper_, lower_, std::equal_to<char>()), 5);
  ASSERT_EQ(
      LevenshteinDistance(gtl::ArraySlice<char>(lower_.data(), lower_.size()),
                          gtl::ArraySlice<char>(upper_.data(), upper_.size()),
                          std::equal_to<char>()),
      5);
  auto no_case_cmp = [](char c1, char c2) {
    return std::tolower(c1) == std::tolower(c2);
  };
  ASSERT_EQ(LevenshteinDistance(lower_, upper_, no_case_cmp), 3);
  ASSERT_EQ(LevenshteinDistance(upper_, lower_, no_case_cmp), 3);
}

TEST_F(LevenshteinDistanceTest, Vectors) {
  ASSERT_EQ(
      LevenshteinDistance(std::string("algorithm"), std::string("altruistic"),
                          std::equal_to<char>()),
      6);
}

static void BM_EditDistanceHelper(::testing::benchmark::State& state, int len,
                                  bool completely_different) {
  string a =
      "The quick brown fox jumped over the lazy dog and on and on and on"
      " Every good boy deserves fudge.  In fact, this is a very long sentence  "
      " w/many bytes..";
  while (a.size() < static_cast<size_t>(len)) {
    a = a + a;
  }
  string b = a;
  if (completely_different) {
    for (size_t i = 0; i < b.size(); i++) {
      b[i]++;
    }
  }
  for (auto s : state) {
    LevenshteinDistance(gtl::ArraySlice<char>(a.data(), len),
                        gtl::ArraySlice<char>(b.data(), len),
                        std::equal_to<char>());
  }
}

static void BM_EditDistanceSame(::testing::benchmark::State& state) {
  BM_EditDistanceHelper(state, state.range(0), false);
}
static void BM_EditDistanceDiff(::testing::benchmark::State& state) {
  BM_EditDistanceHelper(state, state.range(0), true);
}

BENCHMARK(BM_EditDistanceSame)->Arg(5);
BENCHMARK(BM_EditDistanceSame)->Arg(50);
BENCHMARK(BM_EditDistanceSame)->Arg(200);
BENCHMARK(BM_EditDistanceSame)->Arg(1000);
BENCHMARK(BM_EditDistanceDiff)->Arg(5);
BENCHMARK(BM_EditDistanceDiff)->Arg(50);
BENCHMARK(BM_EditDistanceDiff)->Arg(200);
BENCHMARK(BM_EditDistanceDiff)->Arg(1000);

}  // namespace
}  // namespace gtl
}  // namespace tensorflow
