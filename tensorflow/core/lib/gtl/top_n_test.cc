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

// Unit test for TopN.

#include "tensorflow/core/lib/gtl/top_n.h"

#include <string>
#include <vector>

#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/random/simple_philox.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"

namespace {

using tensorflow::gtl::TopN;
using tensorflow::random::PhiloxRandom;
using tensorflow::random::SimplePhilox;
using tensorflow::string;

// Move the contents from an owned raw pointer, returning by value.
// Objects are easier to manage by value.
template <class T>
T ConsumeRawPtr(T *p) {
  T tmp = std::move(*p);
  delete p;
  return tmp;
}

template <class Cmp>
void TestIntTopNHelper(size_t limit, size_t n_elements, const Cmp &cmp,
                       SimplePhilox *random, bool test_peek,
                       bool test_extract_unsorted) {
  LOG(INFO) << "Testing limit=" << limit << ", n_elements=" << n_elements
            << ", test_peek=" << test_peek
            << ", test_extract_unsorted=" << test_extract_unsorted;
  TopN<int, Cmp> top(limit, cmp);
  std::vector<int> shadow(n_elements);
  for (int i = 0; i != n_elements; ++i) shadow[i] = random->Uniform(limit);
  for (int e : shadow) top.push(e);
  std::sort(shadow.begin(), shadow.end(), cmp);
  size_t top_size = std::min(limit, n_elements);
  EXPECT_EQ(top_size, top.size());
  if (test_peek && top_size != 0) {
    EXPECT_EQ(shadow[top_size - 1], top.peek_bottom());
  }
  std::vector<int> v;
  if (test_extract_unsorted) {
    v = ConsumeRawPtr(top.ExtractUnsorted());
    std::sort(v.begin(), v.end(), cmp);
  } else {
    v = ConsumeRawPtr(top.Extract());
  }
  EXPECT_EQ(top_size, v.size());
  for (int i = 0; i != top_size; ++i) {
    VLOG(1) << "Top element " << v[i];
    EXPECT_EQ(shadow[i], v[i]);
  }
}

template <class Cmp>
void TestIntTopN(size_t limit, size_t n_elements, const Cmp &cmp,
                 SimplePhilox *random) {
  // Test peek_bottom() and Extract()
  TestIntTopNHelper(limit, n_elements, cmp, random, true, false);
  // Test Extract()
  TestIntTopNHelper(limit, n_elements, cmp, random, false, false);
  // Test peek_bottom() and ExtractUnsorted()
  TestIntTopNHelper(limit, n_elements, cmp, random, true, true);
  // Test ExtractUnsorted()
  TestIntTopNHelper(limit, n_elements, cmp, random, false, true);
}

TEST(TopNTest, Misc) {
  PhiloxRandom philox(1, 1);
  SimplePhilox random(&philox);

  TestIntTopN(0, 5, std::greater<int>(), &random);
  TestIntTopN(32, 0, std::greater<int>(), &random);
  TestIntTopN(6, 6, std::greater<int>(), &random);
  TestIntTopN(6, 6, std::less<int>(), &random);
  TestIntTopN(1000, 999, std::greater<int>(), &random);
  TestIntTopN(1000, 1000, std::greater<int>(), &random);
  TestIntTopN(1000, 1001, std::greater<int>(), &random);
  TestIntTopN(2300, 28393, std::less<int>(), &random);
  TestIntTopN(30, 100, std::greater<int>(), &random);
  TestIntTopN(100, 30, std::less<int>(), &random);
  TestIntTopN(size_t(-1), 3, std::greater<int>(), &random);
  TestIntTopN(size_t(-1), 0, std::greater<int>(), &random);
  TestIntTopN(0, 5, std::greater<int>(), &random);
}

TEST(TopNTest, String) {
  LOG(INFO) << "Testing strings";

  TopN<string> top(3);
  EXPECT_TRUE(top.empty());
  top.push("abracadabra");
  top.push("waldemar");
  EXPECT_EQ(2, top.size());
  EXPECT_EQ("abracadabra", top.peek_bottom());
  top.push("");
  EXPECT_EQ(3, top.size());
  EXPECT_EQ("", top.peek_bottom());
  top.push("top");
  EXPECT_EQ(3, top.size());
  EXPECT_EQ("abracadabra", top.peek_bottom());
  top.push("Google");
  top.push("test");
  EXPECT_EQ(3, top.size());
  EXPECT_EQ("test", top.peek_bottom());
  TopN<string> top2(top);
  TopN<string> top3(5);
  top3 = top;
  EXPECT_EQ("test", top3.peek_bottom());
  {
    std::vector<string> s = ConsumeRawPtr(top.Extract());
    EXPECT_EQ(s[0], "waldemar");
    EXPECT_EQ(s[1], "top");
    EXPECT_EQ(s[2], "test");
  }

  top2.push("zero");
  EXPECT_EQ(top2.peek_bottom(), "top");

  {
    std::vector<string> s = ConsumeRawPtr(top2.Extract());
    EXPECT_EQ(s[0], "zero");
    EXPECT_EQ(s[1], "waldemar");
    EXPECT_EQ(s[2], "top");
  }
  {
    std::vector<string> s = ConsumeRawPtr(top3.Extract());
    EXPECT_EQ(s[0], "waldemar");
    EXPECT_EQ(s[1], "top");
    EXPECT_EQ(s[2], "test");
  }

  TopN<string> top4(3);
  // Run this test twice to check Reset():
  for (int i = 0; i < 2; ++i) {
    top4.push("abcd");
    top4.push("ijkl");
    top4.push("efgh");
    top4.push("mnop");
    std::vector<string> s = ConsumeRawPtr(top4.Extract());
    EXPECT_EQ(s[0], "mnop");
    EXPECT_EQ(s[1], "ijkl");
    EXPECT_EQ(s[2], "efgh");
    top4.Reset();
  }
}

// Test that pointers aren't leaked from a TopN if we use the 2-argument version
// of push().
TEST(TopNTest, Ptr) {
  LOG(INFO) << "Testing 2-argument push()";
  TopN<string *> topn(3);
  for (int i = 0; i < 8; ++i) {
    string *dropped = NULL;
    topn.push(new string(std::to_string(i)), &dropped);
    delete dropped;
  }

  for (int i = 8; i > 0; --i) {
    string *dropped = NULL;
    topn.push(new string(std::to_string(i)), &dropped);
    delete dropped;
  }

  std::vector<string *> extract = ConsumeRawPtr(topn.Extract());
  tensorflow::gtl::STLDeleteElements(&extract);
}

struct PointeeGreater {
  template <typename T>
  bool operator()(const T &a, const T &b) const {
    return *a > *b;
  }
};

TEST(TopNTest, MoveOnly) {
  using StrPtr = std::unique_ptr<string>;
  TopN<StrPtr, PointeeGreater> topn(3);
  for (int i = 0; i < 8; ++i) topn.push(StrPtr(new string(std::to_string(i))));
  for (int i = 8; i > 0; --i) topn.push(StrPtr(new string(std::to_string(i))));

  std::vector<StrPtr> extract = ConsumeRawPtr(topn.Extract());
  EXPECT_EQ(extract.size(), 3);
  EXPECT_EQ(*(extract[0]), "8");
  EXPECT_EQ(*(extract[1]), "7");
  EXPECT_EQ(*(extract[2]), "7");
}

// Test that Nondestructive extracts do not need a Reset() afterwards,
// and that pointers aren't leaked from a TopN after calling them.
TEST(TopNTest, Nondestructive) {
  LOG(INFO) << "Testing Nondestructive extracts";
  TopN<int> top4(4);
  for (int i = 0; i < 8; ++i) {
    top4.push(i);
    std::vector<int> v = ConsumeRawPtr(top4.ExtractNondestructive());
    EXPECT_EQ(std::min(i + 1, 4), v.size());
    for (size_t j = 0; j < v.size(); ++j) EXPECT_EQ(i - j, v[j]);
  }

  TopN<int> top3(3);
  for (int i = 0; i < 8; ++i) {
    top3.push(i);
    std::vector<int> v = ConsumeRawPtr(top3.ExtractUnsortedNondestructive());
    std::sort(v.begin(), v.end(), std::greater<int>());
    EXPECT_EQ(std::min(i + 1, 3), v.size());
    for (size_t j = 0; j < v.size(); ++j) EXPECT_EQ(i - j, v[j]);
  }
}

struct ForbiddenCmp {
  bool operator()(int lhs, int rhs) const {
    LOG(FATAL) << "ForbiddenCmp called " << lhs << " " << rhs;
  }
};

TEST(TopNTest, ZeroLimit) {
  TopN<int, ForbiddenCmp> top(0);
  top.push(1);
  top.push(2);

  int dropped = -1;
  top.push(1, &dropped);
  top.push(2, &dropped);

  std::vector<int> v;
  top.ExtractNondestructive(&v);
  EXPECT_EQ(0, v.size());
}

TEST(TopNTest, Iteration) {
  TopN<int> top(4);
  for (int i = 0; i < 8; ++i) top.push(i);
  std::vector<int> actual(top.unsorted_begin(), top.unsorted_end());
  // Check that we have 4,5,6,7 as the top 4 (in some order, so we sort)
  sort(actual.begin(), actual.end());
  EXPECT_EQ(actual.size(), 4);
  EXPECT_EQ(actual[0], 4);
  EXPECT_EQ(actual[1], 5);
  EXPECT_EQ(actual[2], 6);
  EXPECT_EQ(actual[3], 7);
}
}  // namespace
