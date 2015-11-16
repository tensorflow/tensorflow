#include "tensorflow/core/lib/gtl/iterator_range.h"

#include <vector>
#include <gtest/gtest.h>
#include "tensorflow/core/platform/port.h"

namespace tensorflow {
namespace gtl {
namespace {

TEST(IteratorRange, WholeVector) {
  std::vector<int> v = {2, 3, 5, 7, 11, 13};
  iterator_range<std::vector<int>::iterator> range(v.begin(), v.end());
  int index = 0;
  for (int prime : range) {
    ASSERT_LT(index, v.size());
    EXPECT_EQ(v[index], prime);
    ++index;
  }
  EXPECT_EQ(v.size(), index);
}

TEST(IteratorRange, VectorMakeRange) {
  std::vector<int> v = {2, 3, 5, 7, 11, 13};
  auto range = make_range(v.begin(), v.end());
  int index = 0;
  for (int prime : range) {
    ASSERT_LT(index, v.size());
    EXPECT_EQ(v[index], prime);
    ++index;
  }
  EXPECT_EQ(v.size(), index);
}

TEST(IteratorRange, PartArray) {
  int v[] = {2, 3, 5, 7, 11, 13};
  iterator_range<int*> range(&v[1], &v[4]);  // 3, 5, 7
  int index = 1;
  for (int prime : range) {
    ASSERT_LT(index, TF_ARRAYSIZE(v));
    EXPECT_EQ(v[index], prime);
    ++index;
  }
  EXPECT_EQ(4, index);
}

TEST(IteratorRange, ArrayMakeRange) {
  int v[] = {2, 3, 5, 7, 11, 13};
  auto range = make_range(&v[1], &v[4]);  // 3, 5, 7
  int index = 1;
  for (int prime : range) {
    ASSERT_LT(index, TF_ARRAYSIZE(v));
    EXPECT_EQ(v[index], prime);
    ++index;
  }
  EXPECT_EQ(4, index);
}
}  // namespace
}  // namespace gtl
}  // namespace tensorflow
