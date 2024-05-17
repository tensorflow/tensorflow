/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/client/lib/sorting.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include "xla/client/xla_builder.h"
#include "xla/test.h"
#include "xla/tests/client_library_test_base.h"
#include "xla/tests/test_macros.h"
#include "xla/types.h"

namespace xla {
namespace {

using SortingTest = ClientLibraryTestBase;

XLA_TEST_F(SortingTest, TopK3From8Values) {
  XlaBuilder builder(TestName());
  auto x =
      ConstantR1<float>(&builder, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0});
  xla::GetTupleElement(xla::TopK(x, 3), 0);
  ComputeAndCompareR1<float>(&builder, {7.0, 6.0, 5.0}, {});
}

XLA_TEST_F(SortingTest, TopK3From8Indices) {
  XlaBuilder builder(TestName());
  auto x_rev =
      ConstantR1<float>(&builder, {7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0});
  xla::GetTupleElement(xla::TopK(x_rev, 3), 1);
  ComputeAndCompareR1<int>(&builder, {0, 1, 2}, {});
}

XLA_TEST_F(SortingTest, TopK3From8Int16Indices) {
  XlaBuilder builder(TestName());
  auto x =
      ConstantR1<float>(&builder, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0});
  xla::GetTupleElement(xla::TopK(x, 3, PrimitiveType::S16), 1);
  ComputeAndCompareR1<int16_t>(&builder, {7, 6, 5}, {});
}

XLA_TEST_F(SortingTest, TopKFullSortMinInt) {
  XlaBuilder builder(TestName());
  auto x_rev = ConstantR1<int>(&builder, {std::numeric_limits<int>::min(),
                                          std::numeric_limits<int>::min() + 1,
                                          std::numeric_limits<int>::max()});
  xla::GetTupleElement(xla::TopK(x_rev, 3), 1);
  ComputeAndCompareR1<int>(&builder, {2, 1, 0}, {});
}

XLA_TEST_F(SortingTest, TopKFullSort) {
  XlaBuilder builder(TestName());
  const int kSize = 16;
  std::mt19937 eng;
  std::uniform_real_distribution<float> u_dist(0.0, 100.0);
  auto gen = std::bind(u_dist, eng);
  std::vector<float> inputs(kSize);
  std::generate(inputs.begin(), inputs.end(), gen);
  auto x = ConstantR1<float>(&builder, inputs);
  xla::GetTupleElement(xla::TopK(x, kSize), 0);

  absl::c_sort(inputs, std::greater<float>());
  ComputeAndCompareR1<float>(&builder, inputs, {});
}

XLA_TEST_F(SortingTest, TopKFullSortWithDuplicates) {
  XlaBuilder builder(TestName());
  XlaOp a;
  auto a_data = CreateR1Parameter<int>({1, 1, 2, 2, 1}, 0, "a", &builder, &a);
  xla::GetTupleElement(xla::TopK(a, 5), 1);
  ComputeAndCompareR1<int>(&builder, {2, 3, 0, 1, 4}, {a_data.get()});
}

XLA_TEST_F(SortingTest, TopK3From8Values2Partitions) {
  XlaBuilder builder(TestName());
  auto x =
      ConstantR1<float>(&builder, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0});
  xla::GetTupleElement(xla::TopKWithPartitions(x, 3, /*num_partitions=*/2), 0);
  ComputeAndCompareR1<float>(&builder, {7.0, 6.0, 5.0}, {});
}

XLA_TEST_F(SortingTest, TopK3From8Indices2Partitions) {
  XlaBuilder builder(TestName());
  auto x_rev =
      ConstantR1<float>(&builder, {7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0});
  xla::GetTupleElement(xla::TopKWithPartitions(x_rev, 3, /*num_partitions=*/2),
                       1);
  ComputeAndCompareR1<int>(&builder, {0, 1, 2}, {});
}

XLA_TEST_F(SortingTest, TopK3From8Values3Partitions) {
  XlaBuilder builder(TestName());
  auto x =
      ConstantR1<float>(&builder, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0});
  xla::GetTupleElement(xla::TopKWithPartitions(x, 3, /*num_partitions=*/3), 0);
  ComputeAndCompareR1<float>(&builder, {7.0, 6.0, 5.0}, {});
}

XLA_TEST_F(SortingTest, TopK3From8Indices3Partitions) {
  XlaBuilder builder(TestName());
  auto x_rev =
      ConstantR1<float>(&builder, {7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0});
  xla::GetTupleElement(xla::TopKWithPartitions(x_rev, 3, /*num_partitions=*/3),
                       1);
  ComputeAndCompareR1<int>(&builder, {0, 1, 2}, {});
}

XLA_TEST_F(SortingTest, TopK3From8Values5Partitions) {
  XlaBuilder builder(TestName());
  auto x =
      ConstantR1<float>(&builder, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0});
  xla::GetTupleElement(xla::TopKWithPartitions(x, 3, /*num_partitions=*/5), 0);
  ComputeAndCompareR1<float>(&builder, {7.0, 6.0, 5.0}, {});
}

XLA_TEST_F(SortingTest, DISABLED_TopKLargeInput) {
  XlaBuilder builder(TestName());
  Array<float> input({2, 1000000});
  input.FillRandom(1.0f, 2.0f);
  auto x =
      CreateConstantFromLiteral(LiteralUtil::CreateFromArray(input), &builder);
  Array2D<float> expected_array(2, 1000);
  expected_array.Fill(2.0f);
  xla::GetTupleElement(xla::TopK(x, 1000), 0);
  ErrorSpec error_spec(10.0f, 10.0f);
  ComputeAndCompareR2<float>(&builder, expected_array, {}, error_spec);
}

XLA_TEST_F(SortingTest, TopK3From8Indices5Partitions) {
  XlaBuilder builder(TestName());
  auto x_rev =
      ConstantR1<float>(&builder, {7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0});
  xla::GetTupleElement(xla::TopKWithPartitions(x_rev, 3, /*num_partitions=*/5),
                       1);
  ComputeAndCompareR1<int>(&builder, {0, 1, 2}, {});
}

XLA_TEST_F(SortingTest, TopK3From8Int16Indices5Partitions) {
  XlaBuilder builder(TestName());
  auto x_rev =
      ConstantR1<float>(&builder, {7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0});
  xla::GetTupleElement(xla::TopKWithPartitions(x_rev, 3, /*num_partitions=*/5,
                                               PrimitiveType::S16),
                       1);
  ComputeAndCompareR1<int16_t>(&builder, {0, 1, 2}, {});
}

XLA_TEST_F(SortingTest, TopKFullSortWithDuplicates2Partitions) {
  XlaBuilder builder(TestName());
  XlaOp a;
  auto a_data = CreateR1Parameter<int>({1, 1, 2, 2, 1}, 0, "a", &builder, &a);
  xla::GetTupleElement(xla::TopKWithPartitions(a, 3, /*num_partitions=*/2), 1);
  ComputeAndCompareR1<int>(&builder, {2, 3, 0}, {a_data.get()});
}

}  // namespace
}  // namespace xla
