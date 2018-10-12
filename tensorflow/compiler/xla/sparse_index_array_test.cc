/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/sparse_index_array.h"

#include <vector>

#include "tensorflow/compiler/xla/test.h"

namespace xla {
namespace {

TEST(SparseIndexArrayTest, Sort) {
  SparseIndexArray a(10, 3);
  a.Append({2, 3, 4});
  a.Append({3, 4, 5});
  a.Append({1, 2, 3});
  a.Append({5, 6, 7});
  a.Append({4, 5, 6});
  a.Append({6, 7, 8});
  std::vector<double> values = {
      12.0, 13.0, 11.0, 15.0, 14.0, 16.0,
  };
  a.SortWithValues<double>(absl::MakeSpan(values));
  ASSERT_EQ(a.data(), std::vector<int64>({1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5,
                                          6, 7, 6, 7, 8}));
  ASSERT_EQ(values, std::vector<double>({11.0, 12.0, 13.0, 14.0, 15.0, 16.0}));
}

}  // namespace
}  // namespace xla
