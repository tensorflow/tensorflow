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

#include "tensorflow/core/util/tensor_slice_set.h"

#include <vector>
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {

namespace checkpoint {

namespace {

// A simple test: we have a 2-d tensor of shape 4 X 5 that looks like this:
//
//   0   1   2   3   4
//   5   6   7   8   9
//  10  11  12  13  14
//  15  16  17  18  19
//
// We assume this is a row-major matrix.
//
// Testing the meta version of the tensor slice set.
TEST(TensorSliceSetTest, QueryMetaTwoD) {
  TensorShape shape({4, 5});

  TensorSliceSet tss(shape, DT_INT32);
  // We store a few slices.

  // Slice #1 is the top two rows:
  //   0   1   2   3   4
  //   5   6   7   8   9
  //   .   .   .   .   .
  //   .   .   .   .   .
  TensorSlice slice_1 = TensorSlice::ParseOrDie("0,2:-");
  TF_CHECK_OK(tss.Register(slice_1, "slice_1"));

  // Slice #2 is the bottom left corner
  //   .   .   .   .   .
  //   .   .   .   .   .
  //  10  11  12   .   .
  //  15  16  17   .   .
  TensorSlice slice_2 = TensorSlice::ParseOrDie("2,2:0,3");
  TF_CHECK_OK(tss.Register(slice_2, "slice_2"));

  // Slice #3 is the bottom right corner
  //   .   .   .   .   .
  //   .   .   .   .   .
  //   .   .   .   .   .
  //   .   .   .  18  19
  TensorSlice slice_3 = TensorSlice::ParseOrDie("3,1:3,2");
  TF_CHECK_OK(tss.Register(slice_3, "slice_3"));

  // Notice that we leave a hole in the tensor
  //   .   .   .   .   .
  //   .   .   .   .   .
  //   .   .   . (13) (14)
  //   .   .   .   .   .

  // Now we query some of the slices

  // Slice #1 is an exact match
  //   0   1   2   3   4
  //   5   6   7   8   9
  //   .   .   .   .   .
  //   .   .   .   .   .
  // We just need slice_1 for this
  {
    TensorSlice s = TensorSlice::ParseOrDie("0,2:-");
    std::vector<std::pair<TensorSlice, string>> results;
    EXPECT_TRUE(tss.QueryMeta(s, &results));
    EXPECT_EQ(1, results.size());
    EXPECT_EQ("0,2:-", results[0].first.DebugString());
    EXPECT_EQ("slice_1", results[0].second);
  }

  // Slice #2 is a subset match
  //   .   .   .   .   .
  //   5   6   7   8   9
  //   .   .   .   .   .
  //   .   .   .   .   .
  // We just need slice_1 for this
  {
    TensorSlice s = TensorSlice::ParseOrDie("1,1:-");
    std::vector<std::pair<TensorSlice, string>> results;
    EXPECT_TRUE(tss.QueryMeta(s, &results));
    EXPECT_EQ(1, results.size());
    EXPECT_EQ("0,2:-", results[0].first.DebugString());
    EXPECT_EQ("slice_1", results[0].second);
  }

  // Slice #3 is a more complicated match: it needs the combination of a couple
  // of slices
  //   .   .   .   .   .
  //   5   6   7   .   .
  //  10  11  12   .   .
  //   .   .   .   .   .
  // We need both slice_1 and slice_2 for this.
  {
    TensorSlice s = TensorSlice::ParseOrDie("1,2:0,3");
    std::vector<std::pair<TensorSlice, string>> results;
    EXPECT_TRUE(tss.QueryMeta(s, &results));
    EXPECT_EQ(2, results.size());
    // Allow results to be returned in either order
    if (results[0].second == "slice_2") {
      EXPECT_EQ("2,2:0,3", results[0].first.DebugString());
      EXPECT_EQ("slice_2", results[0].second);
      EXPECT_EQ("0,2:-", results[1].first.DebugString());
      EXPECT_EQ("slice_1", results[1].second);
    } else {
      EXPECT_EQ("0,2:-", results[0].first.DebugString());
      EXPECT_EQ("slice_1", results[0].second);
      EXPECT_EQ("2,2:0,3", results[1].first.DebugString());
      EXPECT_EQ("slice_2", results[1].second);
    }
  }

  // Slice #4 includes the hole and so there is no match
  //   .   .   .   .   .
  //   .   .   7   8   9
  //   .   .  12  13  14
  //   .   .   .   .   .
  {
    TensorSlice s = TensorSlice::ParseOrDie("1,2:2,3");
    std::vector<std::pair<TensorSlice, string>> results;
    EXPECT_FALSE(tss.QueryMeta(s, &results));
    EXPECT_EQ(0, results.size());
  }
}

static void BM_RegisterOneByOne(::testing::benchmark::State& state) {
  TensorShape shape({static_cast<int>(state.max_iterations), 41});
  TensorSliceSet slice_set(shape, DT_INT32);
  int i = 0;
  for (auto s : state) {
    TensorSlice part({{i, 1}, {0, -1}});
    TF_CHECK_OK(slice_set.Register(part, part.DebugString()));
    ++i;
  }
}

BENCHMARK(BM_RegisterOneByOne);

}  // namespace

}  // namespace checkpoint

}  // namespace tensorflow
