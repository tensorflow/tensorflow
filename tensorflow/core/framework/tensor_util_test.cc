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

#include "tensorflow/core/framework/tensor_util.h"

#include <vector>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(TensorUtil, DeepCopy0d) {
  Tensor x(DT_FLOAT, TensorShape({}));
  x.scalar<float>()() = 10.0;

  // Make y a deep copy of x and then change it.
  Tensor y = tensor::DeepCopy(x);
  y.scalar<float>()() = 20.0;

  // x doesn't change
  EXPECT_EQ(10.0, x.scalar<float>()());

  // Change x.
  x.scalar<float>()() = 30.0;

  // Y doesn't change.
  EXPECT_EQ(20.0, y.scalar<float>()());

  Tensor z = tensor::DeepCopy(y);

  // Change y.
  y.scalar<float>()() = 40.0;

  // The final states should all be different.
  EXPECT_EQ(20.0, z.scalar<float>()());
  EXPECT_EQ(30.0, x.scalar<float>()());
  EXPECT_EQ(40.0, y.scalar<float>()());

  // Should have the same shape and type.
  EXPECT_EQ(TensorShape({}), x.shape());
  EXPECT_EQ(TensorShape({}), y.shape());
  EXPECT_EQ(TensorShape({}), z.shape());

  EXPECT_EQ(DT_FLOAT, x.dtype());
  EXPECT_EQ(DT_FLOAT, y.dtype());
  EXPECT_EQ(DT_FLOAT, z.dtype());
}

TEST(TensorUtil, DeepCopyZeroElements) {
  Tensor x;
  Tensor y = tensor::DeepCopy(x);
  EXPECT_EQ(TensorShape({0}), y.shape());
  EXPECT_EQ(DT_FLOAT, y.dtype());
  EXPECT_EQ(0, y.NumElements());
}

TEST(TensorUtil, DeepCopy) {
  Tensor x(DT_FLOAT, TensorShape({1}));
  x.flat<float>()(0) = 10.0;

  // Make y a deep copy of x and then change it.
  Tensor y = tensor::DeepCopy(x);
  y.flat<float>()(0) = 20.0;

  // x doesn't change
  EXPECT_EQ(10.0, x.flat<float>()(0));

  // Change x.
  x.flat<float>()(0) = 30.0;

  // Y doesn't change.
  EXPECT_EQ(20.0, y.flat<float>()(0));

  Tensor z = tensor::DeepCopy(y);

  // Change y.
  y.flat<float>()(0) = 40.0;

  // The final states should all be different.
  EXPECT_EQ(20.0, z.flat<float>()(0));
  EXPECT_EQ(30.0, x.flat<float>()(0));
  EXPECT_EQ(40.0, y.flat<float>()(0));

  // Should have the same shape and type.
  EXPECT_EQ(TensorShape({1}), x.shape());
  EXPECT_EQ(TensorShape({1}), y.shape());
  EXPECT_EQ(TensorShape({1}), z.shape());

  EXPECT_EQ(DT_FLOAT, x.dtype());
  EXPECT_EQ(DT_FLOAT, y.dtype());
  EXPECT_EQ(DT_FLOAT, z.dtype());

  // Test string deep copy
  Tensor str1(DT_STRING, TensorShape({2}));
  str1.flat<string>()(0) = "foo1";
  str1.flat<string>()(1) = "foo2";
  Tensor str2 = tensor::DeepCopy(str1);
  str2.flat<string>()(0) = "bar1";
  str2.flat<string>()(1) = "bar2";
  EXPECT_NE(str2.flat<string>()(0), str1.flat<string>()(0));
}

TEST(TensorUtil, DeepCopySlice) {
  Tensor x(DT_INT32, TensorShape({10}));
  x.flat<int32>().setConstant(1);

  // Slice 'x' -- y still refers to the same buffer.
  Tensor y = x.Slice(2, 6);

  // Do a deep copy of y, which is a slice.
  Tensor z = tensor::DeepCopy(y);

  // Set x to be different.
  x.flat<int32>().setConstant(2);

  EXPECT_EQ(TensorShape({10}), x.shape());
  EXPECT_EQ(TensorShape({4}), y.shape());
  EXPECT_EQ(TensorShape({4}), z.shape());
  EXPECT_EQ(DT_INT32, x.dtype());
  EXPECT_EQ(DT_INT32, y.dtype());
  EXPECT_EQ(DT_INT32, z.dtype());

  // x and y should now all be '2', but z should be '1'.
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(2, x.flat<int32>()(i));
  }
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(2, y.unaligned_flat<int32>()(i));
    EXPECT_EQ(1, z.flat<int32>()(i));
  }
}

TEST(TensorUtil, Concat) {
  std::vector<int64> sizes = {1, 4, 5};
  std::vector<Tensor> to_concat;
  int64 total_size = 0;
  int offset = 0;
  for (size_t entry = 0; entry < sizes.size(); ++entry) {
    const int64 size = sizes[entry];
    Tensor tensor(DT_INT32, TensorShape({size, 2}));
    for (int i = offset; i < offset + size; ++i) {
      for (int j = 0; j < 2; ++j) {
        tensor.matrix<int32>()(i - offset, j) = 2 * i + j;
      }
    }
    to_concat.push_back(tensor);
    total_size += size;
    offset += size;
  }

  Tensor concated;
  TF_ASSERT_OK(tensor::Concat(to_concat, &concated));
  ASSERT_EQ(TensorShape({total_size, 2}), concated.shape());
  for (int i = 0; i < total_size; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(2 * i + j, concated.matrix<int32>()(i, j));
    }
  }
}

TEST(TensorUtil, Split) {
  Tensor to_split(DT_INT64, TensorShape({10, 2}));
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 2; ++j) {
      to_split.matrix<int64>()(i, j) = 2 * i + j;
    }
  }

  std::vector<int64> sizes = {1, 4, 5};
  std::vector<Tensor> splits;
  TF_ASSERT_OK(tensor::Split(to_split, sizes, &splits));
  ASSERT_EQ(sizes.size(), splits.size());

  int offset = 0;
  for (size_t entry = 0; entry < splits.size(); ++entry) {
    const int64 size = sizes[entry];
    const Tensor& split = splits[entry];

    ASSERT_EQ(TensorShape({size, 2}), split.shape());
    for (int i = offset; i < offset + size; ++i) {
      for (int j = 0; j < 2; ++j) {
        EXPECT_EQ(2 * i + j, split.matrix<int64>()(i - offset, j));
      }
    }

    offset += size;
  }
}

TEST(TensorUtil, ConcatSplitStrings) {
  Tensor x(DT_STRING, TensorShape({4, 3}));
  for (int i = 0; i < 4 * 3; ++i) {
    x.flat<string>()(i) = strings::StrCat("foo_", i);
  }

  std::vector<Tensor> split;
  TF_ASSERT_OK(tensor::Split(x, {2, 1, 1}, &split));
  Tensor x_round_tripped;
  TF_ASSERT_OK(tensor::Concat(split, &x_round_tripped));
  ASSERT_EQ(x.shape(), x_round_tripped.shape());
  for (int i = 0; i < 4 * 3; ++i) {
    EXPECT_EQ(x.flat<string>()(i), x_round_tripped.flat<string>()(i));
  }

  // Ensure that no memory is being shared between 'x' and 'x_round_tripped'.
  for (int i = 0; i < 4 * 3; ++i) {
    x_round_tripped.flat<string>()(i) = strings::StrCat("bar_", i);
  }
  for (int i = 0; i < 4 * 3; ++i) {
    EXPECT_NE(x.flat<string>()(i), x_round_tripped.flat<string>()(i));
  }
}

}  // namespace
}  // namespace tensorflow
