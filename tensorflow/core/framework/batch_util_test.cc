/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(CopyContiguousSlicesTest, CompatibleShape) {
  Tensor src(DT_FLOAT, {7, 1, 2});
  Tensor dst(DT_FLOAT, {9, 2, 1});
  auto s = batch_util::CopyContiguousSlices(
      src, /*src_offset=*/2, /*dst_offset=*/0, /*num_slices=*/5, &dst);
  ASSERT_EQ(error::OK, s.code());
}

TEST(CopyContiguousSlicesTest, SourceOffsetOutOfRange) {
  Tensor src(DT_FLOAT, {7, 1, 2});
  Tensor dst(DT_FLOAT, {9, 2, 1});
  auto s = batch_util::CopyContiguousSlices(
      src, /*src_offset=*/7, /*dst_offset=*/0, /*num_slices=*/5, &dst);
  ASSERT_EQ(error::FAILED_PRECONDITION, s.code());
}

TEST(CopyContiguousSlicesTest, DstOffsetOutOfRange) {
  Tensor src(DT_FLOAT, {7, 1, 2});
  Tensor dst(DT_FLOAT, {9, 2, 1});
  auto s = batch_util::CopyContiguousSlices(
      src, /*src_offset=*/0, /*dst_offset=*/0, /*num_slices=*/8, &dst);
  ASSERT_EQ(error::FAILED_PRECONDITION, s.code());
}

TEST(CopyContiguousSlicesTest, CheckDstWithExpectedValues) {
  auto src = test::AsTensor<float>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                   TensorShape({5, 2}));
  Tensor dst(DT_FLOAT, {9, 2, 1});
  auto s = batch_util::CopyContiguousSlices(
      src, /*src_offset=*/1, /*dst_offset=*/5, /*num_slices=*/3, &dst);
  ASSERT_EQ(error::OK, s.code());
  test::ExpectTensorEqual<float>(
      test::AsTensor<float>({2, 3, 4, 5, 6, 7}, TensorShape({3, 2, 1})),
      dst.Slice(5, 8));
}

}  // namespace
}  // namespace tensorflow
