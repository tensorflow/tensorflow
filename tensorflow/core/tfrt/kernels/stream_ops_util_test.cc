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
#include "tensorflow/core/tfrt/kernels/stream_ops_util.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_matcher.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/tfrt/kernels/stream_ops_util_constants.h"

namespace tensorflow {
namespace tfrt_stub {
namespace {

using ::tensorflow::test::AsScalar;
using ::tensorflow::test::AsTensor;
using ::tensorflow::test::TensorEq;
using ::testing::ElementsAre;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;
using ::testing::status::IsOkAndHolds;

int64_t RequestId(int64_t step_id, uint32_t id) {
  return (step_id << kStepIdBitSize) | id;
}

TEST(UnbatchStreamResultsTest, ScalarStepId) {
  const tensorflow::Tensor step_ids = AsScalar<int64_t>(1);
  const std::vector<tensorflow::Tensor> tensors = {
      AsScalar<int32_t>(1),
      AsTensor<int32_t>({2, 3}),
  };
  EXPECT_THAT(UnbatchStreamResults(step_ids, tensors),
              IsOkAndHolds(UnorderedElementsAre(
                  Pair(1, ElementsAre(TensorEq(AsScalar<int32_t>(1)),
                                      TensorEq(AsTensor<int32_t>({2, 3})))))));
}

TEST(UnbatchStreamResultsTest, Batched) {
  const tensorflow::Tensor step_ids = AsTensor<int64_t>(
      {RequestId(1, 0), RequestId(1, 1), RequestId(2, 0), RequestId(3, 0)});
  const std::vector<tensorflow::Tensor> tensors = {
      AsTensor<int32_t>({1, 2, 3, 4}),
      AsTensor<int32_t>({5, 6, 7, 8}),
  };
  EXPECT_THAT(UnbatchStreamResults(step_ids, tensors),
              IsOkAndHolds(UnorderedElementsAre(
                  Pair(1, ElementsAre(TensorEq(AsTensor<int32_t>({1, 2})),
                                      TensorEq(AsTensor<int32_t>({5, 6})))),
                  Pair(2, ElementsAre(TensorEq(AsTensor<int32_t>({3})),
                                      TensorEq(AsTensor<int32_t>({7})))),
                  Pair(3, ElementsAre(TensorEq(AsTensor<int32_t>({4})),
                                      TensorEq(AsTensor<int32_t>({8})))))));
}

TEST(UnbatchStreamResultsTest, BatchedUnordered) {
  const tensorflow::Tensor step_ids = AsTensor<int64_t>(
      {RequestId(2, 0), RequestId(1, 0), RequestId(1, 1), RequestId(3, 0)});
  const std::vector<tensorflow::Tensor> tensors = {
      AsTensor<int32_t>({20, 10, 10, 30}),
  };
  EXPECT_THAT(UnbatchStreamResults(step_ids, tensors),
              IsOkAndHolds(UnorderedElementsAre(
                  Pair(1, ElementsAre(TensorEq(AsTensor<int32_t>({10, 10})))),
                  Pair(2, ElementsAre(TensorEq(AsTensor<int32_t>({20})))),
                  Pair(3, ElementsAre(TensorEq(AsTensor<int32_t>({30})))))));
}

TEST(UnbatchStreamResultsTest, PaddingOneExample) {
  const tensorflow::Tensor step_ids = AsTensor<int64_t>(
      {RequestId(1, 0), RequestId(1, 0), RequestId(1, 0), RequestId(1, 0)});
  const std::vector<tensorflow::Tensor> tensors = {
      AsTensor<int32_t>({10, 10, 10, 10}),
  };
  EXPECT_THAT(UnbatchStreamResults(step_ids, tensors),
              IsOkAndHolds(UnorderedElementsAre(
                  Pair(1, ElementsAre(TensorEq(AsTensor<int32_t>({10})))))));
}

TEST(UnbatchStreamResultsTest, PaddingMultipleExamples) {
  const tensorflow::Tensor step_ids = AsTensor<int64_t>(
      {RequestId(1, 0), RequestId(1, 1), RequestId(2, 0), RequestId(1, 0)});
  const std::vector<tensorflow::Tensor> tensors = {
      AsTensor<int32_t>({10, 20, 30, 10}),
  };
  EXPECT_THAT(UnbatchStreamResults(step_ids, tensors),
              IsOkAndHolds(UnorderedElementsAre(
                  Pair(1, ElementsAre(TensorEq(AsTensor<int32_t>({10, 20})))),
                  Pair(2, ElementsAre(TensorEq(AsTensor<int32_t>({30})))))));
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace tensorflow
