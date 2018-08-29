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

#include "tensorflow/contrib/lite/tools/accuracy/ilsvrc/imagenet_topk_eval.h"
#include <gtest/gtest.h>

namespace tensorflow {
namespace metrics {
namespace {

const int kNumCategories = 1001;

Tensor CreateStringTensor(const string& value) {
  Tensor tensor(DT_STRING, TensorShape({}));
  tensor.scalar<string>()() = value;
  return tensor;
}

Tensor CreateOutputTensor() {
  Tensor tensor(DT_FLOAT, TensorShape({1, kNumCategories}));
  for (int i = 0; i < kNumCategories; i++) {
    tensor.flat<float>()(i) = 0;
  }
  return tensor;
}

std::vector<string> CreateGroundTruth() {
  std::vector<string> ground_truth;
  ground_truth.reserve(kNumCategories);
  for (int i = 0; i < kNumCategories; i++) {
    string category;
    strings::StrAppend(&category, i);
    ground_truth.push_back(category);
  }
  return ground_truth;
}

TEST(ImagenetTopKAccuracy, AllCorrect) {
  ImagenetTopKAccuracy acc_top_5(CreateGroundTruth(), 5);
  auto accuracies = acc_top_5.GetTopKAccuracySoFar();
  EXPECT_EQ(0, accuracies.number_of_images);
  EXPECT_EQ(5, accuracies.topk_counts.size());

  for (int i : accuracies.topk_counts) {
    EXPECT_EQ(0, i);
  }
  // First image was correctly identified as "0".
  Tensor tensor = CreateOutputTensor();
  tensor.flat<float>()(0) = 0.8;

  TF_CHECK_OK(acc_top_5.ComputeEval({tensor}, CreateStringTensor("0")));
  accuracies = acc_top_5.GetTopKAccuracySoFar();
  EXPECT_EQ(1, accuracies.number_of_images);

  for (int i : accuracies.topk_counts) {
    EXPECT_EQ(1, i);
  }
  tensor.flat<float>()(1) = 0.9;
  TF_CHECK_OK(acc_top_5.ComputeEval({tensor}, CreateStringTensor("1")));
  accuracies = acc_top_5.GetTopKAccuracySoFar();
  EXPECT_EQ(2, accuracies.number_of_images);

  for (int i : accuracies.topk_counts) {
    EXPECT_EQ(2, i);
  }
}

TEST(ImagenetTopKAccuracy, Top5) {
  ImagenetTopKAccuracy acc_top_5(CreateGroundTruth(), 5);
  auto accuracies = acc_top_5.GetTopKAccuracySoFar();
  EXPECT_EQ(0, accuracies.number_of_images);
  EXPECT_EQ(5, accuracies.topk_counts.size());

  // For first image, with ground truth "0" probabilities were
  // 0.5 for "0",
  // "0.6" for 1,
  // "0.7" for 2,
  // "0.8" for 3,
  // "0.9" for 4.
  // remaining all zeroes.

  // First image was correctly identified as "0".
  Tensor tensor = CreateOutputTensor();
  tensor.flat<float>()(0) = 0.5;
  tensor.flat<float>()(1) = 0.6;
  tensor.flat<float>()(2) = 0.7;
  tensor.flat<float>()(3) = 0.8;
  tensor.flat<float>()(4) = 0.9;

  TF_CHECK_OK(acc_top_5.ComputeEval({tensor}, CreateStringTensor("0")));
  accuracies = acc_top_5.GetTopKAccuracySoFar();
  EXPECT_EQ(1, accuracies.number_of_images);
  EXPECT_EQ(1, accuracies.topk_counts[4]);

  for (int i = 0; i < 4; i++) {
    EXPECT_EQ(0, accuracies.topk_counts[i]);
  }

  // Now for "1" only last two buckets are going to be affected.
  TF_CHECK_OK(acc_top_5.ComputeEval({tensor}, CreateStringTensor("1")));
  accuracies = acc_top_5.GetTopKAccuracySoFar();
  EXPECT_EQ(2, accuracies.number_of_images);
  EXPECT_EQ(1, accuracies.topk_counts[3]);
  EXPECT_EQ(2, accuracies.topk_counts[4]);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(0, accuracies.topk_counts[i]);
  }

  // All buckets will be affected.
  TF_CHECK_OK(acc_top_5.ComputeEval({tensor}, CreateStringTensor("4")));
  accuracies = acc_top_5.GetTopKAccuracySoFar();
  EXPECT_EQ(3, accuracies.number_of_images);
  EXPECT_EQ(1, accuracies.topk_counts[0]);
  EXPECT_EQ(1, accuracies.topk_counts[1]);
  EXPECT_EQ(1, accuracies.topk_counts[2]);
  EXPECT_EQ(2, accuracies.topk_counts[3]);
  EXPECT_EQ(3, accuracies.topk_counts[4]);

  // No buckets will be affected
  TF_CHECK_OK(acc_top_5.ComputeEval({tensor}, CreateStringTensor("10")));
  accuracies = acc_top_5.GetTopKAccuracySoFar();
  EXPECT_EQ(4, accuracies.number_of_images);
  EXPECT_EQ(1, accuracies.topk_counts[0]);
  EXPECT_EQ(1, accuracies.topk_counts[1]);
  EXPECT_EQ(1, accuracies.topk_counts[2]);
  EXPECT_EQ(2, accuracies.topk_counts[3]);
  EXPECT_EQ(3, accuracies.topk_counts[4]);
}

}  // namespace

}  // namespace metrics
}  // namespace tensorflow

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
