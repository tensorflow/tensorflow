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

#define EIGEN_USE_THREADS

#include "tensorflow/core/tpu/kernels/sharding_utils.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/status.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"

namespace tensorflow {
namespace {
Eigen::ThreadPoolDevice CreateThreadPoolDevice() {
  constexpr int kMaxParallelism = 16;
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "Resharding", kMaxParallelism);

  Eigen::ThreadPoolDevice device(thread_pool->AsEigenThreadPool(),
                                 kMaxParallelism);
  return device;
}

TEST(XlaNDSplitterTest, NoSplits) {
  auto device = CreateThreadPoolDevice();

  const TensorShape input_shape({2, 2, 2});
  const std::vector<int32_t> num_splits = {1, 1, 1};
  const std::vector<int> paddings(num_splits.size(), 0);
  const int num_outputs = 1;
  auto input_tensor =
      test::AsTensor<int32_t>({0, 1, 2, 3, 4, 5, 6, 7}, input_shape);

  std::vector<Tensor> output_tensors;
  output_tensors.resize(num_outputs);
  auto allocate_output_fn = [&](int i, const TensorShape& output_slice_shape,
                                Tensor** tensor) {
    if (i < 0 || i >= output_tensors.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Index ", i, " out of range [0, ", output_tensors.size(), "]"));
    }
    output_tensors[i] = Tensor(tensorflow::DT_INT32, output_slice_shape);
    *tensor = &output_tensors[i];
    return absl::OkStatus();
  };
  auto assign_or_copy_value_fn = [&](const Tensor& input) -> Status {
    output_tensors[0] = input;
    return absl::OkStatus();
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto splitter, (XlaNDSplitter<Eigen::ThreadPoolDevice, int32_t>::Create(
                         num_splits, num_outputs, paddings,
                         /*has_paddings=*/false)));
  TF_ASSERT_OK(splitter.Split(&input_tensor, "test", assign_or_copy_value_fn,
                              allocate_output_fn, device));

  ASSERT_EQ(output_tensors.size(), 1);
  test::ExpectTensorEqual<int32_t>(
      output_tensors[0], test::AsTensor<int32_t>({0, 1, 2, 3, 4, 5, 6, 7},
                                                 TensorShape({2, 2, 2})));
}

TEST(XlaNDSplitterTest, NoSplitsWithPadding) {
  auto device = CreateThreadPoolDevice();

  const TensorShape input_shape({2, 1, 1});
  const std::vector<int32_t> num_splits = {1, 1, 1};
  const std::vector<int> paddings = {0, 1, 1};
  const int num_outputs = 1;
  auto input_tensor = test::AsTensor<int32_t>({0, 1}, input_shape);

  std::vector<Tensor> output_tensors;
  output_tensors.resize(num_outputs);
  auto allocate_output_fn = [&](int i, const TensorShape& output_slice_shape,
                                Tensor** tensor) {
    if (i < 0 || i >= output_tensors.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Index ", i, " out of range [0, ", output_tensors.size(), "]"));
    }
    output_tensors[i] = Tensor(tensorflow::DT_INT32, output_slice_shape);
    *tensor = &output_tensors[i];
    return absl::OkStatus();
  };
  auto assign_or_copy_value_fn = [&](const Tensor& input) -> Status {
    output_tensors[0] = input;
    return absl::OkStatus();
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto splitter, (XlaNDSplitter<Eigen::ThreadPoolDevice, int32_t>::Create(
                         num_splits, num_outputs, paddings,
                         /*has_paddings=*/true)));

  TF_ASSERT_OK(splitter.Split(&input_tensor, "test", assign_or_copy_value_fn,
                              allocate_output_fn, device));

  ASSERT_EQ(output_tensors.size(), 1);
  std::vector<int32_t> expected_values(3 * 3 * 3);
  test::ExpectTensorEqual<int32_t>(
      output_tensors[0], test::AsTensor<int32_t>({0, 0, 0, 0, 1, 0, 0, 0},
                                                 TensorShape({2, 2, 2})));
}

TEST(XlaNDSplitterTest, SplitNoPadding) {
  auto device = CreateThreadPoolDevice();

  const TensorShape input_shape({4, 4});
  const std::vector<int32_t> num_splits = {2, 2};
  const std::vector<int32_t> paddings(num_splits.size(), 0);
  const int num_outputs = 4;
  auto input_tensor = test::AsTensor<int32_t>(
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, input_shape);

  std::vector<Tensor> output_tensors;
  output_tensors.resize(num_outputs);
  auto allocate_output_fn = [&](int i, const TensorShape& output_slice_shape,
                                Tensor** tensor) {
    if (i < 0 || i >= output_tensors.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Index ", i, " out of range [0, ", output_tensors.size(), "]"));
    }
    output_tensors[i] = Tensor(tensorflow::DT_INT32, output_slice_shape);
    *tensor = &output_tensors[i];
    return absl::OkStatus();
  };
  auto assign_or_copy_value_fn = [&](const Tensor& input) -> Status {
    output_tensors[0] = input;
    return absl::OkStatus();
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto splitter, (XlaNDSplitter<Eigen::ThreadPoolDevice, int32_t>::Create(
                         num_splits, num_outputs, paddings,
                         /*has_paddings=*/true)));

  TF_ASSERT_OK(splitter.Split(&input_tensor, "test", assign_or_copy_value_fn,
                              allocate_output_fn, device));

  ASSERT_EQ(output_tensors.size(), num_outputs);
  test::ExpectTensorEqual<int32_t>(
      output_tensors[0],
      test::AsTensor<int32_t>({0, 1, 4, 5}, TensorShape({2, 2})));
  test::ExpectTensorEqual<int32_t>(
      output_tensors[1],
      test::AsTensor<int32_t>({2, 3, 6, 7}, TensorShape({2, 2})));
  test::ExpectTensorEqual<int32_t>(
      output_tensors[2],
      test::AsTensor<int32_t>({8, 9, 12, 13}, TensorShape({2, 2})));
  test::ExpectTensorEqual<int32_t>(
      output_tensors[3],
      test::AsTensor<int32_t>({10, 11, 14, 15}, TensorShape({2, 2})));
}

TEST(XlaNDSplitterTest, SplitPartialPadding) {
  auto device = CreateThreadPoolDevice();

  const TensorShape input_shape({3, 3});
  const std::vector<int32_t> num_splits = {2, 2};
  const std::vector<int32_t> paddings = {1, 1};
  const int num_outputs = 4;
  auto input_tensor =
      test::AsTensor<int32_t>({0, 1, 2, 3, 4, 5, 6, 7, 8}, input_shape);

  std::vector<Tensor> output_tensors;
  output_tensors.resize(num_outputs);
  auto allocate_output_fn = [&](int i, const TensorShape& output_slice_shape,
                                Tensor** tensor) {
    if (i < 0 || i >= output_tensors.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Index ", i, " out of range [0, ", output_tensors.size(), "]"));
    }
    output_tensors[i] = Tensor(tensorflow::DT_INT32, output_slice_shape);
    *tensor = &output_tensors[i];
    return absl::OkStatus();
  };
  auto assign_or_copy_value_fn = [&](const Tensor& input) -> Status {
    output_tensors[0] = input;
    return absl::OkStatus();
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto splitter, (XlaNDSplitter<Eigen::ThreadPoolDevice, int32_t>::Create(
                         num_splits, num_outputs, paddings,
                         /*has_paddings=*/true)));

  TF_ASSERT_OK(splitter.Split(&input_tensor, "test", assign_or_copy_value_fn,
                              allocate_output_fn, device));

  ASSERT_EQ(output_tensors.size(), num_outputs);
  test::ExpectTensorEqual<int32_t>(
      output_tensors[0],
      test::AsTensor<int32_t>({0, 1, 3, 4}, TensorShape({2, 2})));
  test::ExpectTensorEqual<int32_t>(
      output_tensors[1],
      test::AsTensor<int32_t>({2, 0, 5, 0}, TensorShape({2, 2})));
  test::ExpectTensorEqual<int32_t>(
      output_tensors[2],
      test::AsTensor<int32_t>({6, 7, 0, 0}, TensorShape({2, 2})));
  test::ExpectTensorEqual<int32_t>(
      output_tensors[3],
      test::AsTensor<int32_t>({8, 0, 0, 0}, TensorShape({2, 2})));
}

TEST(XlaNDSplitterTest, SplitCompletePadding) {
  auto device = CreateThreadPoolDevice();

  const TensorShape input_shape({2, 1});
  const std::vector<int32_t> num_splits = {2, 2};
  const std::vector<int32_t> paddings = {2, 3};
  const int num_outputs = 4;
  auto input_tensor = test::AsTensor<int32_t>({0, 1}, input_shape);

  std::vector<Tensor> output_tensors;
  output_tensors.resize(num_outputs);
  auto allocate_output_fn = [&](int i, const TensorShape& output_slice_shape,
                                Tensor** tensor) {
    if (i < 0 || i >= output_tensors.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Index ", i, " out of range [0, ", output_tensors.size(), "]"));
    }
    output_tensors[i] = Tensor(tensorflow::DT_INT32, output_slice_shape);
    *tensor = &output_tensors[i];
    return absl::OkStatus();
  };
  auto assign_or_copy_value_fn = [&](const Tensor& input) -> Status {
    output_tensors[0] = input;
    return absl::OkStatus();
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto splitter, (XlaNDSplitter<Eigen::ThreadPoolDevice, int32_t>::Create(
                         num_splits, num_outputs, paddings,
                         /*has_paddings=*/true)));

  TF_ASSERT_OK(splitter.Split(&input_tensor, "test", assign_or_copy_value_fn,
                              allocate_output_fn, device));

  ASSERT_EQ(output_tensors.size(), num_outputs);
  test::ExpectTensorEqual<int32_t>(
      output_tensors[0],
      test::AsTensor<int32_t>({0, 0, 1, 0}, TensorShape({2, 2})));
  test::ExpectTensorEqual<int32_t>(
      output_tensors[1],
      test::AsTensor<int32_t>({0, 0, 0, 0}, TensorShape({2, 2})));
  test::ExpectTensorEqual<int32_t>(
      output_tensors[2],
      test::AsTensor<int32_t>({0, 0, 0, 0}, TensorShape({2, 2})));
  test::ExpectTensorEqual<int32_t>(
      output_tensors[3],
      test::AsTensor<int32_t>({0, 0, 0, 0}, TensorShape({2, 2})));
}

TEST(XlaNDConcatenatorTest, NoConcats) {
  auto device = CreateThreadPoolDevice();

  const TensorShape input_shape({2, 2, 2});
  const TensorShape output_shape({2, 2, 2});
  const std::vector<int32_t> num_concats = {1, 1, 1};
  const std::vector<int> paddings(num_concats.size(), 0);
  int num_slices = 1;
  auto tensor0 = test::AsTensor<int32_t>({0, 1, 2, 3, 4, 5, 6, 7}, input_shape);
  std::vector<Tensor> input_tensors;
  input_tensors.push_back(tensor0);

  std::vector<Tensor> output_tensors;
  output_tensors.reserve(1);
  auto get_output_fn = [&]() {
    output_tensors.push_back(Tensor(tensorflow::DT_INT32, output_shape));
    return &output_tensors.back();
  };
  auto assign_or_copy_value_fn = [&](const Tensor& input) -> Status {
    output_tensors.push_back(input);
    return absl::OkStatus();
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto concatenator,
      (XlaNDConcatenator<Eigen::ThreadPoolDevice, int32_t>::Create(
          num_concats, num_slices, paddings,
          /*has_paddings=*/true)));

  TF_ASSERT_OK(concatenator.ComputeInternal(absl::MakeSpan(input_tensors),
                                            assign_or_copy_value_fn,
                                            get_output_fn, device));

  ASSERT_EQ(output_tensors.size(), 1);
  test::ExpectTensorEqual<int32_t>(
      output_tensors[0], test::AsTensor<int32_t>({0, 1, 2, 3, 4, 5, 6, 7},
                                                 TensorShape({2, 2, 2})));
}

TEST(XlaNDConcatenatorTest, ConcatNoPadding) {
  auto device = CreateThreadPoolDevice();

  const TensorShape input_shape({2, 2});
  const TensorShape output_shape({4, 4});
  const std::vector<int32_t> num_concats = {2, 2};
  const std::vector<int> paddings(num_concats.size(), 0);
  int num_slices = 4;
  auto tensor0 = test::AsTensor<int32_t>({0, 1, 2, 3}, input_shape);
  auto tensor1 = test::AsTensor<int32_t>({4, 5, 6, 7}, input_shape);
  auto tensor2 = test::AsTensor<int32_t>({8, 9, 10, 11}, input_shape);
  auto tensor3 = test::AsTensor<int32_t>({12, 13, 14, 15}, input_shape);
  std::vector<Tensor> input_tensors;
  input_tensors.push_back(tensor0);
  input_tensors.push_back(tensor1);
  input_tensors.push_back(tensor2);
  input_tensors.push_back(tensor3);

  std::vector<Tensor> output_tensors;
  output_tensors.reserve(1);
  auto get_output_fn = [&]() {
    output_tensors.push_back(Tensor(tensorflow::DT_INT32, output_shape));
    return &output_tensors.back();
  };
  auto assign_or_copy_value_fn = [&](const Tensor& input) -> Status {
    output_tensors.push_back(input);
    return absl::OkStatus();
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto concatenator,
      (XlaNDConcatenator<Eigen::ThreadPoolDevice, int32_t>::Create(
          num_concats, num_slices, paddings,
          /*has_paddings=*/true)));

  TF_ASSERT_OK(concatenator.ComputeInternal(absl::MakeSpan(input_tensors),
                                            assign_or_copy_value_fn,
                                            get_output_fn, device));
  ASSERT_EQ(output_tensors.size(), 1);
  test::ExpectTensorEqual<int32_t>(
      output_tensors[0], test::AsTensor<int32_t>({0, 1, 4, 5, 2, 3, 6, 7, 8, 9,
                                                  12, 13, 10, 11, 14, 15},
                                                 TensorShape({4, 4})));
}

TEST(XlaNDConcatenatorTest, ConcatPartialPadding) {
  auto device = CreateThreadPoolDevice();

  const TensorShape input_shape({2, 2});
  const TensorShape output_shape({3, 3});
  const std::vector<int32_t> num_concats = {2, 2};
  const std::vector<int> paddings = {1, 1};
  int num_slices = 4;
  auto tensor0 = test::AsTensor<int32_t>({0, 1, 2, 3}, input_shape);
  auto tensor1 = test::AsTensor<int32_t>({4, 5, 6, 7}, input_shape);
  auto tensor2 = test::AsTensor<int32_t>({8, 9, 10, 11}, input_shape);
  auto tensor3 = test::AsTensor<int32_t>({12, 13, 14, 15}, input_shape);
  std::vector<Tensor> input_tensors;
  input_tensors.push_back(tensor0);
  input_tensors.push_back(tensor1);
  input_tensors.push_back(tensor2);
  input_tensors.push_back(tensor3);

  std::vector<Tensor> output_tensors;
  output_tensors.reserve(1);
  auto get_output_fn = [&]() {
    output_tensors.push_back(Tensor(tensorflow::DT_INT32, output_shape));
    return &output_tensors.back();
  };
  auto assign_or_copy_value_fn = [&](const Tensor& input) -> Status {
    output_tensors.push_back(input);
    return absl::OkStatus();
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto concatenator,
      (XlaNDConcatenator<Eigen::ThreadPoolDevice, int32_t>::Create(
          num_concats, num_slices, paddings,
          /*has_paddings=*/true)));

  TF_ASSERT_OK(concatenator.ComputeInternal(absl::MakeSpan(input_tensors),
                                            assign_or_copy_value_fn,
                                            get_output_fn, device));

  ASSERT_EQ(output_tensors.size(), 1);
  test::ExpectTensorEqual<int32_t>(
      output_tensors[0], test::AsTensor<int32_t>({0, 1, 4, 2, 3, 6, 8, 9, 12},
                                                 TensorShape({3, 3})));
}

TEST(XlaNDConcatenatorTest, ConcatCompletePadding) {
  auto device = CreateThreadPoolDevice();

  const TensorShape input_shape({2, 2});
  const TensorShape output_shape({2, 2});
  const std::vector<int32_t> num_concats = {2, 2};
  const std::vector<int> paddings = {2, 2};
  int num_slices = 4;
  auto tensor0 = test::AsTensor<int32_t>({0, 1, 2, 3}, input_shape);
  auto tensor1 = test::AsTensor<int32_t>({4, 5, 6, 7}, input_shape);
  auto tensor2 = test::AsTensor<int32_t>({8, 9, 10, 11}, input_shape);
  auto tensor3 = test::AsTensor<int32_t>({12, 13, 14, 15}, input_shape);
  std::vector<Tensor> input_tensors;
  input_tensors.push_back(tensor0);
  input_tensors.push_back(tensor1);
  input_tensors.push_back(tensor2);
  input_tensors.push_back(tensor3);

  std::vector<Tensor> output_tensors;
  output_tensors.reserve(1);
  auto get_output_fn = [&]() {
    output_tensors.push_back(Tensor(tensorflow::DT_INT32, output_shape));
    return &output_tensors.back();
  };
  auto assign_or_copy_value_fn = [&](const Tensor& input) -> Status {
    output_tensors.push_back(input);
    return absl::OkStatus();
  };

  TF_ASSERT_OK_AND_ASSIGN(
      auto concatenator,
      (XlaNDConcatenator<Eigen::ThreadPoolDevice, int32_t>::Create(
          num_concats, num_slices, paddings,
          /*has_paddings=*/true)));

  TF_ASSERT_OK(concatenator.ComputeInternal(absl::MakeSpan(input_tensors),
                                            assign_or_copy_value_fn,
                                            get_output_fn, device));

  ASSERT_EQ(output_tensors.size(), 1);
  test::ExpectTensorEqual<int32_t>(
      output_tensors[0],
      test::AsTensor<int32_t>({0, 1, 2, 3}, TensorShape({2, 2})));
}

}  // namespace
}  // namespace tensorflow
