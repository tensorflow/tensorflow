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
#include "tensorflow/core/kernels/data/split_utils.h"

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/data/dataset_test_base.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace {
std::string full_name(const std::string& name) {
  return FullName("test", name);
}

Status SaveAndRestore(SplitProvider* split_provider) {
  VariantTensorDataWriter writer;
  TF_RETURN_IF_ERROR(split_provider->Save(full_name, &writer));
  std::vector<const VariantTensorData*> variants;
  writer.GetData(&variants);
  VariantTensorDataReader reader(variants);
  TF_RETURN_IF_ERROR(split_provider->Restore(full_name, &reader));
  return Status::OK();
}

Status CheckOutput(SplitProvider* split_provider,
                   std::vector<Tensor> expected) {
  int64 next = 0;
  bool end_of_splits = false;
  while (!end_of_splits) {
    Tensor split;
    TF_RETURN_IF_ERROR(split_provider->GetNext(&split, &end_of_splits));
    if (!end_of_splits) {
      test::ExpectEqual(split, expected[next++]);
    }
  }
  EXPECT_EQ(next, expected.size());
  return Status::OK();
}

TEST(IndexSplitProviderTest, Empty) {
  IndexSplitProvider split_provider(0);
  TF_EXPECT_OK(
      CheckOutput(&split_provider, CreateTensors<int64>(TensorShape({}), {})));
}

TEST(IndexSplitProviderTest, One) {
  IndexSplitProvider split_provider(1);
  TF_EXPECT_OK(CheckOutput(&split_provider,
                           CreateTensors<int64>(TensorShape({}), {{0}})));
}

TEST(IndexSplitProviderTest, Three) {
  IndexSplitProvider split_provider(3);
  TF_EXPECT_OK(CheckOutput(
      &split_provider, CreateTensors<int64>(TensorShape({}), {{0}, {1}, {2}})));
}

TEST(IndexSplitProviderTest, SaveAndRestore) {
  IndexSplitProvider split_provider(4);
  std::vector<Tensor> expected =
      CreateTensors<int64>(TensorShape({}), {{0}, {1}, {2}, {3}});
  for (int i = 0; i < expected.size(); ++i) {
    TF_ASSERT_OK(SaveAndRestore(&split_provider));
    Tensor split;
    bool end_of_splits = true;
    TF_ASSERT_OK(split_provider.GetNext(&split, &end_of_splits));
    EXPECT_FALSE(end_of_splits);
    test::ExpectEqual(split, expected[i]);
  }
  TF_ASSERT_OK(SaveAndRestore(&split_provider));
  Tensor split;
  bool end_of_splits = false;
  TF_ASSERT_OK(split_provider.GetNext(&split, &end_of_splits));
  EXPECT_TRUE(end_of_splits);
}

TEST(ShardingSplitProviderTest, TwoWayShardZero) {
  auto base = std::make_shared<IndexSplitProvider>(4);
  ShardingSplitProvider split_provider(2, 0, base);
  TF_EXPECT_OK(CheckOutput(&split_provider,
                           CreateTensors<int64>(TensorShape({}), {{0}, {2}})));
}

TEST(ShardingSplitProviderTest, TwoWayShardOne) {
  auto base = std::make_shared<IndexSplitProvider>(4);
  ShardingSplitProvider split_provider(2, 1, base);
  TF_EXPECT_OK(CheckOutput(&split_provider,
                           CreateTensors<int64>(TensorShape({}), {{1}, {3}})));
}

TEST(ShardingSplitProviderTest, ThreeWayShardOne) {
  auto base = std::make_shared<IndexSplitProvider>(6);
  ShardingSplitProvider split_provider(3, 1, base);
  TF_EXPECT_OK(CheckOutput(&split_provider,
                           CreateTensors<int64>(TensorShape({}), {{1}, {4}})));
}

TEST(ShardingSplitProviderTest, Empty) {
  auto base = std::make_shared<IndexSplitProvider>(1);
  ShardingSplitProvider split_provider(2, 1, base);
  TF_EXPECT_OK(
      CheckOutput(&split_provider, CreateTensors<int64>(TensorShape({}), {})));
}

TEST(ShardingSplitProviderTest, SaveAndRestore) {
  auto base = std::make_shared<IndexSplitProvider>(6);
  std::vector<Tensor> expected =
      CreateTensors<int64>(TensorShape({}), {{1}, {4}});
  ShardingSplitProvider split_provider(3, 1, base);
  for (int i = 0; i < expected.size(); ++i) {
    TF_ASSERT_OK(SaveAndRestore(&split_provider));
    Tensor split;
    bool end_of_splits = true;
    TF_ASSERT_OK(split_provider.GetNext(&split, &end_of_splits));
    EXPECT_FALSE(end_of_splits);
    test::ExpectEqual(split, expected[i]);
  }
  TF_ASSERT_OK(SaveAndRestore(&split_provider));
  Tensor split;
  bool end_of_splits = false;
  TF_ASSERT_OK(split_provider.GetNext(&split, &end_of_splits));
  EXPECT_TRUE(end_of_splits);
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
