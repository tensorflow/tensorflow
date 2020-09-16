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
#include "tensorflow/core/kernels/data/split_providers.h"

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/data/dataset_test_base.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace data {
namespace {
constexpr char kFullNameRandomHex[] = "60d899aa0d8ce4351e7c3b419e92d25b";
constexpr char kPipe[] = "|";
constexpr char kColon[] = ":";
constexpr char kSplits[] = "splits";
constexpr char kSplitsSize[] = "splits_size";

std::string full_name(const std::string& name) {
  return strings::StrCat(kFullNameRandomHex, kPipe, "test", kColon, name);
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

// A split provider that provides pre-defined splits.
class TestSplitProvider : public SplitProvider {
 public:
  explicit TestSplitProvider(std::vector<Tensor> splits) : splits_(splits) {}

  Status GetNext(Tensor* split, bool* end_of_splits) override {
    *end_of_splits = i_ >= splits_.size();
    if (*end_of_splits) {
      return Status::OK();
    }
    *split = splits_[i_++];
    return Status::OK();
  }

  Status Reset() override {
    i_ = 0;
    return Status::OK();
  }

  Status Save(std::function<std::string(std::string)> full_name,
              IteratorStateWriter* writer) override {
    TF_RETURN_IF_ERROR(
        writer->WriteScalar(full_name(kSplitsSize), splits_.size()));
    for (int i = 0; i < splits_.size(); ++i) {
      TF_RETURN_IF_ERROR(writer->WriteTensor(
          full_name(absl::StrCat(kSplits, "[", i, "]")), splits_[i]));
    }
    TF_RETURN_IF_ERROR(writer->WriteScalar(full_name("i_"), i_));
    return Status::OK();
  }

  Status Restore(std::function<std::string(std::string)> full_name,
                 IteratorStateReader* reader) override {
    int64 splits_size;
    TF_RETURN_IF_ERROR(
        reader->ReadScalar(full_name(kSplitsSize), &splits_size));
    splits_.clear();
    for (int i = 0; i < splits_size; ++i) {
      splits_.emplace_back();
      TF_RETURN_IF_ERROR(reader->ReadTensor(
          full_name(absl::StrCat(kSplits, "[", i, "]")), &splits_.back()));
    }
    TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("i_"), &i_));
    return Status::OK();
  }

 private:
  std::vector<Tensor> splits_;
  int64 i_ = 0;
};

Status CheckOutput(ShardingSplitProvider& split_provider,
                   std::vector<Tensor> expected) {
  int64 next = 0;
  bool end_of_splits = false;
  while (!end_of_splits) {
    Tensor split;
    TF_RETURN_IF_ERROR(split_provider.GetNext(&split, &end_of_splits));
    if (!end_of_splits) {
      test::ExpectEqual(split, expected[next++]);
    }
  }
  EXPECT_EQ(next, expected.size());
  return Status::OK();
}

TEST(ShardingSplitProvider, TwoWayShardZero) {
  auto base = std::make_shared<TestSplitProvider>(
      CreateTensors<int64>(TensorShape({}), {{0}, {1}, {2}, {3}}));
  ShardingSplitProvider split_provider(2, 0, base);
  TF_EXPECT_OK(CheckOutput(split_provider,
                           CreateTensors<int64>(TensorShape({}), {{0}, {2}})));
}

TEST(ShardingSplitProvider, TwoWayShardOne) {
  auto base = std::make_shared<TestSplitProvider>(
      CreateTensors<int64>(TensorShape({}), {{0}, {1}, {2}, {3}}));
  ShardingSplitProvider split_provider(2, 1, base);
  TF_EXPECT_OK(CheckOutput(split_provider,
                           CreateTensors<int64>(TensorShape({}), {{1}, {3}})));
}

TEST(ShardingSplitProvider, ThreeWayShardOne) {
  auto base = std::make_shared<TestSplitProvider>(
      CreateTensors<int64>(TensorShape({}), {{0}, {1}, {2}, {3}, {4}, {5}}));
  ShardingSplitProvider split_provider(3, 1, base);
  TF_EXPECT_OK(CheckOutput(split_provider,
                           CreateTensors<int64>(TensorShape({}), {{1}, {4}})));
}

TEST(ShardingSplitProvider, Empty) {
  auto base = std::make_shared<TestSplitProvider>(
      CreateTensors<int64>(TensorShape({}), {{0}}));
  ShardingSplitProvider split_provider(2, 1, base);
  TF_EXPECT_OK(
      CheckOutput(split_provider, CreateTensors<int64>(TensorShape({}), {})));
}

TEST(ShardingSplitProvider, SaveAndRestore) {
  auto base = std::make_shared<TestSplitProvider>(
      CreateTensors<int64>(TensorShape({}), {{0}, {1}, {2}, {3}, {4}, {5}}));
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
