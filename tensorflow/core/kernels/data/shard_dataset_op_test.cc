/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
#include "tensorflow/core/kernels/data/shard_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "shard_dataset";

class ShardDatasetParams : public DatasetParams {
 public:
  template <typename T>
  ShardDatasetParams(T input_dataset_params, int64 num_shards, int64 index,
                     bool require_non_empty, DataTypeVector output_dtypes,
                     std::vector<PartialTensorShape> output_shapes,
                     string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        num_shards_(num_shards),
        index_(index),
        require_non_empty_(require_non_empty) {
    input_dataset_params_.push_back(absl::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    return CreateTensors<int64>(TensorShape({}), {{num_shards_}, {index_}});
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->clear();
    input_names->emplace_back(ShardDatasetOp::kInputDataset);
    input_names->emplace_back(ShardDatasetOp::kNumShards);
    input_names->emplace_back(ShardDatasetOp::kIndex);
    return Status::OK();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    attr_vector->clear();
    attr_vector->emplace_back(ShardDatasetOp::kRequireNonEmpty,
                              require_non_empty_);
    attr_vector->emplace_back(ShardDatasetOp::kOutputTypes, output_dtypes_);
    attr_vector->emplace_back(ShardDatasetOp::kOutputShapes, output_shapes_);
    return Status::OK();
  }

  string dataset_type() const override { return ShardDatasetOp::kDatasetType; }

 private:
  int64 num_shards_;
  int64 index_;
  bool require_non_empty_;
};

class ShardDatasetOpTest : public DatasetOpsTestBase {};

// Test Case 1: simple case.
ShardDatasetParams ShardDatasetParams1() {
  return ShardDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*num_shards=*/5,
                            /*index=*/2,
                            /*require_non_empty=*/true,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({})},
                            /*node_name=*/kNodeName);
}

// Test Case 2: zero offset.
ShardDatasetParams ShardDatasetParams2() {
  return ShardDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*num_shards=*/5,
                            /*index=*/0,
                            /*require_non_empty=*/true,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({})},
                            /*node_name=*/kNodeName);
}

// Test Case 3: iterator ends before first element.
ShardDatasetParams ShardDatasetParams3() {
  return ShardDatasetParams(RangeDatasetParams(0, 1, 1),
                            /*num_shards=*/5,
                            /*index=*/2,
                            /*require_non_empty=*/true,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({})},
                            /*node_name=*/kNodeName);
}

// Test Case 4: larger num_shards.
ShardDatasetParams ShardDatasetParams4() {
  return ShardDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*num_shards=*/7,
                            /*index=*/5,
                            /*require_non_empty=*/true,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({})},
                            /*node_name=*/kNodeName);
}

// Test Case 5: index == num_shards.
ShardDatasetParams ShardDatasetParams5() {
  return ShardDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*num_shards=*/5,
                            /*index=*/4,
                            /*require_non_empty=*/true,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({})},
                            /*node_name=*/kNodeName);
}

// Test Case 6: similar with test_case_5 but the number of outputs could not be
// divided evenly by num_shards.
ShardDatasetParams ShardDatasetParams6() {
  return ShardDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*num_shards=*/4,
                            /*index=*/3,
                            /*require_non_empty=*/true,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({})},
                            /*node_name=*/kNodeName);
}

// Test Case 7: num_shard is larger than the cardinality of input dataset;
// require_non_empty = false.
ShardDatasetParams ShardDatasetParams7() {
  return ShardDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*num_shards=*/20,
                            /*index=*/5,
                            /*require_non_empty=*/false,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({})},
                            /*node_name=*/kNodeName);
}

// Test Case 8: similar with test_case_7 but require_non_empty = true.
ShardDatasetParams InvalidShardDatasetParamsWithNoElemForEachShard() {
  return ShardDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*num_shards=*/20,
                            /*index=*/5,
                            /*require_non_empty=*/true,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({})},
                            /*node_name=*/kNodeName);
}

// Test Case 9: index is greater than the number of shards.
ShardDatasetParams InvalidShardDatasetParams1() {
  return ShardDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*num_shards=*/5,
                            /*index=*/7,
                            /*require_non_empty=*/false,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({})},
                            /*node_name=*/kNodeName);
}

// Test Case 10: negative index.
ShardDatasetParams InvalidShardDatasetParams2() {
  return ShardDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*num_shards=*/5,
                            /*index=*/-3,
                            /*require_non_empty=*/true,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({})},
                            /*node_name=*/kNodeName);
}

// Test Case 11: negative number of shards.
ShardDatasetParams InvalidShardDatasetParams3() {
  return ShardDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*num_shards=*/-3,
                            /*index=*/1,
                            /*require_non_empty=*/true,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({})},
                            /*node_name=*/kNodeName);
}

// Test Case 12: zero number of shards.
ShardDatasetParams InvalidShardDatasetParams4() {
  return ShardDatasetParams(RangeDatasetParams(0, 10, 1),
                            /*num_shards=*/0,
                            /*index=*/1,
                            /*require_non_empty=*/true,
                            /*output_dtypes=*/{DT_INT64},
                            /*output_shapes=*/{PartialTensorShape({})},
                            /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<ShardDatasetParams>> GetNextTestCases() {
  return {
      {/*dataset_params=*/ShardDatasetParams1(),
       /*expected_outputs=*/CreateTensors<int64>(TensorShape{}, {{2}, {7}})},
      {/*dataset_params=*/ShardDatasetParams2(),
       /*expected_outputs=*/CreateTensors<int64>(TensorShape{}, {{0}, {5}})},
      {/*dataset_params=*/ShardDatasetParams3(),
       /*expected_outputs=*/{}},
      {/*dataset_params=*/ShardDatasetParams4(),
       /*expected_outputs=*/CreateTensors<int64>(TensorShape{}, {{5}})},
      {/*dataset_params=*/ShardDatasetParams5(),
       /*expected_outputs=*/CreateTensors<int64>(TensorShape{}, {{4}, {9}})},
      {/*dataset_params=*/ShardDatasetParams6(),
       /*expected_outputs=*/CreateTensors<int64>(TensorShape{}, {{3}, {7}})},
      {/*dataset_params=*/ShardDatasetParams7(),
       /*expected_outputs=*/CreateTensors<int64>(TensorShape{}, {{5}})}};
}

ITERATOR_GET_NEXT_TEST_P(ShardDatasetOpTest, ShardDatasetParams,
                         GetNextTestCases())

TEST_F(ShardDatasetOpTest, DatasetNodeName) {
  auto dataset_params = ShardDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(dataset_params.node_name()));
}

TEST_F(ShardDatasetOpTest, DatasetTypeString) {
  auto dataset_params = ShardDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(
      CheckDatasetTypeString(name_utils::OpName(ShardDatasetOp::kDatasetType)));
}

TEST_F(ShardDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = ShardDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

TEST_F(ShardDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = ShardDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes({PartialTensorShape({})}));
}

std::vector<CardinalityTestCase<ShardDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/ShardDatasetParams1(),
           /*expected_cardinality=*/2},
          {/*dataset_params=*/ShardDatasetParams2(),
           /*expected_cardinality=*/2},
          {/*dataset_params=*/ShardDatasetParams3(),
           /*expected_cardinality=*/0},
          {/*dataset_params=*/ShardDatasetParams4(),
           /*expected_cardinality=*/1},
          {/*dataset_params=*/ShardDatasetParams5(),
           /*expected_cardinality=*/2},
          {/*dataset_params=*/ShardDatasetParams6(),
           /*expected_cardinality=*/2},
          {/*dataset_params=*/ShardDatasetParams7(),
           /*expected_cardinality=*/1}};
}

DATASET_CARDINALITY_TEST_P(ShardDatasetOpTest, ShardDatasetParams,
                           CardinalityTestCases())

TEST_F(ShardDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = ShardDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

TEST_F(ShardDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = ShardDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes({PartialTensorShape({})}));
}

TEST_F(ShardDatasetOpTest, IteratorPrefix) {
  auto dataset_params = ShardDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      ShardDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

std::vector<IteratorSaveAndRestoreTestCase<ShardDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {
      {/*dataset_params=*/ShardDatasetParams1(),
       /*breakpoints=*/{0, 1, 5},
       /*expected_outputs=*/CreateTensors<int64>(TensorShape{}, {{2}, {7}})},
      {/*dataset_params=*/ShardDatasetParams2(),
       /*breakpoints=*/{0, 1, 5},
       /*expected_outputs=*/CreateTensors<int64>(TensorShape{}, {{0}, {5}})},
      {/*dataset_params=*/ShardDatasetParams3(),
       /*breakpoints=*/{0, 1},
       /*expected_outputs=*/{}},
      {/*dataset_params=*/ShardDatasetParams4(),
       /*breakpoints=*/{0, 5},
       /*expected_outputs=*/CreateTensors<int64>(TensorShape{}, {{5}})},
      {/*dataset_params=*/ShardDatasetParams5(),
       /*breakpoints=*/{0, 1, 5},
       /*expected_outputs=*/CreateTensors<int64>(TensorShape{}, {{4}, {9}})},
      {/*dataset_params=*/ShardDatasetParams6(),
       /*breakpoints=*/{0, 1, 5},
       /*expected_outputs=*/CreateTensors<int64>(TensorShape{}, {{3}, {7}})},
      {/*dataset_params=*/ShardDatasetParams7(),
       /*breakpoints=*/{0, 5},
       /*expected_outputs=*/CreateTensors<int64>(TensorShape{}, {{5}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(ShardDatasetOpTest, ShardDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(ShardDatasetOpTest, NoElemForEachShard) {
  auto dataset_params = InvalidShardDatasetParamsWithNoElemForEachShard();
  TF_ASSERT_OK(Initialize(dataset_params));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  EXPECT_EQ(
      iterator_->GetNext(iterator_ctx_.get(), &out_tensors, &end_of_sequence)
          .code(),
      tensorflow::error::INVALID_ARGUMENT);
}

TEST_F(ShardDatasetOpTest, InvalidArguments) {
  std::vector<ShardDatasetParams> invalid_dataset_params = {
      InvalidShardDatasetParams1(), InvalidShardDatasetParams2(),
      InvalidShardDatasetParams3(), InvalidShardDatasetParams4()};
  for (const auto& dataset_params : invalid_dataset_params) {
    EXPECT_EQ(Initialize(dataset_params).code(),
              tensorflow::error::INVALID_ARGUMENT);
  }
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
