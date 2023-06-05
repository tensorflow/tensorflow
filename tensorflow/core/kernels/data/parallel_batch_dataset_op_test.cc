/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/data/parallel_batch_dataset_op.h"

#include "tensorflow/core/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "parallel_batch_dataset";
constexpr int kOpVersion = 1;

class ParallelBatchDatasetParams : public DatasetParams {
 public:
  template <typename T>
  ParallelBatchDatasetParams(T input_dataset_params, int64_t batch_size,
                             int64_t num_parallel_calls, bool drop_remainder,
                             DataTypeVector output_dtypes,
                             std::vector<PartialTensorShape> output_shapes,
                             const bool parallel_copy,
                             const std::string& deterministic, string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        batch_size_(batch_size),
        num_parallel_calls_(num_parallel_calls),
        drop_remainder_(drop_remainder),
        parallel_copy_(parallel_copy),
        deterministic_(deterministic) {
    input_dataset_params_.push_back(std::make_unique<T>(input_dataset_params));
    op_version_ = kOpVersion;
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    Tensor batch_size = CreateTensor<int64_t>(TensorShape({}), {batch_size_});
    Tensor num_parallel_calls =
        CreateTensor<int64_t>(TensorShape({}), {num_parallel_calls_});
    Tensor drop_remainder =
        CreateTensor<bool>(TensorShape({}), {drop_remainder_});
    return {batch_size, num_parallel_calls, drop_remainder};
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
    *input_names = {ParallelBatchDatasetOp::kInputDataset,
                    ParallelBatchDatasetOp::kBatchSize,
                    ParallelBatchDatasetOp::kNumParallelCalls,
                    ParallelBatchDatasetOp::kDropRemainder};
    return OkStatus();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    *attr_vector = {
        {"parallel_copy", parallel_copy_},
        {"output_types", output_dtypes_},
        {"output_shapes", output_shapes_},
        {"deterministic", deterministic_},
        {"metadata", ""},
    };
    return OkStatus();
  };

  string dataset_type() const override {
    return ParallelBatchDatasetOp::kDatasetType;
  }

 private:
  int64_t batch_size_;
  int64_t num_parallel_calls_;
  bool drop_remainder_;
  bool parallel_copy_;
  std::string deterministic_;
};

class ParallelBatchDatasetOpTest : public DatasetOpsTestBase {};

// Test Case 1: test ParallelBatchDataset with `drop_remainder` = false and a
// batch size that can evenly split the input dataset.
ParallelBatchDatasetParams ParallelBatchDatasetParams1() {
  return ParallelBatchDatasetParams(
      RangeDatasetParams(0, 12, 1),
      /*batch_size=*/4,
      /*num_parallel_calls=*/1,
      /*drop_remainder=*/false,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({4})},
      /*parallel_copy=*/false,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*node_name=*/kNodeName);
}

// Test Case 2: test ParallelBatchDataset with `drop_remainder` = true and a
// batch size that can evenly split the input dataset.
ParallelBatchDatasetParams ParallelBatchDatasetParams2() {
  return ParallelBatchDatasetParams(
      RangeDatasetParams(0, 12, 1),
      /*batch_size=*/4,
      /*num_parallel_calls=*/1,
      /*drop_remainder=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({4})},
      /*parallel_copy=*/false,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*node_name=*/kNodeName);
}

// Test Case 3: test ParallelBatchDataset with `drop_remainder` = false and a
// batch size that can not evenly split the input dataset.
ParallelBatchDatasetParams ParallelBatchDatasetParams3() {
  return ParallelBatchDatasetParams(
      RangeDatasetParams(0, 10, 1),
      /*batch_size=*/3,
      /*num_parallel_calls=*/1,
      /*drop_remainder=*/false,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({-1})},
      /*parallel_copy=*/false,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*node_name=*/kNodeName);
}

// Test Case 4: test ParallelBatchDataset with `drop_remainder` = true and a
// batch size that can not evenly split the input dataset.
ParallelBatchDatasetParams ParallelBatchDatasetParams4() {
  return ParallelBatchDatasetParams(
      RangeDatasetParams(0, 10, 1),
      /*batch_size=*/3,
      /*num_parallel_calls=*/1,
      /*drop_remainder=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({3})},
      /*parallel_copy=*/false,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*node_name=*/kNodeName);
}

// Test Case 5: test ParallelBatchDataset with `drop_remainder` = true and
// `batch_size` > the cardinality of the input dataset.
ParallelBatchDatasetParams ParallelBatchDatasetParams5() {
  return ParallelBatchDatasetParams(
      RangeDatasetParams(0, 10, 1),
      /*batch_size=*/12,
      /*num_parallel_calls=*/1,
      /*drop_remainder=*/true,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({12})},
      /*parallel_copy=*/false,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*node_name=*/kNodeName);
}

// Test Case 6: test ParallelBatchDataset with `drop_remainder` = false and
// `batch_size` > the cardinality of the input dataset.
ParallelBatchDatasetParams ParallelBatchDatasetParams6() {
  return ParallelBatchDatasetParams(
      RangeDatasetParams(0, 10, 1),
      /*batch_size=*/12,
      /*num_parallel_calls=*/1,
      /*drop_remainder=*/false,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({-1})},
      /*parallel_copy=*/false,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*node_name=*/kNodeName);
}

// Test Case 7: test ParallelBatchDataset with `drop_remainder` = false and
// the output of the input dataset is empty.
ParallelBatchDatasetParams ParallelBatchDatasetParams7() {
  return ParallelBatchDatasetParams(
      RangeDatasetParams(0, 0, 1),
      /*batch_size=*/4,
      /*num_parallel_calls=*/1,
      /*drop_remainder=*/false,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({4})},
      /*parallel_copy=*/false,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*node_name=*/kNodeName);
}

// Test Case 8: test ParallelBatchDataset with `num_parallel_calls` = 2.
ParallelBatchDatasetParams ParallelBatchDatasetParams8() {
  return ParallelBatchDatasetParams(
      RangeDatasetParams(0, 12, 1),
      /*batch_size=*/4,
      /*num_parallel_calls=*/2,
      /*drop_remainder=*/false,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({4})},
      /*parallel_copy=*/false,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*node_name=*/kNodeName);
}

// Test Case 9: test ParallelBatchDataset with `num_parallel_calls` = 4.
ParallelBatchDatasetParams ParallelBatchDatasetParams9() {
  return ParallelBatchDatasetParams(
      RangeDatasetParams(0, 12, 1),
      /*batch_size=*/4,
      /*num_parallel_calls=*/4,
      /*drop_remainder=*/false,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({4})},
      /*parallel_copy=*/false,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*node_name=*/kNodeName);
}

// Test Case 10: test ParallelBatchDataset with `parallel_copy` = true and a
// batch size that can evenly split the input dataset.
ParallelBatchDatasetParams ParallelBatchDatasetParams10() {
  return ParallelBatchDatasetParams(
      RangeDatasetParams(0, 12, 1),
      /*batch_size=*/4,
      /*num_parallel_calls=*/1,
      /*drop_remainder=*/false,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({4})},
      /*parallel_copy=*/true,
      /*deterministic=*/DeterminismPolicy::kDeterministic,
      /*node_name=*/kNodeName);
}

// Test Case 11: test ParallelBatchDataset with an invalid batch size.
ParallelBatchDatasetParams InvalidBatchSizeParallelBatchDatasetParams() {
  return ParallelBatchDatasetParams(
      RangeDatasetParams(0, 10, 1),
      /*batch_size=*/-1,
      /*num_parallel_calls=*/1,
      /*drop_remainder=*/false,
      /*output_dtypes=*/{DT_INT64},
      /*output_shapes=*/{PartialTensorShape({3})},
      /*parallel_copy=*/false,
      /*deterministic=*/DeterminismPolicy::kNondeterministic,
      /*node_name=*/kNodeName);
}

std::vector<GetNextTestCase<ParallelBatchDatasetParams>> GetNextTestCases() {
  return {
      {/*dataset_params=*/ParallelBatchDatasetParams1(),
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape({4}),
                              {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})},
      {/*dataset_params=*/ParallelBatchDatasetParams2(),
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape({4}),
                              {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})},
      {/*dataset_params=*/ParallelBatchDatasetParams3(),
       /*expected_outputs=*/
       {CreateTensor<int64_t>(TensorShape({3}), {0, 1, 2}),
        CreateTensor<int64_t>(TensorShape({3}), {3, 4, 5}),
        CreateTensor<int64_t>(TensorShape({3}), {6, 7, 8}),
        CreateTensor<int64_t>(TensorShape({1}), {9})}},
      {/*dataset_params=*/ParallelBatchDatasetParams4(),
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape({3}),
                              {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}})},
      {/*dataset_params=*/ParallelBatchDatasetParams5(),
       /*expected_outputs=*/{}},
      {/*dataset_params=*/ParallelBatchDatasetParams6(),
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape({10}),
                              {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}})},
      {/*dataset_params=*/ParallelBatchDatasetParams7(),
       /*expected_outputs=*/{}},
      {/*dataset_params=*/ParallelBatchDatasetParams8(),
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape({4}),
                              {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})},
      {/*dataset_params=*/ParallelBatchDatasetParams9(),
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape({4}),
                              {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})},
      {/*dataset_params=*/ParallelBatchDatasetParams10(),
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape({4}),
                              {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})}};
}

ITERATOR_GET_NEXT_TEST_P(ParallelBatchDatasetOpTest, ParallelBatchDatasetParams,
                         GetNextTestCases())

TEST_F(ParallelBatchDatasetOpTest, DatasetNodeName) {
  auto parallel_batch_dataset_params = ParallelBatchDatasetParams1();
  TF_ASSERT_OK(Initialize(parallel_batch_dataset_params));
  TF_ASSERT_OK(CheckDatasetNodeName(parallel_batch_dataset_params.node_name()));
}

TEST_F(ParallelBatchDatasetOpTest, DatasetTypeString) {
  auto parallel_batch_dataset_params = ParallelBatchDatasetParams1();
  TF_ASSERT_OK(Initialize(parallel_batch_dataset_params));
  name_utils::OpNameParams params;
  params.op_version = parallel_batch_dataset_params.op_version();
  TF_ASSERT_OK(CheckDatasetTypeString(
      name_utils::OpName(ParallelBatchDatasetOp::kDatasetType, params)));
}

TEST_F(ParallelBatchDatasetOpTest, DatasetOutputDtypes) {
  auto parallel_batch_dataset_params = ParallelBatchDatasetParams1();
  TF_ASSERT_OK(Initialize(parallel_batch_dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

std::vector<DatasetOutputShapesTestCase<ParallelBatchDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/ParallelBatchDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}},
          {/*dataset_params=*/ParallelBatchDatasetParams2(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}},
          {/*dataset_params=*/ParallelBatchDatasetParams3(),
           /*expected_output_shapes=*/{PartialTensorShape({-1})}},
          {/*dataset_params=*/ParallelBatchDatasetParams4(),
           /*expected_output_shapes=*/{PartialTensorShape({3})}},
          {/*dataset_params=*/ParallelBatchDatasetParams5(),
           /*expected_output_shapes=*/{PartialTensorShape({12})}},
          {/*dataset_params=*/ParallelBatchDatasetParams6(),
           /*expected_output_shapes=*/{PartialTensorShape({-1})}},
          {/*dataset_params=*/ParallelBatchDatasetParams7(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}},
          {/*dataset_params=*/ParallelBatchDatasetParams8(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}},
          {/*dataset_params=*/ParallelBatchDatasetParams9(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}},
          {/*dataset_params=*/ParallelBatchDatasetParams10(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(ParallelBatchDatasetOpTest,
                             ParallelBatchDatasetParams,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<ParallelBatchDatasetParams>>
CardinalityTestCases() {
  return {{/*dataset_params=*/ParallelBatchDatasetParams1(),
           /*expected_cardinality=*/3},
          {/*dataset_params=*/ParallelBatchDatasetParams2(),
           /*expected_cardinality=*/3},
          {/*dataset_params=*/ParallelBatchDatasetParams3(),
           /*expected_cardinality=*/4},
          {/*dataset_params=*/ParallelBatchDatasetParams4(),
           /*expected_cardinality=*/3},
          {/*dataset_params=*/ParallelBatchDatasetParams5(),
           /*expected_cardinality=*/0},
          {/*dataset_params=*/ParallelBatchDatasetParams6(),
           /*expected_cardinality=*/1},
          {/*dataset_params=*/ParallelBatchDatasetParams7(),
           /*expected_cardinality=*/0},
          {/*dataset_params=*/ParallelBatchDatasetParams8(),
           /*expected_cardinality=*/3},
          {/*dataset_params=*/ParallelBatchDatasetParams9(),
           /*expected_cardinality=*/3},
          {/*dataset_params=*/ParallelBatchDatasetParams10(),
           /*expected_cardinality=*/3}};
}

DATASET_CARDINALITY_TEST_P(ParallelBatchDatasetOpTest,
                           ParallelBatchDatasetParams, CardinalityTestCases())

TEST_F(ParallelBatchDatasetOpTest, IteratorOutputDtypes) {
  auto parallel_batch_dataset_params = ParallelBatchDatasetParams1();
  TF_ASSERT_OK(Initialize(parallel_batch_dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

std::vector<IteratorOutputShapesTestCase<ParallelBatchDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/ParallelBatchDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}},
          {/*dataset_params=*/ParallelBatchDatasetParams2(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}},
          {/*dataset_params=*/ParallelBatchDatasetParams3(),
           /*expected_output_shapes=*/{PartialTensorShape({-1})}},
          {/*dataset_params=*/ParallelBatchDatasetParams4(),
           /*expected_output_shapes=*/{PartialTensorShape({3})}},
          {/*dataset_params=*/ParallelBatchDatasetParams5(),
           /*expected_output_shapes=*/{PartialTensorShape({12})}},
          {/*dataset_params=*/ParallelBatchDatasetParams6(),
           /*expected_output_shapes=*/{PartialTensorShape({-1})}},
          {/*dataset_params=*/ParallelBatchDatasetParams7(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}},
          {/*dataset_params=*/ParallelBatchDatasetParams8(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}},
          {/*dataset_params=*/ParallelBatchDatasetParams9(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}},
          {/*dataset_params=*/ParallelBatchDatasetParams10(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(ParallelBatchDatasetOpTest,
                              ParallelBatchDatasetParams,
                              IteratorOutputShapesTestCases())

TEST_F(ParallelBatchDatasetOpTest, IteratorOutputPrefix) {
  auto parallel_batch_dataset_params = ParallelBatchDatasetParams1();
  TF_ASSERT_OK(Initialize(parallel_batch_dataset_params));
  name_utils::IteratorPrefixParams params;
  params.op_version = parallel_batch_dataset_params.op_version();
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      ParallelBatchDatasetOp::kDatasetType,
      parallel_batch_dataset_params.iterator_prefix(), params)));
}

std::vector<IteratorSaveAndRestoreTestCase<ParallelBatchDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {
      {/*dataset_params=*/ParallelBatchDatasetParams1(),
       /*breakpoints=*/{0, 1, 5},
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape({4}),
                              {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})},
      {/*dataset_params=*/ParallelBatchDatasetParams2(),
       /*breakpoints=*/{0, 1, 5},
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape({4}),
                              {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})},
      {/*dataset_params=*/ParallelBatchDatasetParams3(),
       /*breakpoints=*/{0, 1, 5},
       /*expected_outputs=*/
       {CreateTensor<int64_t>(TensorShape({3}), {0, 1, 2}),
        CreateTensor<int64_t>(TensorShape({3}), {3, 4, 5}),
        CreateTensor<int64_t>(TensorShape({3}), {6, 7, 8}),
        CreateTensor<int64_t>(TensorShape({1}), {9})}},
      {/*dataset_params=*/ParallelBatchDatasetParams4(),
       /*breakpoints=*/{0, 1, 5},
       /*expected_outputs=*/
       {CreateTensor<int64_t>(TensorShape({3}), {0, 1, 2}),
        CreateTensor<int64_t>(TensorShape({3}), {3, 4, 5}),
        CreateTensor<int64_t>(TensorShape({3}), {6, 7, 8})}},
      {/*dataset_params=*/ParallelBatchDatasetParams5(),
       /*breakpoints=*/{0, 1, 5},
       /*expected_outputs=*/{}},
      {/*dataset_params=*/ParallelBatchDatasetParams6(),
       /*breakpoints=*/{0, 1, 5},
       /*expected_outputs=*/
       {CreateTensor<int64_t>(TensorShape({10}),
                              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})}},
      {/*dataset_params=*/ParallelBatchDatasetParams7(),
       /*breakpoints=*/{0, 1, 5},
       /*expected_outputs=*/{}},
      {/*dataset_params=*/ParallelBatchDatasetParams8(),
       /*breakpoints=*/{0, 1, 5},
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape({4}),
                              {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})},
      {/*dataset_params=*/ParallelBatchDatasetParams9(),
       /*breakpoints=*/{0, 1, 5},
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape({4}),
                              {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})},
      {/*dataset_params=*/ParallelBatchDatasetParams10(),
       /*breakpoints=*/{0, 1, 5},
       /*expected_outputs=*/
       CreateTensors<int64_t>(TensorShape({4}),
                              {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(ParallelBatchDatasetOpTest,
                                 ParallelBatchDatasetParams,
                                 IteratorSaveAndRestoreTestCases())

TEST_F(ParallelBatchDatasetOpTest, InvalidParallelBatchSize) {
  auto parallel_batch_dataset_params =
      InvalidBatchSizeParallelBatchDatasetParams();
  EXPECT_EQ(Initialize(parallel_batch_dataset_params).code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
