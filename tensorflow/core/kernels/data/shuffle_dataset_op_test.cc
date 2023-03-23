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
#include "tensorflow/core/kernels/data/shuffle_dataset_op.h"

#include <string>
#include <utility>

#include "tensorflow/core/data/dataset_test_base.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/serialization_utils.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kShuffleNodeName[] = "shuffle_dataset";
constexpr char kShuffleAndRepeatNodeName[] = "shuffle_and_repeat_dataset";

class ShuffleDatasetParams : public DatasetParams {
 public:
  template <typename T>
  ShuffleDatasetParams(T input_dataset_params, int64_t buffer_size,
                       int64_t seed, int64_t seed2, int64_t count,
                       bool reshuffle_each_iteration,
                       DataTypeVector output_dtypes,
                       std::vector<PartialTensorShape> output_shapes,
                       string node_name)
      : DatasetParams(std::move(output_dtypes), std::move(output_shapes),
                      std::move(node_name)),
        buffer_size_(buffer_size),
        seed_(seed),
        seed2_(seed2),
        count_(count),
        reshuffle_each_iteration_(reshuffle_each_iteration) {
    input_dataset_params_.push_back(std::make_unique<T>(input_dataset_params));
    iterator_prefix_ =
        name_utils::IteratorPrefix(input_dataset_params.dataset_type(),
                                   input_dataset_params.iterator_prefix());
  }

  std::vector<Tensor> GetInputTensors() const override {
    std::vector<Tensor> input_tensors = {
        CreateTensor<int64_t>(TensorShape({}), {buffer_size_}),
        CreateTensor<int64_t>(TensorShape({}), {seed_}),
        CreateTensor<int64_t>(TensorShape({}), {seed2_})};
    if (count_ != 1) {
      input_tensors.emplace_back(
          CreateTensor<int64_t>(TensorShape({}), {count_}));
    }
    return input_tensors;
  }

  Status GetInputNames(std::vector<string>* input_names) const override {
    input_names->clear();
    input_names->emplace_back(ShuffleDatasetOpBase::kInputDataset);
    input_names->emplace_back(ShuffleDatasetOpBase::kBufferSize);
    input_names->emplace_back(ShuffleDatasetOpBase::kSeed);
    input_names->emplace_back(ShuffleDatasetOpBase::kSeed2);
    if (count_ != 1) {
      input_names->emplace_back(ShuffleAndRepeatDatasetOp::kCount);
    }
    return OkStatus();
  }

  Status GetAttributes(AttributeVector* attr_vector) const override {
    attr_vector->clear();
    attr_vector->emplace_back("output_types", output_dtypes_);
    attr_vector->emplace_back("output_shapes", output_shapes_);
    attr_vector->emplace_back("reshuffle_each_iteration",
                              reshuffle_each_iteration_);
    attr_vector->emplace_back("metadata", "");
    return OkStatus();
  }

  string dataset_type() const override {
    if (count_ != 1) {
      return ShuffleAndRepeatDatasetOp::kDatasetType;
    }
    return ShuffleDatasetOp::kDatasetType;
  }

  int64_t count() const { return count_; }

 private:
  int64_t buffer_size_;
  int64_t seed_;
  int64_t seed2_;
  int64_t count_;
  bool reshuffle_each_iteration_;
};

class ShuffleDatasetOpTest : public DatasetOpsTestBase {};

// Test case 1: test shuffle_dataset with reshuffle_each_iteration = false.
ShuffleDatasetParams ShuffleDatasetParams1() {
  return ShuffleDatasetParams(RangeDatasetParams(0, 10, 1),
                              /*buffer_size=*/3,
                              /*seed=*/1,
                              /*seed2=*/2,
                              /*count=*/1,
                              /*reshuffle_each_iteration=*/false,
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/kShuffleNodeName);
}

// Test case 2: test shuffle_dataset with reshuffle_each_iteration = true.
ShuffleDatasetParams ShuffleDatasetParams2() {
  return ShuffleDatasetParams(RangeDatasetParams(0, 10, 1),
                              /*buffer_size=*/10,
                              /*seed=*/1,
                              /*seed2=*/2,
                              /*count=*/1,
                              /*reshuffle_each_iteration=*/true,
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/kShuffleNodeName);
}

// Test case 3: similar with the test case 2 but a smaller buffer size than
// the input dataset.
ShuffleDatasetParams ShuffleDatasetParams3() {
  return ShuffleDatasetParams(RangeDatasetParams(0, 10, 1),
                              /*buffer_size=*/2,
                              /*seed=*/1,
                              /*seed2=*/2,
                              /*count=*/1,
                              /*reshuffle_each_iteration=*/true,
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/kShuffleNodeName);
}

// Test case 4: similar with the test case 2 but has different seeds.
ShuffleDatasetParams ShuffleDatasetParams4() {
  return ShuffleDatasetParams(RangeDatasetParams(0, 10, 1),
                              /*buffer_size=*/10,
                              /*seed=*/2,
                              /*seed2=*/2,
                              /*count=*/1,
                              /*reshuffle_each_iteration=*/true,
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/kShuffleNodeName);
}

// Test case 5: test shuffle_dataset with buffer_size = 1 &
// reshuffle_each_iteration = true.
ShuffleDatasetParams ShuffleDatasetParams5() {
  return ShuffleDatasetParams(RangeDatasetParams(0, 10, 1),
                              /*buffer_size=*/1,
                              /*seed=*/1,
                              /*seed2=*/2,
                              /*count=*/1,
                              /*reshuffle_each_iteration=*/true,
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/kShuffleNodeName);
}

// Test case 6: test shuffle_dataset with an empty input dataset.
ShuffleDatasetParams ShuffleDatasetParams6() {
  return ShuffleDatasetParams(RangeDatasetParams(0, 0, 1),
                              /*buffer_size=*/10,
                              /*seed=*/1,
                              /*seed2=*/2,
                              /*count=*/1,
                              /*reshuffle_each_iteration=*/true,
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/kShuffleNodeName);
}

// Test case 7: test shuffle_and_repeat_dataset with buffer_size = 10 &
// count = 2.
ShuffleDatasetParams ShuffleDatasetParams7() {
  return ShuffleDatasetParams(RangeDatasetParams(0, 10, 1),
                              /*buffer_size=*/10,
                              /*seed=*/1,
                              /*seed2=*/2,
                              /*count=*/2,
                              /*reshuffle_each_iteration=*/false,
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/kShuffleAndRepeatNodeName);
}

// Test case 8: test shuffle_and_repeat_dataset with buffer_size = 10 &
// count = -1
ShuffleDatasetParams ShuffleDatasetParams8() {
  return ShuffleDatasetParams(RangeDatasetParams(0, 3, 1),
                              /*buffer_size=*/10,
                              /*seed=*/1,
                              /*seed2=*/2,
                              /*count=*/-1,
                              /*reshuffle_each_iteration=*/false,
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/kShuffleAndRepeatNodeName);
}

ShuffleDatasetParams ShuffleDatasetParamsWithInvalidBufferSize() {
  return ShuffleDatasetParams(RangeDatasetParams(0, 0, 1),
                              /*buffer_size=*/-1,
                              /*seed=*/1,
                              /*seed2=*/2,
                              /*count=*/1,
                              /*reshuffle_each_iteration=*/false,
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/kShuffleNodeName);
}

ShuffleDatasetParams ShuffleAndRepeatDatasetParamsWithInvalidBufferSize() {
  return ShuffleDatasetParams(RangeDatasetParams(0, 0, 1),
                              /*buffer_size=*/-1,
                              /*seed=*/1,
                              /*seed2=*/2,
                              /*count=*/2,
                              /*reshuffle_each_iteration=*/false,
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/kShuffleAndRepeatNodeName);
}

ShuffleDatasetParams ShuffleAndRepeatDatasetParamsWithInvalidCount() {
  return ShuffleDatasetParams(RangeDatasetParams(0, 0, 1),
                              /*buffer_size=*/10,
                              /*seed=*/1,
                              /*seed2=*/2,
                              /*count=*/0,
                              /*reshuffle_each_iteration=*/false,
                              /*output_dtypes=*/{DT_INT64},
                              /*output_shapes=*/{PartialTensorShape({})},
                              /*node_name=*/kShuffleAndRepeatNodeName);
}

template <typename T>
struct GetNextTestCase {
  T dataset_params;
  std::vector<Tensor> expected_shuffle_outputs;
  std::vector<Tensor> expected_reshuffle_outputs;
};

std::vector<GetNextTestCase<ShuffleDatasetParams>> GetNextTestCases() {
  return {
      {/*dataset_params=*/ShuffleDatasetParams1(),
       /*expected_shuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}), {{2}, {3}, {0}, {5}, {6}, {4}, {7}, {8}, {9}, {1}}),
       /*expected_reshuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}),
           {{2}, {3}, {0}, {5}, {6}, {4}, {7}, {8}, {9}, {1}})},
      {/*dataset_params=*/ShuffleDatasetParams2(),
       /*expected_shuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}), {{2}, {6}, {1}, {3}, {9}, {5}, {0}, {8}, {7}, {4}}),
       /*expected_reshuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}),
           {{1}, {6}, {0}, {5}, {2}, {7}, {4}, {3}, {9}, {8}})},
      {/*dataset_params=*/ShuffleDatasetParams3(),
       /*expected_shuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}), {{0}, {2}, {1}, {3}, {5}, {6}, {4}, {7}, {8}, {9}}),
       /*expected_reshuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}),
           {{1}, {0}, {2}, {3}, {4}, {5}, {6}, {7}, {9}, {8}})},
      {/*dataset_params=*/ShuffleDatasetParams4(),
       /*expected_shuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}), {{3}, {0}, {8}, {1}, {5}, {4}, {7}, {2}, {6}, {9}}),
       /*expected_reshuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}),
           {{4}, {6}, {9}, {0}, {1}, {8}, {2}, {7}, {3}, {5}})},
      {/*dataset_params=*/ShuffleDatasetParams5(),
       /*expected_shuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}), {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}}),
       /*expected_reshuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}),
           {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
      {/*dataset_params=*/ShuffleDatasetParams6(),
       /*expected_shuffle_outputs=*/{},
       /*expected_reshuffle_outputs=*/{}},
      {/*dataset_params=*/ShuffleDatasetParams7(),
       /*expected_shuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}), {{9}, {0}, {8}, {6}, {1}, {3}, {7}, {2}, {4}, {5},
                             {9}, {0}, {8}, {6}, {1}, {3}, {7}, {2}, {4}, {5}}),
       /*expected_reshuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}),
           {{9}, {0}, {8}, {6}, {1}, {3}, {7}, {2}, {4}, {5},
            {9}, {0}, {8}, {6}, {1}, {3}, {7}, {2}, {4}, {5}})},
      {/*dataset_params=*/ShuffleDatasetParams8(),
       /*expected_shuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}),
           {{2}, {0}, {1}, {2}, {0}, {1}, {2}, {0}, {1}, {2}, {0},
            {1}, {2}, {0}, {1}, {2}, {0}, {1}, {2}, {0}, {1}}),
       /*expected_reshuffle_outputs=*/
       CreateTensors<int64_t>(
           TensorShape({}),
           {{2}, {0}, {1}, {2}, {0}, {1}, {2}, {0}, {1}, {2}, {0},
            {1}, {2}, {0}, {1}, {2}, {0}, {1}, {2}, {0}, {1}})}};
}

class ParameterizedGetNextTest : public ShuffleDatasetOpTest,
                                 public ::testing::WithParamInterface<
                                     GetNextTestCase<ShuffleDatasetParams>> {};

TEST_P(ParameterizedGetNextTest, GetNext) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  bool end_of_sequence = false;
  std::vector<Tensor> shuffled_out_tensors;
  while (!end_of_sequence) {
    std::vector<Tensor> next;
    TF_EXPECT_OK(
        iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence));
    shuffled_out_tensors.insert(shuffled_out_tensors.end(), next.begin(),
                                next.end());
    // For the forever-repeat case, we test only a finite number of steps of
    // the infinite sequence.
    if (test_case.dataset_params.count() == -1 &&
        shuffled_out_tensors.size() ==
            test_case.expected_shuffle_outputs.size()) {
      break;
    }
  }

  // Reshuffle the dataset.
  end_of_sequence = false;
  TF_ASSERT_OK(dataset_->MakeIterator(
      iterator_ctx_.get(), /*parent=*/nullptr,
      test_case.dataset_params.iterator_prefix(), &iterator_));
  std::vector<Tensor> reshuffled_out_tensors;
  while (!end_of_sequence) {
    std::vector<Tensor> next;
    TF_EXPECT_OK(
        iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence));
    reshuffled_out_tensors.insert(reshuffled_out_tensors.end(), next.begin(),
                                  next.end());
    // For the forever-repeat case, we test only a finite number of steps of
    // the infinite sequence.
    if (test_case.dataset_params.count() == -1 &&
        reshuffled_out_tensors.size() ==
            test_case.expected_shuffle_outputs.size()) {
      break;
    }
  }

  TF_EXPECT_OK(ExpectEqual(shuffled_out_tensors,
                           test_case.expected_shuffle_outputs,
                           /*compare_order=*/true));
  TF_EXPECT_OK(ExpectEqual(reshuffled_out_tensors,
                           test_case.expected_reshuffle_outputs,
                           /*compare_order=*/true));
}

INSTANTIATE_TEST_CASE_P(ShuffleDatasetOpTest, ParameterizedGetNextTest,
                        ::testing::ValuesIn(GetNextTestCases()));

std::vector<DatasetNodeNameTestCase<ShuffleDatasetParams>>
DatasetNodeNameTestCases() {
  return {{/*dataset_params=*/ShuffleDatasetParams1(),
           /*expected_node_name=*/kShuffleNodeName},
          {/*dataset_params=*/ShuffleDatasetParams7(),
           /*expected_node_name=*/kShuffleAndRepeatNodeName}};
}

DATASET_NODE_NAME_TEST_P(ShuffleDatasetOpTest, ShuffleDatasetParams,
                         DatasetNodeNameTestCases())

std::vector<DatasetTypeStringTestCase<ShuffleDatasetParams>>
DatasetTypeStringTestCases() {
  return {{/*dataset_params=*/ShuffleDatasetParams1(),
           /*expected_dataset_type_string=*/name_utils::OpName(
               ShuffleDatasetOp::kDatasetType)},
          {/*dataset_params=*/ShuffleDatasetParams7(),
           /*expected_dataset_type_string=*/
           name_utils::OpName(ShuffleAndRepeatDatasetOp::kDatasetType)}};
}

DATASET_TYPE_STRING_TEST_P(ShuffleDatasetOpTest, ShuffleDatasetParams,
                           DatasetTypeStringTestCases())

TEST_F(ShuffleDatasetOpTest, DatasetOutputDtypes) {
  auto dataset_params = ShuffleDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputDtypes({DT_INT64}));
}

TEST_F(ShuffleDatasetOpTest, DatasetOutputShapes) {
  auto dataset_params = ShuffleDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckDatasetOutputShapes(dataset_params.output_shapes()));
}

std::vector<CardinalityTestCase<ShuffleDatasetParams>> CardinalityTestCases() {
  return {{/*dataset_params=*/ShuffleDatasetParams1(),
           /*expected_cardinality=*/10},
          {/*dataset_params=*/ShuffleDatasetParams2(),
           /*expected_cardinality=*/10},
          {/*dataset_params=*/ShuffleDatasetParams3(),
           /*expected_cardinality=*/10},
          {/*dataset_params=*/ShuffleDatasetParams4(),
           /*expected_cardinality=*/10},
          {/*dataset_params=*/ShuffleDatasetParams5(),
           /*expected_cardinality=*/10},
          {/*dataset_params=*/ShuffleDatasetParams6(),
           /*expected_cardinality=*/0},
          {/*dataset_params=*/ShuffleDatasetParams7(),
           /*expected_cardinality=*/20},
          {/*dataset_params=*/ShuffleDatasetParams8(),
           /*expected_cardinality=*/kInfiniteCardinality}};
}

DATASET_CARDINALITY_TEST_P(ShuffleDatasetOpTest, ShuffleDatasetParams,
                           CardinalityTestCases())

TEST_F(ShuffleDatasetOpTest, IteratorOutputDtypes) {
  auto dataset_params = ShuffleDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputDtypes({DT_INT64}));
}

TEST_F(ShuffleDatasetOpTest, IteratorOutputShapes) {
  auto dataset_params = ShuffleDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorOutputShapes(dataset_params.output_shapes()));
}

TEST_F(ShuffleDatasetOpTest, IteratorOutputPrefix) {
  auto dataset_params = ShuffleDatasetParams1();
  TF_ASSERT_OK(Initialize(dataset_params));
  TF_ASSERT_OK(CheckIteratorPrefix(name_utils::IteratorPrefix(
      ShuffleDatasetOp::kDatasetType, dataset_params.iterator_prefix())));
}

template <typename T>
struct IteratorSaveAndRestoreTestCase {
  T dataset_params;
  std::vector<int> breakpoints;
  std::vector<Tensor> expected_shuffle_outputs;
};

std::vector<IteratorSaveAndRestoreTestCase<ShuffleDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/ShuffleDatasetParams1(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_shuffle_outputs=*/
           CreateTensors<int64_t>(
               TensorShape({}),
               {{2}, {3}, {0}, {5}, {6}, {4}, {7}, {8}, {9}, {1}})},
          {/*dataset_params=*/ShuffleDatasetParams2(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_shuffle_outputs=*/
           CreateTensors<int64_t>(
               TensorShape({}),
               {{2}, {6}, {1}, {3}, {9}, {5}, {0}, {8}, {7}, {4}})},
          {/*dataset_params=*/ShuffleDatasetParams3(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_shuffle_outputs=*/
           CreateTensors<int64_t>(
               TensorShape({}),
               {{0}, {2}, {1}, {3}, {5}, {6}, {4}, {7}, {8}, {9}})},
          {/*dataset_params=*/ShuffleDatasetParams4(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_shuffle_outputs=*/
           CreateTensors<int64_t>(
               TensorShape({}),
               {{3}, {0}, {8}, {1}, {5}, {4}, {7}, {2}, {6}, {9}})},
          {/*dataset_params=*/ShuffleDatasetParams5(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_shuffle_outputs=*/
           CreateTensors<int64_t>(
               TensorShape({}),
               {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}})},
          {/*dataset_params=*/ShuffleDatasetParams6(),
           /*breakpoints=*/{0, 4, 11},
           /*expected_shuffle_outputs=*/{}},
          {/*dataset_params=*/ShuffleDatasetParams7(),
           /*breakpoints=*/{0, 5, 22},
           /*expected_shuffle_outputs=*/
           CreateTensors<int64_t>(
               TensorShape({}),
               {{9}, {0}, {8}, {6}, {1}, {3}, {7}, {2}, {4}, {5},
                {9}, {0}, {8}, {6}, {1}, {3}, {7}, {2}, {4}, {5}})},
          {/*dataset_params=*/ShuffleDatasetParams8(),
           /*breakpoints=*/{0, 5, 20},
           /*expected_shuffle_outputs=*/
           CreateTensors<int64_t>(
               TensorShape({}),
               {{2}, {0}, {1}, {2}, {0}, {1}, {2}, {0}, {1}, {2}, {0},
                {1}, {2}, {0}, {1}, {2}, {0}, {1}, {2}, {0}, {1}})}};
}

class ParameterizedIteratorSaveAndRestoreTest
    : public ShuffleDatasetOpTest,
      public ::testing::WithParamInterface<
          IteratorSaveAndRestoreTestCase<ShuffleDatasetParams>> {};

TEST_P(ParameterizedIteratorSaveAndRestoreTest, IteratorSaveAndRestore) {
  auto test_case = GetParam();
  TF_ASSERT_OK(Initialize(test_case.dataset_params));

  std::unique_ptr<SerializationContext> serialization_ctx;
  TF_ASSERT_OK(CreateSerializationContext(&serialization_ctx));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  int cur_iteration = 0;
  const std::vector<int>& breakpoints = test_case.breakpoints;
  for (int breakpoint : breakpoints) {
    VariantTensorDataWriter writer;
    TF_EXPECT_OK(iterator_->Save(serialization_ctx.get(), &writer));
    std::vector<const VariantTensorData*> data;
    writer.GetData(&data);
    VariantTensorDataReader reader(data);
    TF_EXPECT_OK(RestoreIterator(iterator_ctx_.get(), &reader,
                                 test_case.dataset_params.iterator_prefix(),
                                 *dataset_, &iterator_));

    while (cur_iteration <= breakpoint) {
      std::vector<Tensor> next;
      TF_EXPECT_OK(
          iterator_->GetNext(iterator_ctx_.get(), &next, &end_of_sequence));
      out_tensors.insert(out_tensors.end(), next.begin(), next.end());
      cur_iteration++;
    }
  }

  TF_EXPECT_OK(ExpectEqual(out_tensors, test_case.expected_shuffle_outputs,
                           /*compare_order=*/true));
}

INSTANTIATE_TEST_CASE_P(ShuffleDatasetOpTest,
                        ParameterizedIteratorSaveAndRestoreTest,
                        ::testing::ValuesIn(IteratorSaveAndRestoreTestCases()));

TEST_F(ShuffleDatasetOpTest, InvalidArguments) {
  std::vector<ShuffleDatasetParams> dataset_params_vec(
      {ShuffleDatasetParamsWithInvalidBufferSize(),
       ShuffleAndRepeatDatasetParamsWithInvalidBufferSize(),
       ShuffleAndRepeatDatasetParamsWithInvalidCount()});
  for (const auto& dataset_params : dataset_params_vec) {
    EXPECT_EQ(Initialize(dataset_params).code(),
              absl::StatusCode::kInvalidArgument);
  }
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
